"""
此代码实现检测功能
"""

import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

# from model.mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from model.mvn.models.triangulation import AlgebraicTriangulationNet

from model.mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, \
    VolumetricCELoss

from model.mvn.utils import img, multiview, op, vis, misc, cfg

# from model.mvn.datasets import human36m, cmupanoptic, example_dataset
from model.mvn.datasets import little_car

from model.mvn.datasets import utils as dataset_utils

DEBUG = False
is_train = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True,
                        help="Path, where config file is stored")

    parser.add_argument("--local_rank", type=int,
                        help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="./logs",
                        help="Path, where logs will be stored")

    args = parser.parse_args()
    return args


def setup_little_car_dataloaders(config):
    # val
    val_dataset = little_car.LittleCar(
        dataset_root=config.dataset.val.dataset_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(
            config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(
            config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(
            config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(
            config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(
            config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
        drop_last=False
    )

    return val_dataloader


def setup_dataloaders(config):
    if config.dataset.kind == 'little_car':
        val_dataloader = setup_little_car_dataloaders(config)
    else:
        raise NotImplementedError(
            "Unknown dataset: {}".format(config.dataset.kind))

    return val_dataloader


def setup_experiment(config, model_name):
    prefix = "demo_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title,
                                     datetime.now().strftime("%d.%m.%Y-%H：%M：%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def detect(model, config, dataloader, device, n_iters_total=0, master=False,
           experiment_dir=None, writer=None):
    name = "val"
    model_type = config.model.name

    model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    save_extra_data = config.save_extra_data if hasattr(
        config, "save_extra_data") else False

    if save_extra_data:
        extra_data = defaultdict(list)

    transfer_cmu_h36m = config.model.transfer_cmu_to_human36m if hasattr(
        config.model, "transfer_cmu_to_human36m") else False

    print("Transfer CMU to H36M: ", transfer_cmu_h36m)
    print("Using GT Pelvis position: ", config.model.use_gt_pelvis if hasattr(config.model, "use_gt_pelvis") else False)
    print("Using cameras: ", dataloader.dataset.choose_cameras)
    print("Debug Mode: ", DEBUG)
    train_eval_mode = "Demo"

    # no gradients as we are only testing/evaluating 没有梯度，因为我们只是在测试评估
    with torch.no_grad():
        end = time.time()

        iterator = enumerate(dataloader)

        if not is_train and config.opt.n_iters_per_epoch_val is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch_val)

        '''
        Data breakdown:
        - For each of the (max) 31 cameras in CMU dataset:
            - OpenCV Image: Numpy array [Note: likely cropped to smaller shape]
            - BBOX Detection for the image: (left, top, right, bottom) tuple
            - Camera: `Camera` object from `multiview.py`
        - Index: int
        - Keypoints (gt): NP Array, (17, 4)
        - Keypoints (pred): NP Array, (17, 4) [Note: may not be there]
        '''
        ignore_batch = []

        for iter_i, batch in iterator:  # 每4张图是一个batch，用来预测
            with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end

                if batch is None:
                    print(
                        f"[{train_eval_mode}] Found None batch: {iter_i}")
                    continue

                if DEBUG:
                    print(f"{train_eval_mode} batch {iter_i}...")
                    print(f"[{train_eval_mode}, {iter_i}] Preparing batch... ", end="")

                images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(
                    batch, device, config)

                if DEBUG:
                    print("Prepared!")

                if DEBUG:
                    print(f"[{train_eval_mode}, {iter_i}] Running {model_type} model... ", end="")

                keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None
                if model_type == "alg" or model_type == "ransac":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
                        images_batch, proj_matricies_batch, batch)
                elif model_type == "vol":
                    keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(
                        images_batch, proj_matricies_batch, batch)
                else:
                    raise NotImplementedError(
                        f"Unknown model type {model_type}")

                if DEBUG:
                    print("Done!")

                # batch shape[2] is likely to be the number of channels
                # n_views is also the number of cameras being used in this batch
                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(
                    images_batch.shape[3:])
                n_joints = keypoints_3d_pred.shape[1]

                # Due to differences in model used, it may be possible that the gt and pred keypoints have different scales
                # Set this difference in scaling in the config.yaml
                # file由于使用的模型不同，可能 gt 和 pred 关键点有不同的比例 在 config.yaml 文件中设置这个比例差异

                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(
                    config.opt, "scale_keypoints_3d") else 1.0

                # force ground truth keypoints to fit config kind
                keypoints_gt_original = keypoints_3d_gt.clone()

                # 1-view case
                # TODO: Totally remove for CMU dataset (which doesnt have pelvis-offset errors)?
                if n_views == 1:
                    print(
                        f"[{train_eval_mode}, {iter_i}] {config.kind} 1-view case: batch {iter_i}, images {images_batch.shape}")

                    if config.kind == "human36m":
                        base_joint = 6
                    elif config.kind in ["coco", "cmu", "cmupanoptic"]:
                        base_joint = 11

                    keypoints_3d_pred_transformed = keypoints_3d_pred.clone()
                    keypoints_3d_pred_transformed[:, torch.arange(
                        n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_pred = keypoints_3d_pred_transformed

                if DEBUG:
                    print("Done!")

                # calculate metrics
                if DEBUG:
                    print(
                        f"[{train_eval_mode}, {iter_i}] Calculating metrics... ", end="")

                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(
                        keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])

                    if save_extra_data:
                        extra_data['images'].append(batch['images'])
                        extra_data['detections'].append(batch['detections'])
                        extra_data['cameras'].append(batch['cameras'])

                # plot visualization
                # NOTE: transfer_cmu_h36m has a visualisation error, and connectivity dict needs to be h36m
                if master:
                    if n_iters_total % config.vis_freq == 0:  # or total_l2.item() > 500.0:
                        vis_kind = config.kind if hasattr(
                            config, "kind") else "coco"
                        pred_kind = config.pred_kind if hasattr(
                            config, "pred_kind") else None

                        if transfer_cmu_h36m and pred_kind is None:
                            pred_kind = "human36m"

                        # NOTE: Because of transfering, using original gt instead of truncated ones
                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                                None, keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                max_n_cols=10,
                                pred_kind=pred_kind
                            )
                            writer.add_image(
                                f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1),
                                global_step=n_iters_total)

                            heatmaps_vis = vis.visualize_heatmaps(
                                images_batch, heatmaps_pred,
                                kind=pred_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=10
                            )
                            writer.add_image(
                                f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1),
                                global_step=n_iters_total)

                            if model_type == "vol":
                                volumes_vis = vis.visualize_volumes(
                                    images_batch, volumes_pred, proj_matricies_batch,
                                    kind=pred_kind,
                                    cuboids_batch=cuboids_pred,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=1, max_n_cols=16
                                )
                                writer.add_image(
                                    f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1),
                                    global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0:
                        for p_name, p in model.named_parameters():
                            try:
                                writer.add_histogram(
                                    p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit()

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar(f"{name}/batch_time",
                                      batch_time, n_iters_total)
                    writer.add_scalar(f"{name}/data_time",
                                      data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar(f"{name}/batch_size",
                                      batch_size, n_iters_total)
                    writer.add_scalar(f"{name}/n_views",
                                      n_views, n_iters_total)

                    n_iters_total += 1

            if DEBUG:
                print(f"Training of batch {iter_i} complete!")

    # save results file 保存结果文件
    if master:
        checkpoint_dir = os.path.join(
            experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # dump results
        with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
            if DEBUG:
                print(
                    f"Dumping results to {checkpoint_dir}/results.pkl... ", end="")
            pickle.dump(results, fout, protocol=4)
            if DEBUG:
                print("Dumped!")

        # dump extra data as pkl file if need to reconstruct anything
        if save_extra_data:
            with open(os.path.join(checkpoint_dir, "extra_data.pkl"), 'wb') as fout:
                if DEBUG:
                    print(
                        f"Dumping extra data to {checkpoint_dir}/extra_data.pkl... ", end="")

                pickle.dump(extra_data, fout, protocol=4)

                if DEBUG:
                    print("Dumped!")

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}", np.mean(value))

    print(f"{train_eval_mode} complete!")

    return n_iters_total


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    config = cfg.load_config(args.config)

    global DEBUG
    DEBUG = config.debug_mode if hasattr(config, "debug_mode") else False
    print("Debugging Mode: ", DEBUG)

    master = True
    device = torch.device(config.device)

    # config
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    if hasattr(config.opt, "n_objects_per_epoch_val"):
        config.opt.n_iters_per_epoch_val = config.opt.n_objects_per_epoch_val // config.opt.val_batch_size
    else:
        config.opt.n_iters_per_epoch_val = None

    model = {
        # "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        # "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    print("Loading data...")
    val_dataloader = setup_dataloaders(config)

    # experiment 实验
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__)

    detect(model=model, config=config, dataloader=val_dataloader, device=device,
           experiment_dir=experiment_dir, writer=writer)

    print("Done.")


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))

    main(args)
