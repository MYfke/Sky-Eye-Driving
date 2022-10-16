import argparse
import json
import os
import pickle
import shutil
import time
from collections import defaultdict
from datetime import datetime
from itertools import islice

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import autograd
from torch.utils.data import DataLoader

from model.mvn.datasets import little_car, utils as dataset_utils, human36m
from model.mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, \
    VolumetricCELoss
from model.mvn.models.triangulation import AlgebraicTriangulationNet
from model.mvn.utils import misc
from model.mvn.utils import cfg, vis


def parse_args():
    """
    解析参数
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path, where config file is stored 存放配置文件的路径",
                        default='./model/config/little_car/train/little_car_alg.yaml')
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done 如果设置，则只进行评估")
    parser.add_argument('--eval_dataset', type=str, default='val',
                        help="Dataset split on which evaluate. Can be 'train' and 'val' 评估的数据集拆分。可以是 'train' 和 'val")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node 节点上进程的本地等级")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility 可重复性的随机种子")

    parser.add_argument("--logdir", type=str, default="./logs",
                        help="Path, where logs will be stored 路径，将存储日志的位置")

    args = parser.parse_args()
    return args


def setup_little_car_dataloaders(config, is_train):
    """
    is_train = Ture 时，返回训练集和验证集
    is_train = False 时，返回空训练集，和正常验证集
    """
    train_dataloader = None
    if is_train:
        # train  训练集
        train_dataset = little_car.LittleCar(
            train=True,
            test=False,
            kind=config.kind,
            image_shape=config.image_shape,

            dataset_root=config.dataset.train.dataset_root,
            labels_path=config.dataset.train.labels_path,

            scale_bbox=config.dataset.train.scale_bbox,
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train,
                                                                          "ignore_cameras") else []

        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle,  # debatable 有争议的
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val  验证集
    val_dataset = little_car.LittleCar(
        train=False,
        test=True,
        kind=config.kind,
        image_shape=config.image_shape,

        dataset_root=config.dataset.val.dataset_root,
        labels_path=config.dataset.val.labels_path,

        scale_bbox=config.dataset.val.scale_bbox,
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],

        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader,



def setup_dataloaders(config, is_train=True):
    if config.dataset.kind == "little_car":
        train_dataloader, val_dataloader = setup_little_car_dataloaders(config, is_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader


def setup_experiment(config, model_name, is_train=True):
    prefix = "train_" if is_train else "eval_"

    if config.title:
        experiment_title = config.title
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}-{}'.format(experiment_title, datetime.now().strftime("%Y.%m.%d-%H：%M：%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def one_epoch(model, criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, caption='',
              master=False, experiment_dir=None, writer=None):
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    # used to turn on/off gradients
    # 用于关闭渐变
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader)  # TODO 这里可以跟回调函数结合一下
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end

                if batch is None:
                    print("Found None batch")
                    continue

                images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(
                    batch, device, config)

                keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None
                if model_type == "alg" or model_type == "ransac":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch,
                                                                                                  proj_matricies_batch,
                                                                                                  batch)  # TODO 此处向模型传入图片
                elif model_type == "vol":
                    keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(
                        images_batch, proj_matricies_batch, batch)

                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(
                    images_batch.shape[3:])
                n_joints = keypoints_3d_pred.shape[1]

                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)

                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

                # 1-view case
                if n_views == 1:
                    if config.kind == "human36m":
                        base_joint = 6
                    elif config.kind == "coco":
                        base_joint = 11

                    keypoints_3d_gt_transformed = keypoints_3d_gt.clone()
                    keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[
                                                                                            :,
                                                                                            base_joint:base_joint + 1]
                    keypoints_3d_gt = keypoints_3d_gt_transformed

                    keypoints_3d_pred_transformed = keypoints_3d_pred.clone()
                    keypoints_3d_pred_transformed[:,
                    torch.arange(n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_pred = keypoints_3d_pred_transformed

                # calculate loss 计算损失
                total_loss = 0.0
                loss = criterion(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d,
                                 keypoints_3d_binary_validity_gt)
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # volumetric ce loss
                use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt,
                                                                                      "use_volumetric_ce_loss") else False
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()

                    loss = volumetric_ce_criterion(coord_volumes_pred, volumes_pred, keypoints_3d_gt,
                                                   keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt,
                                                                             "volumetric_ce_loss_weight") else 1.0
                    total_loss += weight * loss

                metric_dict['total_loss'].append(total_loss.item())

                if is_train:
                    opt.zero_grad()
                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(
                        filter(lambda x: x[1].requires_grad, model.named_parameters())))

                    opt.step()

                # calculate metrics
                l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d,
                                       keypoints_3d_binary_validity_gt)
                metric_dict['l2'].append(l2.item())

                # base point l2
                if base_points_pred is not None:
                    base_point_l2_list = []
                    for batch_i in range(batch_size):
                        base_point_pred = base_points_pred[batch_i]

                        if config.model.kind == "coco":
                            base_point_gt = (keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d[batch_i, 12, :3]) / 2
                        elif config.model.kind == "mpii":
                            base_point_gt = keypoints_3d_gt[batch_i, 6, :3]

                        base_point_l2_list.append(torch.sqrt(torch.sum(
                            (base_point_pred * scale_keypoints_3d - base_point_gt * scale_keypoints_3d) ** 2)).item())

                    base_point_l2 = 0.0 if len(base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                    metric_dict['base_point_l2'].append(base_point_l2)

                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])

                # plot visualization
                if master:
                    if n_iters_total % config.vis_freq == 0:  # or total_l2.item() > 500.0:
                        vis_kind = config.kind
                        if (config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False):
                            vis_kind = "coco"

                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                                keypoints_3d_gt, keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                max_n_cols=10
                            )
                            writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1),
                                             global_step=n_iters_total)

                            heatmaps_vis = vis.visualize_heatmaps(
                                images_batch, heatmaps_pred,
                                kind=vis_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=10
                            )
                            writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1),
                                             global_step=n_iters_total)

                            if model_type == "vol":
                                volumes_vis = vis.visualize_volumes(
                                    images_batch, volumes_pred, proj_matricies_batch,
                                    kind=vis_kind,
                                    cuboids_batch=cuboids_pred,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=1, max_n_cols=16
                                )
                                writer.add_image(f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1),
                                                 global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0:
                        for p_name, p in model.named_parameters():
                            try:
                                writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit()

                    # dump to tensorboard per-iter loss/metric stats
                    if is_train:
                        for title, value in metric_dict.items():
                            writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
                    writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                    writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

                    n_iters_total += 1

    # calculate evaluation metrics
    if master:
        if not is_train:
            results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)
    return n_iters_total




def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))
    master = True
    config = cfg.load_config(args.config)
    device = torch.device(config.device)

    # config  配置
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    # 有三种模型，此处开始创建模型
    model = {
        # "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        # "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint, map_location='cpu')
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    # criterion  损失函数
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer  优化器
    opt = None
    if not args.eval:
        if config.model.name == "vol":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(),
                  'lr': config.opt.process_features_lr if hasattr(config.opt,
                                                                  "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(),
                  'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                 ],
                lr=config.opt.lr
            )
        else:
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    # datasets  数据集
    print("Loading data...")
    train_dataloader, val_dataloader = setup_dataloaders(config)

    # experiment  实验
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    if not args.eval:
        # train loop  训练循环
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):

            n_iters_total_train = one_epoch(model, criterion, opt, config, train_dataloader, device, epoch,
                                            n_iters_total=n_iters_total_train, is_train=True, master=master,
                                            experiment_dir=experiment_dir, writer=writer)
            n_iters_total_val = one_epoch(model, criterion, opt, config, val_dataloader, device, epoch,
                                          n_iters_total=n_iters_total_val, is_train=False, master=master,
                                          experiment_dir=experiment_dir, writer=writer)

            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            print(f"{n_iters_total_train} iters done.{n_iters_total_train} 迭代完成。")
    else:
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, device, 0, n_iters_total=0, is_train=False,
                      master=master, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, device, 0, n_iters_total=0, is_train=False,
                      master=master, experiment_dir=experiment_dir, writer=writer)

    print("Done.")


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
