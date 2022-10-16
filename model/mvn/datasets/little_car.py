import os
from collections import defaultdict

import numpy as np
import cv2

from torch.utils.data import Dataset

from model.mvn.utils.multiview import Camera
from model.mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from model.mvn.utils import cfg


class LittleCar(Dataset):
    """
        Example dataset class for multiview tasks. 多视图任务的示例数据集类。
        Adapted from the original dataset classes (human36m.py, cmupanoptic.py) 改编自原始数据集类（human36m.py、cmupanoptic.py）
    """

    def __init__(self,
                 dataset_root,
                 labels_path,
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=True,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 cuboid_side=250.0,
                 scale_bbox=1.0,
                 square_bbox=True,
                 norm_image=True,
                 kind="little_car",
                 transfer_cmu_to_human36m=True,
                 ignore_cameras=[],
                 choose_cameras=[],
                 crop=True,
                 frames_split_file=None
                 ):
        """
            example_root:
                Path to directory in dataset containing all the data 包含所有数据的数据集中目录的路径

            labels_path:
                Path to 'example-multiview-labels-{BBOX_SOURCE}bboxes.npy'

            retain_every_n_frames_in_test:
                By default, there are 159 181 frames in training set and 26 634 in test (val) set.
                With this parameter, test set frames will be evenly skipped frames so that the
                test set size is `26634 // retain_every_n_frames_test`.
                Use a value of 13 to get 2049 frames in test set.
                默认情况下，训练集中有 159 181 帧，测试（验证）集中有 26 634 帧。使用此参数，测试集帧将均匀跳过帧，
                因此测试集大小为 `26634 retain_every_n_frames_test`。使用值 13 在测试集中获得 2049 帧。

            kind:
                Keypoint format, 'cmu' (for now)

            choose_cameras:
                A list with indices of cameras to exclude (0 to 30 inclusive)
                包含要排除的摄像机索引的列表（包括 0 到 30）

            ignore_cameras:
                A list with indices of cameras to exclude (0 to 30 inclusive)
        """
        assert train or test, '`ExampleDataset` must be constructed with at least ' \
                              'one of `test=True` / `train=True`'
        assert kind in 'little_car'

        self.dataset_root = dataset_root
        self.labels_path = labels_path
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.square_bbox = square_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.crop = crop
        self.transfer_cmu_to_human36m = transfer_cmu_to_human36m

        self.labels = np.load(labels_path, allow_pickle=True).item()

        # TODO: Change according to the number of cameras the dataset has
        # 根据数据集具有的摄像机数量进行更改
        self.total_cameras = 4

        # Populate ignore cameras array
        # 填充忽略相机数组
        self.ignore_cameras = []
        self.choose_cameras = []

        for i, camera_id in enumerate(self.labels['camera_names']):
            # assert camera_id <= self.total_cameras, f"The example dataset only has {self.total_cameras} cameras in total. Please change your config file!"

            if (len(choose_cameras) <= 0) or (camera_id in choose_cameras):
                if camera_id not in ignore_cameras:
                    self.choose_cameras.append(i)

            if camera_id in ignore_cameras:
                self.ignore_cameras.append(i)

        assert len(self.choose_cameras) >= 1, "You must choose at least 1 camera!"

        self.num_keypoints = 19  # 关键点数量

        assert self.labels['table']['keypoints'].shape[1] == self.num_keypoints, "Error with keypoints in 'labels' file"

        # TODO: It is possible that you do not have pre-processed/ground 3D keypoints
        #   If you don't have them, make sure to comment out the config.dataset.[train|val].pred_results_path
        #   This is needed mainly because the volumetric algorithm needs it for building the cuboid for unprojection
        # 您可能没有预处理的地面3D关键点,如果没有，请确保注释掉config.dataset.[train | val].pred_results_path这主要是因为体积算法需要它来构建不投影的长方体

        # self.keypoints_3d_pred = None
        # if pred_results_path is not None:
        #     pred_results = np.load(pred_results_path, allow_pickle=True)
        #     keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
        #     self.keypoints_3d_pred = keypoints_3d_pred[::retain_every_n_frames_in_test]
        #
        #     assert len(self.keypoints_3d_pred) == len(self), \
        #         f"[train={train}, test={test}] {labels_path} has {len(self)} samples, but '{pred_results_path}' " + \
        #         f"has {len(self.keypoints_3d_pred)}. Are you sure you are using the correct dataset's pre-processed 3D keypoints? The algorithm needs it for building of the cuboid."
        #     # 您确定您使用的是正确数据集的预处理3D关键点吗？该算法需要它来构建长方体。

    def read_frames_split_file(self, frames_split_file=None):
        if frames_split_file is None:
            print(f"[Note] No frame split will be specified.")
            return None

        try:
            frames_split = cfg.load_config(frames_split_file)

            assert ('train' in frames_split and 'val' in frames_split)
        except FileNotFoundError:
            print(
                f"[Warning] File {frames_split_file} not found. No frame split will be specified.")
            return None
        except AssertionError:
            print(
                f"[Warning] Invalid train/val frame splits in {frames_split_file}. No frame split will be specified.")
            return None

        # Reorganise frames split
        new_dict = {}
        for d in frames_split['train']:
            for k in d.keys():
                new_dict[str(k)] = d[k]

        frames_split['train'] = new_dict

        new_dict = {}
        for d in frames_split['val']:
            for k in d.keys():
                new_dict[str(k)] = d[k]

        frames_split['val'] = new_dict

        return frames_split

    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, idx):
        # TODO: Change according to naming conventions
        # 根据命名约定更改
        sample = defaultdict(list)  # return value 返回值
        shot = self.labels['table'][idx]  # 每一个shot就是一帧
        frame_idx = shot['frame_name']  # 帧名称

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx not in self.choose_cameras or camera_idx in self.ignore_cameras:
                continue

            # load bounding box 加载边界框坐标
            left, top, right, bottom = shot['bbox_by_camera_tlbr'][camera_idx]
            bbox = (left, top, right, bottom)

            if top - bottom == 0 or left - right == 0:
                # convention: if the bbox is empty, then this view is missing
                continue

            # quare and scale the bounding boxs 正方形和缩放边界框
            if self.square_bbox:
                bbox = get_square_bbox(bbox)

            bbox = scale_bbox(bbox, self.scale_bbox)

            # TODO: Change according to dataset paths 根据数据集路径更改
            # load image
            # $DIR_ROOT/[action_NAME]/hdImgs/[VIEW_ID]/[VIEW_ID]_[FRAME_ID].jpg
            # NOTE: pad with 0s using {frame_idx:08}
            image_path = os.path.join(
                self.dataset_root, camera_name, f'{frame_idx:08}.png')
            # image_path = os.path.join(
            #     self.dataset_root, camera_name, f'{camera_name}_{frame_idx:08}.jpg')
            #TODO 有时间的话，在图片前加上相机名称
            assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
            image = cv2.imread(image_path)

            # load camera 加载相机
            shot_camera = self.labels['cameras'][camera_idx]
            retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'],   # 逆转录相机
                                   camera_name)

            if self.crop:
                # crop image 按照边界框裁剪图片
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

            if self.image_shape is not None:
                # rescale_size 重新调整大小
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)

                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            if self.norm_image:
                # 对图片进行归一化
                image = normalize_image(image)

            sample['images'].append(image)
            sample['detections'].append(bbox)
            sample['cameras'].append(retval_camera)
            # Maybe remove to save space?
            # sample['proj_matrices'].append(retval_camera.projection)

        # TODO: Can remove or modify depending on whether your dataset has ground truth
        # 根据您的数据集是否具有置信度进行删除或修改
        # 3D keypoints (with real confidences, cos CMU)

        # if 'keypoints' in shot:
        sample['keypoints_3d'] = np.array(shot['keypoints'][:self.num_keypoints])

        # build cuboid
        # 构建长方体
        # base_point = sample['keypoints_3d'][6, :3]
        # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
        # position = base_point - sides / 2
        # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

        # save sample's index
        sample['indexes'] = idx

        # # TODO: Check this? Keypoints are different
        # if self.keypoints_3d_pred is not None:
        #     sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]

        sample.default_factory = None
        return sample

    # TODO: May be able to modify/remove if your dataset doesn't have ground truth data
    # 如果您的数据集没有置信度数据，则可以修改删除
    def evaluate_using_per_pose_error(self, per_pose_error, split_by_subject):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for action_idx in range(len(self.labels['action_names'])):
                action_mask = (self.labels['table']['action_idx'] == action_idx) & mask
                action_per_pose_error = per_pose_error[action_mask]
                action_scores[self.labels['action_names'][action_idx]] = {
                    'total_loss': action_per_pose_error.sum(), 'frame_count': len(action_per_pose_error)
                }

            for k, v in action_scores.items():
                action_scores[k] = float('nan') if v['frame_count'] == 0 else (v['total_loss'] / v['frame_count'])

            return action_scores

        print("Evaluating average actions...")
        person_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }

        '''
        for person_id in range(len(self.labels['person_ids'])):
            person_mask = self.labels['table']['person_id'] == person_id
            person_scores[person_id] = \
                evaluate_by_actions(self, per_pose_error, person_mask)
        '''

        print("Evaluation complete!")

        return person_scores

    def evaluate(self, keypoints_3d_predicted, split_by_subject=False):
        keypoints_gt = self.labels['table']['keypoints'][:, :, :3]

        # Likely due to batch size problems
        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            try:
                print("Predicted shape:", keypoints_3d_predicted.shape, "GT Shape:", keypoints_gt.shape)
                keypoints_gt = keypoints_gt[:keypoints_3d_predicted.shape[0],
                               :keypoints_3d_predicted.shape[1],
                               :keypoints_3d_predicted.shape[2]]

                print(f"Forcing keypoints_gt to new shape {keypoints_gt.shape}")
            except:
                raise ValueError(
                    '`keypoints_3d_predicted` shape should be %s, got %s' % \
                    (keypoints_gt.shape, keypoints_3d_predicted.shape))

        assert keypoints_3d_predicted.shape == keypoints_gt.shape, '`keypoints_3d_predicted` shape should be %s, got %s' % \
                                                                   (keypoints_gt.shape, keypoints_3d_predicted.shape)

        # # TODO: Conversion Code
        # # TODO: Remove unnecessary 4th coordinate (confidences)
        # def remap_keypoints(keypoints, kind_from, kind_to):
        #     # Keypoint maps are in `vis.py`
        #     print(JOINT_NAMES_DICT)
        #
        #     values_from = JOINT_NAMES_DICT[kind_from].values()
        #     values_to = JOINT_NAMES_DICT[kind_to].values()
        #
        #     keypoints_new = []
        #
        #     for i, val in enumerate(values_to):
        #         keypoints_new.append(keypoint_new)
        #
        #     return keypoints_new

        # keypoints_gt = remap_keypoints(keypoints_gt, "cmu", "coco")
        # keypoints_3d_predicted = map_keypoints_cmu_to_h36m(keypoints_3d_predicted, "cmu", "coco")

        # Transfer
        if self.transfer_cmu_to_human36m:
            human36m_joints = [10, 11, 15, 14, 1, 4]
            cmu_joints = [10, 8, 9, 7, 14, 13]

            keypoints_gt = keypoints_gt[:, human36m_joints, :]
            keypoints_3d_predicted = keypoints_3d_predicted[:, cmu_joints, :]

        # mean error per 16/17 joints in mm, for each pose
        per_pose_error = np.sqrt(((keypoints_gt - keypoints_3d_predicted) ** 2).sum(2)).mean(1)
        # print(per_pose_error)

        # relative mean error per 16/17 joints in mm, for each pose
        # root_index = 6 if self.kind == "mpii" else 6
        root_index = 0

        try:
            keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
            keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:,
                                                                       root_index:root_index + 1, :]
            per_pose_error_relative = np.sqrt(
                ((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)
        except:
            print("[Warning] Cannot calculate relative mean error")
            per_pose_error_relative = per_pose_error

        result = {
            'per_pose_error': self.evaluate_using_per_pose_error(per_pose_error, split_by_subject),
            'per_pose_error_relative': self.evaluate_using_per_pose_error(per_pose_error_relative, split_by_subject)
        }

        return result['per_pose_error_relative']['Average']['Average'], result


if __name__ == '__main__':
    little_car_dataset = LittleCar()
    print(little_car_dataset)
