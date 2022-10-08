import torch
from torch import nn

from model.mvn.utils import op, multiview

from model.mvn.models import pose_resnet


class AlgebraicTriangulationNet(nn.Module):
    """
    代数三角测量方法
    """

    def __init__(self, config, device='cpu'):
        super().__init__()

        self.use_confidences = config.model.use_confidences

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)  # 此处骨干网络

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension  将 n_views 维度重塑为批量维度
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integral  前向骨干和整体
        if self.use_confidences:
            heatmaps, _, alg_confidences, _ = self.backbone(images)  # 向backbone中传入图片组
        else:
            heatmaps, _, _, _ = self.backbone(images)
            alg_confidences = torch.ones(batch_size * n_views, heatmaps.shape[1]).type(torch.float).to(device)

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d, heatmaps = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # reshape back 重塑背部
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])

        # norm confidences 规范置信度
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes 计算形状
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(
            heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape 增加维度keypoints_2d，因为图像形状与热图形状不相同
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate 三角测量
        try:
            keypoints_3d = multiview.triangulate_batch_of_points(
                proj_matricies, keypoints_2d,
                confidences_batch=alg_confidences
            )
        except RuntimeError as e:
            print("Error: ", e)

            # print("confidences =", confidences_batch_pred)
            print("proj_matricies = ", proj_matricies)
            # print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
            exit()

        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences