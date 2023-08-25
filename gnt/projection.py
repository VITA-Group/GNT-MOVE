import torch
import torch.nn.functional as F


class Projector:
    def __init__(self, device):
        self.device = device

    def inbound(self, pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (
            (pixel_locations[..., 0] <= w - 1.0)
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = (
            2 * pixel_locations / resize_factor - 1.0
        )  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        """
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)).bmm(
            xyz_h.t()[None, ...].repeat(num_views, 1, 1)
        )  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(
            projections[..., 2:3], min=1e-8
        )  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0  # a point is invalid if behind the camera
        return pixel_locations.reshape((num_views,) + original_shape + (2,)), mask.reshape(
            (num_views,) + original_shape
        )

    def compute_angle(self, xyz, query_camera, train_cameras):
        """
        :param xyz: [..., 3]
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = (
            query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)
        )  # [n_views, 4, 4]
        ray2tar_pose = query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)
        ray2tar_pose /= torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6
        ray2train_pose = train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)
        ray2train_pose /= torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))
        return ray_diff

    def compute(self, xyz, query_camera, train_imgs, train_cameras, featmaps):
        """
        :param xyz: [n_rays, n_samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, n_views, h, w, 3]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 ray_diff: [n_rays, n_samples, 4],
                 mask: [n_rays, n_samples, 1]
        """
        assert (
            (train_imgs.shape[0] == 1)
            and (train_cameras.shape[0] == 1)
            and (query_camera.shape[0] == 1)
        ), "only support batch_size=1 for now"

        train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
        query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w
        )  # [n_views, n_rays, n_samples, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat(
            [rgb_sampled, feat_sampled], dim=-1
        )  # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        ray_diff = self.compute_angle(xyz, query_camera, train_cameras)
        ray_diff = ray_diff.permute(1, 2, 0, 3)
        mask = (
            (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]
        )  # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, ray_diff, mask
