import torch

from src.models.clip_ebc import make_density_map


def nwpu_train_collate_fn(batch):
    images, points_list = zip(*batch)
    images = torch.stack(images)
    _, _, H, W = images.shape
    densities = torch.stack([make_density_map(p, H, W) for p in points_list])
    return images, list(points_list), densities
