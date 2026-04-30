import torch

from src.models.clip_ebc import make_density_map


def nwpu_train_collate_fn(batch):
    images, points_list = zip(*batch)
    images = torch.stack(images)
    _, _, H, W = images.shape
    densities = torch.stack([make_density_map(p, H, W) for p in points_list])
    return images, list(points_list), densities


def nwpu_paired_train_collate_fn(batch):
    hr_images = torch.stack([sample["hr_image"] for sample in batch])
    lr_images = torch.stack([sample["lr_image"] for sample in batch])
    points_list = [sample["points"] for sample in batch]
    _, _, H, W = hr_images.shape
    densities = torch.stack([make_density_map(points, H, W) for points in points_list])

    return {
        "hr_images": hr_images,
        "lr_images": lr_images,
        "points": points_list,
        "densities": densities,
        "gt_counts": torch.tensor([sample["gt_count"] for sample in batch], dtype=torch.float32),
        "patch_weights": torch.tensor([sample["patch_weight"] for sample in batch], dtype=torch.float32),
        "hardness": torch.tensor([sample["hardness"] for sample in batch], dtype=torch.float32),
        "is_hard": torch.tensor([sample["is_hard"] for sample in batch], dtype=torch.bool),
        "down_factors": torch.tensor([sample["down_factor"] for sample in batch], dtype=torch.float32),
    }
