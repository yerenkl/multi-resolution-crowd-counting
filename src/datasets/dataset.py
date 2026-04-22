import json
from pathlib import Path

import torch
from PIL import Image

from src.settings import settings


def nwpu_collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), list(labels)


def zoom_pairs_collate_fn(batch):
    hr_imgs, lr_imgs = zip(*batch)
    assert len(set(img.shape for img in hr_imgs)) == 1, (
        "HR images in batch have different shapes — apply a fixed-size transform or use batch_size=1"
    )
    assert len(set(img.shape for img in lr_imgs)) == 1, (
        "LR images in batch have different shapes — apply a fixed-size transform or use batch_size=1"
    )
    return torch.stack(hr_imgs, 0), torch.stack(lr_imgs, 0)


class NWPU(torch.utils.data.Dataset):
    """
    NWPU-Crowd dataset. Supports train, val, and test splits.

    transform must accept (img: PIL.Image, points: Tensor | None) and return
    (img_tensor, points). Use datasets.transforms.Compose to build one.

    train / val: returns (img_tensor, points) where points is (N, 2) float tensor.
    test:        returns (img_tensor, None) — labels are withheld (leaderboard only).
    """

    def __init__(self, split: str = "train", transform=None, image_path = None, jsons_path = None, txt_path = None):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        base = Path(settings.nwpu_dir)

        self.image_dir = Path(image_path) if image_path is not None else base / "images"
        self.json_dir = Path(jsons_path) if jsons_path is not None else base / "jsons"
        self.txt_dir = Path(txt_path) if txt_path is not None else base

        self.transform = transform
        self.labeled = split != "test"

        split_file = self.txt_dir / f"{split}.txt"
        with open(split_file, "r") as f:
            self.image_ids = [
                line.strip().split()[0]
                for line in f
                if line.strip()
            ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")

        if self.labeled:
            with open(self.json_dir / f"{image_id}.json", "r") as f:
                annotation = json.load(f)

            pts = annotation.get("points", [])
            points = (
                torch.tensor(pts, dtype=torch.float32)
                if len(pts) > 0
                else torch.zeros((0, 2), dtype=torch.float32)
            )
        else:
            points = None

        if self.transform is not None:
            img, points = self.transform(img, points)

        return img, points


class ZoomPairs(torch.utils.data.Dataset):
    """Real optical HR/LR pairs. No annotations. Returns (hr_img, lr_img)."""

    def __init__(self, hr_transform=None, lr_transform=None):
        self.root = settings.zoom_pairs_dir
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform if lr_transform is not None else hr_transform
        self.indices = sorted(
            int(p.name) for p in self.root.iterdir()
            if p.is_dir() and p.name.isdigit()
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pair_idx = self.indices[idx]
        pair_dir = self.root / str(pair_idx)
        hr_img = Image.open(pair_dir / f"{pair_idx}_hr.jpg").convert("RGB")
        lr_img = Image.open(pair_dir / f"{pair_idx}_lr.jpg").convert("RGB")

        if self.hr_transform is not None:
            hr_img, _ = self.hr_transform(hr_img, None)
        if self.lr_transform is not None:
            lr_img, _ = self.lr_transform(lr_img, None)

        return hr_img, lr_img
