import json
import torch
from PIL import Image

from src.settings import settings


def nwpu_collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), list(labels)  # labels: list of variable-length tensors


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

    train / val: returns (img, points) where points is an (N, 2) float tensor.
    test:        returns (img, None) — labels are withheld (leaderboard only).
    """

    def __init__(self, transform=None, split: str = "train"):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.root = settings.nwpu_dir
        self.transform = transform
        self.labeled = split != "test"
        with open(self.root / f"{split}.txt") as f:
            self.image_ids = [line.strip().split()[0] for line in f if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.labeled:
            with open(self.root / "jsons" / f"{image_id}.json") as f:
                annotation = json.load(f)
            points = annotation["points"]
            labels = (torch.tensor(points, dtype=torch.float32) if points
                      else torch.zeros((0, 2), dtype=torch.float32))
        else:
            labels = None

        img = Image.open(self.root / "images" / f"{image_id}.jpg").convert("RGB")
        orig_w, orig_h = img.size

        if self.transform is not None:
            img = self.transform(img)
            if labels is not None and labels.numel() > 0:
                _, new_h, new_w = img.shape
                labels[:, 0] *= new_w / orig_w
                labels[:, 1] *= new_h / orig_h

        return img, labels


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
            hr_img = self.hr_transform(hr_img)
        if self.lr_transform is not None:
            lr_img = self.lr_transform(lr_img)
        return hr_img, lr_img


if __name__ == "__main__":
    from src.data.transforms import nwpu_train_transform, nwpu_val_transform, zoom_pairs_transform
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    out_dir = settings.RESULTS_DIR / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- NWPU train / val / test ---
    for split, transform in [
        ("train", nwpu_train_transform),
        ("val",   nwpu_val_transform),
        ("test",  nwpu_val_transform),
    ]:
        loader = DataLoader(
            NWPU(transform=transform, split=split),
            batch_size=8, shuffle=True, collate_fn=nwpu_collate_fn,
        )
        imgs, labels = next(iter(loader))

        fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=150)
        fig.suptitle(f"NWPU — {split} ({len(loader.dataset)} images)", fontsize=14)
        for ax, img, label in zip(axes.flat, imgs, labels):
            ax.imshow(img.permute(1, 2, 0))
            if label is not None and label.numel() > 0:
                ax.scatter(label[:, 0], label[:, 1], c="red", s=5)
            ax.axis("off")

        path = out_dir / f"nwpu_{split}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # --- ZoomPairs: min / p25 / p75 / max ratio ---
    dataset = ZoomPairs(hr_transform=zoom_pairs_transform)

    # Compute HR/LR width ratio for every pair
    ratios = []
    for pair_idx in dataset.indices:
        hr_path = dataset.root / str(pair_idx) / f"{pair_idx}_hr.jpg"
        lr_path = dataset.root / str(pair_idx) / f"{pair_idx}_lr.jpg"
        hr_w = Image.open(hr_path).width
        lr_w = Image.open(lr_path).width
        ratios.append((hr_w / lr_w, pair_idx))
    ratios.sort()

    n = len(ratios)
    selected = [
        ratios[0],
        ratios[round(n * 0.25)],
        ratios[round(n * 0.75)],
        ratios[-1],
    ]
    labels_pct = ["min", "p25", "p75", "max"]

    fig, axes = plt.subplots(4, 2, figsize=(8, 16), dpi=150)
    fig.suptitle(f"ZoomPairs — HR/LR ratio: min / p25 / p75 / max", fontsize=13)
    for (ax_hr, ax_lr), (ratio, pair_idx), label in zip(axes, selected, labels_pct):
        hr, lr = dataset[dataset.indices.index(pair_idx)]
        ax_hr.imshow(hr.permute(1, 2, 0))
        ax_hr.set_title(f"HR  [{label}, pair {pair_idx}, {ratio:.2f}×]", fontsize=9)
        ax_hr.axis("off")
        ax_lr.imshow(lr.permute(1, 2, 0))
        ax_lr.set_title(f"LR", fontsize=9)
        ax_lr.axis("off")

    path = out_dir / "zoom_pairs.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
