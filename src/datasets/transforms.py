import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, points):
        for t in self.transforms:
            img, points = t(img, points)
        return img, points


class RandomCrop:
    """Crop a random region of (scale * size) then keep as-is. Points outside are dropped."""

    def __init__(self, size: int, scale: tuple = (1.0, 2.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img, points):
        orig_w, orig_h = img.size
        crop_size = int(self.size * random.uniform(*self.scale))
        crop_size = min(crop_size, orig_w, orig_h)

        x0 = random.randint(0, max(0, orig_w - crop_size))
        y0 = random.randint(0, max(0, orig_h - crop_size))
        x1, y1 = x0 + crop_size, y0 + crop_size

        img = img.crop((x0, y0, x1, y1))

        if points is not None and points.numel() > 0:
            px, py = points[:, 0], points[:, 1]
            mask = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            points = points[mask].clone()
            if points.numel() > 0:
                points[:, 0] -= x0
                points[:, 1] -= y0

        return img, points


class ResolutionAugment:
    """Downscale then upscale to output_size, simulating a lower-resolution capture."""

    def __init__(self, min_scale: float = 1.0, max_scale: float = 4.0, output_size: int = 224):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.output_size = output_size

    def __call__(self, img, points):
        orig_w, orig_h = img.size
        down_factor = random.uniform(self.min_scale, self.max_scale)

        lr_w = max(self.output_size, int(orig_w / down_factor))
        lr_h = max(self.output_size, int(orig_h / down_factor))
        img = img.resize((lr_w, lr_h), Image.BILINEAR)
        img = img.resize((self.output_size, self.output_size), Image.BILINEAR)

        if points is not None and points.numel() > 0:
            points = points.clone()
            points[:, 0] *= self.output_size / orig_w
            points[:, 1] *= self.output_size / orig_h

        return img, points


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, points):
        if random.random() < self.p:
            w = img.width
            img = TF.hflip(img)
            if points is not None and points.numel() > 0:
                points = points.clone()
                points[:, 0] = w - points[:, 0]
        return img, points


class ToTensor:
    def __call__(self, img, points):
        return TF.to_tensor(img), points


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._norm = T.Normalize(mean=mean, std=std)

    def __call__(self, img, points):
        return self._norm(img), points
