import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


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


def _resize_points(points, scale_x: float, scale_y: float):
    if points is None or points.numel() == 0:
        return points
    points = points.clone()
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    return points


def _sample_down_factor(down_scales, min_scale, max_scale):
    if down_scales is not None:
        return float(random.choice(down_scales))
    return random.uniform(min_scale, max_scale)


def _apply_resolution_degradation(
    img,
    *,
    output_size: int,
    down_factor: float,
    method_weights,
    pre_blur: bool,
    add_noise: bool,
):
    if down_factor > 1:
        if pre_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.3)))
        method = random.choices(ResolutionAugment._DOWNSAMPLE_METHODS, weights=method_weights, k=1)[0]
        lr_size = max(1, int(round(output_size / down_factor)))
        img = img.resize((lr_size, lr_size), method)
        img = img.resize((output_size, output_size), Image.Resampling.BILINEAR)
    else:
        img = img.resize((output_size, output_size), Image.Resampling.BILINEAR)

    if add_noise:
        t = TF.to_tensor(img)
        t = (t + torch.randn_like(t) * (1.5 / 255)).clamp(0, 1)
        img = TF.to_pil_image(t)

    return img


class ResolutionAugment:
    """Downscale then upscale to output_size, simulating a lower-resolution capture.

    down_scales: discrete downscale factors to sample from (1 = native, no degradation)
    pre_blur: Gaussian blur before downscaling to simulate optical blur and avoid aliasing
    add_noise: mild Gaussian noise to simulate sensor noise at low resolutions
    """

    _DOWNSAMPLE_METHODS = [
        Image.Resampling.BILINEAR,
        Image.Resampling.BICUBIC,
        Image.Resampling.NEAREST,
        Image.Resampling.LANCZOS,
    ]

    def __init__(
        self,
        down_scales: tuple | None = (1, 2, 4, 8),
        output_size: int = 224,
        method_weights: tuple = (0.25, 0.25, 0.25, 0.25),
        pre_blur: bool = True,
        add_noise: bool = False,
        min_scale: float | None = None,
        max_scale: float | None = None,
    ):
        if down_scales is None and (min_scale is None or max_scale is None):
            raise ValueError("Provide either down_scales or both min_scale and max_scale")
        if down_scales is not None and min_scale is not None and max_scale is not None:
            raise ValueError("Use either down_scales or min_scale/max_scale, not both")

        self.down_scales = down_scales
        self.output_size = output_size
        self.method_weights = list(method_weights)
        self.pre_blur = pre_blur
        self.add_noise = add_noise
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, points):
        orig_w, orig_h = img.size
        down_factor = _sample_down_factor(self.down_scales, self.min_scale, self.max_scale)
        img = _apply_resolution_degradation(
            img,
            output_size=self.output_size,
            down_factor=down_factor,
            method_weights=self.method_weights,
            pre_blur=self.pre_blur,
            add_noise=self.add_noise,
        )
        points = _resize_points(points, self.output_size / orig_w, self.output_size / orig_h)

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


class HardPatchPairTransform:
    """Mine dense crops, keep some random crops, and return paired HR/LR tensors."""

    def __init__(
        self,
        output_size: int = 224,
        crop_scale: tuple = (1.0, 2.0),
        random_patch_prob: float = 0.25,
        num_candidates: int = 6,
        hard_weight_max: float = 4.0,
        density_temperature: float = 20.0,
        far_field_bonus: float = 0.25,
        down_scales: tuple | None = None,
        min_scale: float = 2.0,
        max_scale: float = 4.0,
        method_weights: tuple = (0.25, 0.25, 0.25, 0.25),
        pre_blur: bool = True,
        add_noise: bool = False,
        hflip_p: float = 0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        if not 0.0 <= random_patch_prob <= 1.0:
            raise ValueError("random_patch_prob must be between 0 and 1")
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        if hard_weight_max < 1.0:
            raise ValueError("hard_weight_max must be at least 1.0")
        if not 0.0 <= far_field_bonus <= 1.0:
            raise ValueError("far_field_bonus must be between 0 and 1")
        if down_scales is None and (min_scale < 1.0 or max_scale < min_scale):
            raise ValueError("min_scale/max_scale must satisfy 1 <= min_scale <= max_scale")

        self.output_size = output_size
        self.crop_scale = crop_scale
        self.hard_mining_prob = 1.0 - random_patch_prob
        self.num_candidates = num_candidates
        self.hard_weight_max = hard_weight_max
        self.density_temperature = density_temperature
        self.far_field_bonus = far_field_bonus
        self.down_scales = down_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.method_weights = list(method_weights)
        self.pre_blur = pre_blur
        self.add_noise = add_noise
        self.hflip_p = hflip_p
        self.normalize = T.Normalize(mean=mean, std=std)

    def _sample_candidate(self, img, points):
        orig_w, orig_h = img.size
        crop_size = int(self.output_size * random.uniform(*self.crop_scale))
        crop_size = min(crop_size, orig_w, orig_h)

        x0 = random.randint(0, max(0, orig_w - crop_size))
        y0 = random.randint(0, max(0, orig_h - crop_size))
        x1, y1 = x0 + crop_size, y0 + crop_size

        crop_img = img.crop((x0, y0, x1, y1))
        crop_points = points
        count = 0.0
        if points is not None and points.numel() > 0:
            px, py = points[:, 0], points[:, 1]
            mask = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
            crop_points = points[mask].clone()
            if crop_points.numel() > 0:
                crop_points[:, 0] -= x0
                crop_points[:, 1] -= y0
            count = float(crop_points.shape[0])

        if count == 0:
            hardness = 0.0
        else:
            density_per_10k = count * 10000.0 / max(crop_size * crop_size, 1)
            density_score = density_per_10k / (density_per_10k + self.density_temperature)
            center_y = (y0 + y1) * 0.5
            top_score = 1.0 - (center_y / max(orig_h, 1))
            hardness = density_score * ((1.0 - self.far_field_bonus) + self.far_field_bonus * top_score)

        patch_weight = 1.0 + (self.hard_weight_max - 1.0) * hardness
        return {
            "img": crop_img,
            "points": crop_points,
            "count": count,
            "hardness": float(max(0.0, min(hardness, 1.0))),
            "patch_weight": float(min(self.hard_weight_max, max(1.0, patch_weight))),
        }

    def _resize_hr(self, img, points):
        orig_w, orig_h = img.size
        hr_img = img.resize((self.output_size, self.output_size), Image.Resampling.BILINEAR)
        points = _resize_points(points, self.output_size / orig_w, self.output_size / orig_h)
        return hr_img, points

    def __call__(self, img, points):
        candidates = [self._sample_candidate(img, points) for _ in range(self.num_candidates)]
        use_hard_mining = (
            random.random() < self.hard_mining_prob
            and max(candidate["hardness"] for candidate in candidates) > 0
        )
        selected = (
            max(candidates, key=lambda candidate: candidate["hardness"])
            if use_hard_mining
            else random.choice(candidates)
        )

        patch_img = selected["img"]
        patch_points = selected["points"]

        if self.hflip_p > 0 and random.random() < self.hflip_p:
            patch_img = TF.hflip(patch_img)
            if patch_points is not None and patch_points.numel() > 0:
                patch_points = patch_points.clone()
                patch_points[:, 0] = patch_img.width - patch_points[:, 0]

        hr_img, patch_points = self._resize_hr(patch_img, patch_points)
        down_factor = _sample_down_factor(self.down_scales, self.min_scale, self.max_scale)
        lr_img = _apply_resolution_degradation(
            patch_img,
            output_size=self.output_size,
            down_factor=down_factor,
            method_weights=self.method_weights,
            pre_blur=self.pre_blur,
            add_noise=self.add_noise,
        )

        return {
            "hr_image": self.normalize(TF.to_tensor(hr_img)),
            "lr_image": self.normalize(TF.to_tensor(lr_img)),
            "points": patch_points,
            "gt_count": selected["count"],
            "patch_weight": selected["patch_weight"],
            "hardness": selected["hardness"],
            "is_hard": use_hard_mining,
            "down_factor": down_factor,
        }
