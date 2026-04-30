import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast
from tqdm import tqdm


def degrade_batch(
    images: Tensor,
    down_scales: Tuple[int, ...] = (2, 4, 8),
    output_size: int = 224,
) -> Tensor:
    B, C, H, W = images.shape
    degraded = images.clone()
    for i in range(B):
        factor = random.choice(down_scales)
        lr_size = max(1, output_size // factor)
        small = F.interpolate(
            images[i : i + 1], size=(lr_size, lr_size), mode="bilinear", align_corners=False
        )
        degraded[i : i + 1] = F.interpolate(
            small, size=(H, W), mode="bilinear", align_corners=False
        )
    return degraded


def dann_train_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    alpha: float,
    dann_weight: float = 1.0,
    down_scales: Tuple[int, ...] = (2, 4, 8),
) -> dict:
    model.train()
    model.set_alpha(alpha)

    domain_loss_fn = torch.nn.BCEWithLogitsLoss()

    total_task_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    for images, points, densities in tqdm(loader, desc="DANN Training"):
        images_hr = images.to(device)
        densities = densities.to(device)
        points = [p.to(device) for p in points]

        images_lr = degrade_batch(images_hr, down_scales=down_scales)

        optimizer.zero_grad()

        with autocast():
            logits_hr, density_hr, domain_logits_hr = model(images_hr)
            task_loss_hr, _ = loss_fn(logits_hr, density_hr, densities, points)
            hr_labels = torch.zeros_like(domain_logits_hr)
            domain_loss_hr = domain_loss_fn(domain_logits_hr, hr_labels)

            logits_lr, density_lr, domain_logits_lr = model(images_lr)
            task_loss_lr, _ = loss_fn(logits_lr, density_lr, densities, points)
            lr_labels = torch.ones_like(domain_logits_lr)
            domain_loss_lr = domain_loss_fn(domain_logits_lr, lr_labels)

            task_loss = task_loss_hr + task_loss_lr
            domain_loss = domain_loss_hr + domain_loss_lr
            loss = task_loss + dann_weight * domain_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_task_loss += task_loss.item()
        total_domain_loss += domain_loss.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "task_loss": total_task_loss / n_batches,
        "domain_loss": total_domain_loss / n_batches,
        "alpha": alpha,
    }


def dann_train_epoch_v2(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    alpha: float,
    dann_weight: float = 1.0,
) -> dict:
    """DANN training with pre-saved downscaled LR images (no on-the-fly degradation)."""
    model.train()
    model.set_alpha(alpha)

    domain_loss_fn = torch.nn.BCEWithLogitsLoss()

    total_task_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="DANN v2 Training"):
        images_hr = batch["hr_images"].to(device)
        images_lr = batch["lr_images"].to(device)
        densities = batch["densities"].to(device)
        points = [p.to(device) for p in batch["points"]]

        optimizer.zero_grad()

        with autocast():
            logits_hr, density_hr, domain_logits_hr = model(images_hr)
            task_loss_hr, _ = loss_fn(logits_hr, density_hr, densities, points)
            hr_labels = torch.zeros_like(domain_logits_hr)
            domain_loss_hr = domain_loss_fn(domain_logits_hr, hr_labels)

            logits_lr, density_lr, domain_logits_lr = model(images_lr)
            task_loss_lr, _ = loss_fn(logits_lr, density_lr, densities, points)
            lr_labels = torch.ones_like(domain_logits_lr)
            domain_loss_lr = domain_loss_fn(domain_logits_lr, lr_labels)

            task_loss = task_loss_hr + task_loss_lr
            domain_loss = domain_loss_hr + domain_loss_lr
            loss = task_loss + dann_weight * domain_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_task_loss += task_loss.item()
        total_domain_loss += domain_loss.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "task_loss": total_task_loss / n_batches,
        "domain_loss": total_domain_loss / n_batches,
        "alpha": alpha,
    }
