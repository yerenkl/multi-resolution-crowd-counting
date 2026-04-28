import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast


def train_epoch(model, loader, loss_fn, optimizer, scaler, device) -> float:
    model.train()
    total_loss = 0.0

    for images, points, densities in tqdm(loader, desc="Training"):
        images = images.to(device)
        densities = densities.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            pred_class, pred_density = model(images)
            loss, loss_info = loss_fn(pred_class, pred_density, densities, points)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_info["loss"].item()

    return total_loss / len(loader)


def _predicted_counts(pred_density):
    return pred_density.reshape(pred_density.shape[0], -1).sum(dim=1)


def _weighted_smooth_l1(pred, target, weights):
    per_sample = F.smooth_l1_loss(pred, target, reduction="none")
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def train_paired_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    *,
    consistency_weight: float = 0.05,
    hard_count_weight: float = 0.25,
    hr_loss_weight: float = 1.0,
    lr_loss_weight: float = 1.0,
) -> dict:
    model.train()
    totals = {
        "loss": 0.0,
        "loss_hr": 0.0,
        "loss_lr": 0.0,
        "hard_count_loss": 0.0,
        "consistency_loss": 0.0,
        "mean_patch_weight": 0.0,
        "hard_fraction": 0.0,
    }

    for batch in tqdm(loader, desc="Training"):
        hr_images = batch["hr_images"].to(device)
        lr_images = batch["lr_images"].to(device)
        densities = batch["densities"].to(device)
        gt_counts = batch["gt_counts"].to(device)
        patch_weights = batch["patch_weights"].to(device)
        points = [p.to(device) for p in batch["points"]]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            pred_class_hr, pred_density_hr = model(hr_images)
            pred_class_lr, pred_density_lr = model(lr_images)

            loss_hr, loss_info_hr = loss_fn(pred_class_hr, pred_density_hr, densities, points)
            loss_lr, loss_info_lr = loss_fn(pred_class_lr, pred_density_lr, densities, points)

            pred_counts_hr = _predicted_counts(pred_density_hr).float()
            pred_counts_lr = _predicted_counts(pred_density_lr).float()

            hard_count_loss = (
                _weighted_smooth_l1(pred_counts_hr, gt_counts, patch_weights)
                + _weighted_smooth_l1(pred_counts_lr, gt_counts, patch_weights)
            )
            consistency_loss = _weighted_smooth_l1(pred_counts_lr, pred_counts_hr.detach(), patch_weights)

            loss = (
                hr_loss_weight * loss_hr
                + lr_loss_weight * loss_lr
                + hard_count_weight * hard_count_loss
                + consistency_weight * consistency_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += loss.item()
        totals["loss_hr"] += loss_info_hr["loss"].item()
        totals["loss_lr"] += loss_info_lr["loss"].item()
        totals["hard_count_loss"] += hard_count_loss.item()
        totals["consistency_loss"] += consistency_loss.item()
        totals["mean_patch_weight"] += patch_weights.mean().item()
        totals["hard_fraction"] += batch["is_hard"].float().mean().item()

    num_batches = len(loader)
    return {name: value / num_batches for name, value in totals.items()}


def eval_epoch(model, device, limit: int = 100) -> dict:
    from src.evaluation.runners import eval_nwpu
    return eval_nwpu(model, device, limit=limit)
