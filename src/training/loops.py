import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def train_epoch(model, loader, loss_fn, optimizer, scaler, device) -> float:
    model.train()
    total_loss = 0.0

    for images, points, densities in tqdm(loader, desc="Training"):
        images = images.to(device)
        densities = densities.to(device)
        points = [p.to(device) for p in points]

        optimizer.zero_grad()
        with autocast():
            pred_class, pred_density = model(images)
            loss, loss_info = loss_fn(pred_class, pred_density, densities, points)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_info["loss"].item()

    return total_loss / len(loader)


def eval_epoch(model, device, downscaled_dir, limit: int = 100) -> dict:
    from src.evaluation.runners import eval_nwpu
    return eval_nwpu(model, device, downscaled_dir, limit=limit)
