"""
Verify which parameters are trainable vs frozen in our training setup.

Prints every trainable parameter name, shape, and count — so we can confirm
that crowd_params picks up exactly VPT tokens, image decoder, projection,
and logit scale (and nothing from the frozen CLIP encoders).

Usage:
    uv run python scripts/verify_trainable_params.py

Expected output:
    Trainable parameters:
      vpt_0 ... vpt_11          — 12 VPT token sets (32 vectors x 768 dim each)
      image_decoder.*            — ResNet block weights/biases
      projection.*               — Conv2d 768->512 weight/bias
      logit_scale                — 1 scalar

    Frozen parameter groups:
      image_encoder              — ~86M params (ViT-B/16, OpenAI pretrained)
      text_encoder               — ~37M params (CLIP text transformer)

    DANN wrapper:
      crowd_params               — all of the above trainable params
      disc_params                — domain classifier MLP weights/biases
      "All trainable params are covered by the optimizer."

    If any trainable param is NOT in the optimizer, it will be flagged as MISSING.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clip_ebc import build_model
from src.dann.model import DANNModel
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    import torch

    device = torch.device("cpu")
    crowd_model = build_model(device)

    logger.info("=== CLIP-EBC (crowd_model) ===")

    trainable = []
    frozen_groups = {}
    for name, p in crowd_model.named_parameters():
        if p.requires_grad:
            trainable.append((name, list(p.shape), p.numel()))
        else:
            group = name.split(".")[0]
            frozen_groups.setdefault(group, [0, 0])
            frozen_groups[group][0] += 1
            frozen_groups[group][1] += p.numel()

    logger.info("Trainable parameters:")
    total_trainable = 0
    for name, shape, n in trainable:
        logger.info(f"  {name:<50s} {str(shape):<20s} {n:>10,}")
        total_trainable += n
    logger.success(f"Total trainable: {total_trainable:,}")

    logger.info("Frozen parameter groups:")
    total_frozen = 0
    for group, (count, n) in sorted(frozen_groups.items()):
        logger.info(f"  {group:<50s} {count:>5} params, {n:>12,} values")
        total_frozen += n
    logger.info(f"Total frozen: {total_frozen:,}")

    logger.info("")
    logger.info("=== DANN wrapper ===")

    dann_model = DANNModel(crowd_model=crowd_model, feature_dim=768, hidden_dim=256, dropout=0.5)

    crowd_params = [p for p in crowd_model.parameters() if p.requires_grad]
    disc_params = list(dann_model.domain_classifier.parameters())

    logger.info(f"crowd_params count: {len(crowd_params)} tensors, {sum(p.numel() for p in crowd_params):,} values")
    logger.info(f"disc_params count:  {len(disc_params)} tensors, {sum(p.numel() for p in disc_params):,} values")

    all_trainable_dann = [(name, p.numel()) for name, p in dann_model.named_parameters() if p.requires_grad]
    in_optimizer = set(id(p) for p in crowd_params + disc_params)
    all_trainable_ids = set(id(p) for _, p in dann_model.named_parameters() if p.requires_grad)

    missing = all_trainable_ids - in_optimizer
    if missing:
        logger.error(f"{len(missing)} trainable params NOT in optimizer:")
        for name, p in dann_model.named_parameters():
            if p.requires_grad and id(p) not in in_optimizer:
                logger.error(f"  MISSING: {name} ({p.numel():,} values)")
    else:
        logger.success("All trainable params are covered by the optimizer.")

    dann_model.remove_hook()


if __name__ == "__main__":
    main()
