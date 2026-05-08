"""
Evaluate CLIP-EBC (ViT-B/16) on NWPU-Crowd val at native resolution.

Usage:
    cd ~/project/multi-resolution-crowd-counting
    uv run python entrypoints/eval_nwpu_native.py [--device cuda:0]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision import transforms as T

from PIL import Image


from src.models_local.clip_ebc import load_model, NORMALIZE  # also puts CLIP_EBC_DIR in sys.path
import torch.nn.functional as F

#
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#
# from utils.eval_utils import sliding_window_predict  # CLIP-EBC utility (in sys.path after clip_ebc import)

WINDOW_SIZE = 224
STRIDE = 224

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.path)

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device, out_dir / "results" / "latest.pth")
    model.eval()


    nwpu_root = Path("/dtu/blackhole/02/137570/MultiRes")
    with open(nwpu_root / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]



    pred_counts, gt_counts = [], []
    img = Image.open(nwpu_root / "images" / f"{image_ids[4]}.jpg").convert("RGB")
    img_tensor = NORMALIZE(T.ToTensor()(img))
    with open(nwpu_root / "jsons" / f"{image_ids[4]}.json") as f:
        gt_count = json.load(f)["human_num"]
    density = predict_count(model, img_tensor, device)
    print(density)
    gt_counts.append(gt_count)


@torch.no_grad()
def predict_count(model, img_tensor, device, window: int = WINDOW_SIZE, stride: int = STRIDE) -> float:
    _, h, w = img_tensor.shape
    if h < window or w < window:
        scale = window / min(h, w)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    # density = sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), window, stride)

    return 0


if __name__ == "__main__":
    main()
