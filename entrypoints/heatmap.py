import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import gc  # Garbage collection

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models_local.clip_ebc import load_model, NORMALIZE
from src.settings import settings


# Setup paths for CLIP-EBC
sys.path.insert(0, "/dtu/blackhole/0a/224426/CLIP-EBC-main")
from utils.eval_utils import sliding_window_predict


DEFAULT_IMAGE_DIR = Path("/dtu/blackhole/02/137570/MultiRes/images")


def image_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


import json

def get_gt_count(json_dir, image_id):
    json_path = json_dir / f"{image_id}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get("human_num", 0)
    return None

def get_density_map(model, img, device, window=224, stride=224):
    preview = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_tensor = NORMALIZE(image_to_tensor(preview))

    _, h, w = img_tensor.shape
    if h < window or w < window:
        scale = window / min(h, w)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    with torch.no_grad():
        density = sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), window, stride)
        predicted_count = density.sum().item()
    return density.squeeze().detach().cpu().numpy(), predicted_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weights-1", type=str, required=True)
    parser.add_argument("--weights-2", type=str, required=True)
    parser.add_argument("--label-1", type=str, default="Model A")
    parser.add_argument("--label-2", type=str, default="Model B")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-viz", type=int, default=4)
    parser.add_argument("--viz-dir", type=Path, default=settings.RESULTS_DIR / "superres" / "comparison")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_dir = settings.nwpu_dir / "images"
    json_dir = settings.nwpu_dir / "jsons"

    with open(settings.nwpu_dir / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()][: args.num_viz]


    # --- PHASE 1: Process Model 1 ---
    print(f"Loading Model 1 from {args.weights_1}")
    model = load_model(device, dir=args.weights_1)
    model.eval()

    results_m1 = {}
    gt_counts = {}  # Dictionary to store GT

    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        if image_path.exists():
            # Get GT Count
            gt_counts[image_id] = get_gt_count(json_dir, image_id)

            with Image.open(image_path) as img:
                results_m1[image_id] = get_density_map(model, img.convert("RGB"), device)

    # Explicitly clear VRAM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- PHASE 2: Process Model 2 ---
    print(f"Loading Model 2 from {args.weights_2}")
    model = load_model(device, dir=args.weights_2)
    model.eval()

    results_m2 = {}
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        if image_path.exists():
            with Image.open(image_path) as img:
                results_m2[image_id] = get_density_map(model, img.convert("RGB"), device)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- PHASE 3: Generate Figures ---
    viz_dir = args.viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        if image_id not in results_m1 or image_id not in results_m2:
            continue

        image_path = image_dir / f"{image_id}.jpg"
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Unpack predictions (map, count)
            dm1, count1 = results_m1[image_id]
            dm2, count2 = results_m2[image_id]
            gt_val = gt_counts.get(image_id, "N/A")

            fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=200)

            # 1. Original Image with GT Count
            axes[0].imshow(img)
            axes[0].set_title(f"ID: {image_id}\nGT Count: {gt_val}", fontsize=16, fontweight='bold')

            # 2. Model A Prediction
            axes[1].imshow(img)
            axes[1].imshow(dm1, cmap="magma", alpha=0.6, extent=(0, img.width, img.height, 0), interpolation="bilinear")
            axes[1].set_title(f"{args.label_1}\nPred: {count1:.2f}", fontsize=16)

            # 3. Model B Prediction
            axes[2].imshow(img)
            axes[2].imshow(dm2, cmap="magma", alpha=0.6, extent=(0, img.width, img.height, 0), interpolation="bilinear")
            axes[2].set_title(f"{args.label_2}\nPred: {count2:.2f}", fontsize=16)

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            output_path = viz_dir / f"{image_id}_compare.png"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)


if __name__ == "__main__":
    main()