import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import gc

# Standard settings and model loads
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models_local.clip_ebc import load_model, NORMALIZE
from src.settings import settings

sys.path.insert(0, "/dtu/blackhole/0a/224426/CLIP-EBC-main")
from utils.eval_utils import sliding_window_predict

DEFAULT_IMAGE_DIR = Path("/dtu/blackhole/02/137570/MultiRes/images")


def image_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


# MODIFIED: Now returns the full data dictionary to access points
def get_gt_data(json_dir, image_id):
    json_path = json_dir / f"{image_id}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
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
    parser.add_argument("--viz-dir", type=Path, default=settings.RESULTS_DIR / "anotated" / "comparison")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_dir = settings.nwpu_dir / "images"
    json_dir = settings.nwpu_dir / "jsons"

    # HARDCODE image IDs if you prefer: image_ids = ["0001", "0002"]
    with open(settings.nwpu_dir / "val.txt") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()][: args.num_viz]

    # --- PHASE 1: Process Model 1 ---
    model = load_model(device, dir=args.weights_1)
    model.eval()

    results_m1 = {}
    gt_info = {}

    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        if image_path.exists():
            gt_info[image_id] = get_gt_data(json_dir, image_id)
            with Image.open(image_path) as img:
                results_m1[image_id] = get_density_map(model, img.convert("RGB"), device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 2: Process Model 2 ---
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
    torch.cuda.empty_cache()

    # --- PHASE 3: Generate Figures with POINTS ---
    viz_dir = args.viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        if image_id not in results_m1 or image_id not in results_m2:
            continue

        image_path = image_dir / f"{image_id}.jpg"
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Prediction totals
            _, count1 = results_m1[image_id]
            _, count2 = results_m2[image_id]

            # Get GT Points
            current_gt = gt_info.get(image_id, {})
            points = np.array(current_gt.get("points", []))
            gt_count = current_gt.get("human_num", "N/A")

            fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=200)

            # 1. Original Image (Clean)
            axes[0].imshow(img)
            axes[0].set_title(f"ID: {image_id}\nGT Count: {gt_count}", fontsize=16, fontweight='bold')

            # 2. GT Points shown on Image
            axes[1].imshow(img)
            if len(points) > 0:
                axes[1].scatter(points[:, 0], points[:, 1], s=10, c='red', marker='o', edgecolors='white',
                                linewidths=0.5)
            axes[1].set_title(f"GT Annotated\nActual Points", fontsize=16)

            # 3. Model Predictions (Text Summary)
            # You can still overlay the heatmap here if you want, 
            # or just show Model A vs Model B side by side.
            axes[2].imshow(img)
            axes[2].set_title(f"Comparison\n{args.label_1}: {count1:.1f} | {args.label_2}: {count2:.1f}", fontsize=16)

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            output_path = viz_dir / f"{image_id}_points_compare.png"
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved point comparison for {image_id}")


if __name__ == "__main__":
    main()