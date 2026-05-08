import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from PIL import Image

from src.models.clip_ebc import load_model  # also puts CLIP_EBC_DIR in sys.path
from src.models.clip_ebc import NORMALIZE
from src.models.clip_ebc import make_density_map
from src.settings import settings

# CLIP-EBC utility for super-res inference (available once CLIP_EBC_DIR is on sys.path).
from utils.eval_utils import sliding_window_predict


DEFAULT_IMAGE_DIR = Path("/dtu/blackhole/02/137570/MultiRes/NWPU_crowd/images")


def image_to_tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def ensure_min_size(tensor: torch.Tensor, window: int) -> torch.Tensor:
    """Upscale if either spatial dim is smaller than the window."""
    _, h, w = tensor.shape
    if h < window or w < window:
        scale = window / min(h, w)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def load_gt_count(image_id: str):
    ann_path = settings.nwpu_dir / "jsons" / f"{image_id}.json"
    if not ann_path.exists():
        return None
    with ann_path.open() as f:
        ann = json.load(f)
    if "human_num" in ann:
        return int(ann["human_num"])
    points = ann.get("points", [])
    return int(len(points))


def build_gt_density_preview(
    *,
    image_id: str,
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
):
    """Build GT density map on a resized (out_h, out_w) canvas.

    NWPU points are stored in original image coordinates, so we scale them to
    the preview size to keep memory usage low.
    """
    ann_path = settings.nwpu_dir / "jsons" / f"{image_id}.json"
    if not ann_path.exists():
        return None, None
    with ann_path.open() as f:
        ann = json.load(f)

    gt_count = int(ann.get("human_num", len(ann.get("points", []))))
    points = np.array(ann.get("points", []), dtype=np.float32)
    if points.size == 0:
        pts = torch.zeros((0, 2), dtype=torch.float32)
    else:
        points = points.copy()
        points[:, 0] = points[:, 0] * (out_w / float(orig_w))
        points[:, 1] = points[:, 1] * (out_h / float(orig_h))
        points[:, 0] = np.clip(points[:, 0], 0, out_w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, out_h - 1)
        pts = torch.from_numpy(points)

    gt = make_density_map(pts, out_h, out_w)
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    return np.squeeze(gt), gt_count


@torch.no_grad()
def predict_density(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    window: int,
    stride: int,
) -> torch.Tensor:
    img_tensor = ensure_min_size(img_tensor, window)
    return sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), window, stride)


def save_model_comparison(
    *,
    img: Image.Image,
    image_id: str,
    gt_map: np.ndarray | None,
    gt_count: int | None,
    density_a: torch.Tensor,
    density_b: torch.Tensor,
    label_a: str,
    label_b: str,
    viz_dir: Path,
):
    density_map_a = density_a.squeeze().detach().cpu().numpy()
    density_map_b = density_b.squeeze().detach().cpu().numpy()
    count_a = float(density_a.sum().item())
    count_b = float(density_b.sum().item())
    diff = count_a - count_b

    err_a = abs(gt_count - count_a) if gt_count is not None else None
    err_b = abs(gt_count - count_b) if gt_count is not None else None

    vmax = max(
        float(density_map_a.max()),
        float(density_map_b.max()),
        float(gt_map.max()) if gt_map is not None else 0.0,
        1e-6,
    )

    diff_map = density_map_a - density_map_b
    diff_lim = max(float(np.abs(diff_map).max()), 1e-6)

    fig, axes = plt.subplots(1, 5, figsize=(35, 7))

    axes[0].imshow(img)
    axes[0].set_title(f"Original\n{image_id}")
    axes[0].axis("off")

    axes[1].imshow(img)
    if gt_map is not None:
        axes[1].imshow(
            gt_map,
            cmap="magma",
            alpha=0.55,
            extent=(0, img.width, img.height, 0),
            interpolation="bilinear",
            vmin=0.0,
            vmax=vmax,
        )
    gt_title = "GT" if gt_count is None else f"GT\ncount={gt_count}"
    axes[1].set_title(gt_title)
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(
        density_map_a,
        cmap="magma",
        alpha=0.55,
        extent=(0, img.width, img.height, 0),
        interpolation="bilinear",
        vmin=0.0,
        vmax=vmax,
    )
    title_a = f"{label_a}\ncount={count_a:.1f}" if err_a is None else f"{label_a}\ncount={count_a:.1f}  err={err_a:.1f}"
    axes[2].set_title(title_a)
    axes[2].axis("off")

    axes[3].imshow(img)
    axes[3].imshow(
        density_map_b,
        cmap="magma",
        alpha=0.55,
        extent=(0, img.width, img.height, 0),
        interpolation="bilinear",
        vmin=0.0,
        vmax=vmax,
    )
    title_b = (
        f"{label_b}\ncount={count_b:.1f}  (Δ={diff:+.1f})"
        if err_b is None
        else f"{label_b}\ncount={count_b:.1f}  err={err_b:.1f}  (Δ={diff:+.1f})"
    )
    axes[3].set_title(title_b)
    axes[3].axis("off")
    # axes[4]
    axes[4].imshow(img)
    diff_im = axes[4].imshow(
        diff_map,
        cmap="coolwarm",
        alpha=0.7,
        extent=(0, img.width, img.height, 0),
        interpolation="bilinear",
        vmin=-diff_lim,
        vmax=diff_lim,
    )
    axes[4].set_title(f"Diff ({label_a} − {label_b})\nrange=[{-diff_lim:.2g}, {diff_lim:.2g}]")
    axes[4].axis("off")

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # replace the fig.colorbar(...) lines with:
    cbar_ax = inset_axes(axes[4], width="4%", height="60%", loc="lower right", borderpad=1)
    cbar = fig.colorbar(diff_im, cax=cbar_ax)
    cbar.set_label("density diff\n(A − B)", fontsize=7)
    cbar.ax.tick_params(labelsize=6, colors="white")
    cbar.ax.yaxis.label.set_color("white") 

    fig.tight_layout(pad=0.2)
    output_path = viz_dir / f"{image_id}_compare.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--num-viz", type=int, default=10, help="Number of images to visualize as heatmaps (single-model mode).")
    parser.add_argument("--viz-dir", type=Path, default=settings.RESULTS_DIR / "superres" / "heatmaps")
    parser.add_argument("--weights-a", type=Path, default=None, help="Checkpoint path for model A (enables compare mode when --weights-b is also set).")
    parser.add_argument("--weights-b", type=Path, default=None, help="Checkpoint path for model B.")
    parser.add_argument("--label-a", type=str, default="Model A")
    parser.add_argument("--label-b", type=str, default="Model B")
    parser.add_argument("--top-k", type=int, default=2, help="How many images (with biggest count differences) to visualize in compare mode.")
    parser.add_argument("--max-images", type=int, default=200, help="How many images from val.txt to scan in compare mode.")
    parser.add_argument("--resize", type=int, default=224, help="Resize images to this square size before inference (keeps compare mode fast).")
    parser.add_argument("--window", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--tta", action="store_true", help="Visualize TTA scale breakdown for --num-viz images.")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    viz_dir = args.viz_dir.expanduser().resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

    image_dir = args.image_dir.expanduser().resolve()
    with open(settings.nwpu_dir / "val.txt") as f:
        all_image_ids = [line.strip().split()[0] for line in f if line.strip()]

    if args.tta:
            weights = args.weights_a or args.weights_b  # accept either flag
            model = load_model(device, weights_path=weights) if weights else load_model(device)
            model.eval()

            image_ids = [3126, 3161, 3183]
            for image_id in image_ids:
                image_path = image_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    continue

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    orig_w, orig_h = img.size
                    base_tensor = image_to_tensor(img)  # C,H,W  unnormalized

                scale_density_maps = []
                for s in args.scales:
                    resized = F.interpolate(
                        base_tensor.unsqueeze(0),
                        scale_factor=s,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    resized = ensure_min_size(resized, args.window)
                    density = predict_density(model, NORMALIZE(resized), device, args.window, args.stride)
                    scale_density_maps.append((s, density.squeeze().detach().cpu().numpy()))

                gt_map, gt_count = build_gt_density_preview(
                    image_id=str(image_id),
                    orig_w=orig_w,
                    orig_h=orig_h,
                    out_w=orig_w,
                    out_h=orig_h,
                )

                # --- NEW PLOTTING LOGIC FOR MULTIPLE SCALES ---
                # 1 for GT + 1 for each scale evaluated
                num_subplots = 1 + len(scale_density_maps)
                fig, axes = plt.subplots(1, num_subplots, figsize=(7 * num_subplots, 7))
                
                # Subplot 0: Original image + GT Map
                axes[0].imshow(img)
                if gt_map is not None:
                    axes[0].imshow(
                        gt_map,
                        cmap="magma",
                        alpha=0.55,
                        extent=(0, img.width, img.height, 0),
                        interpolation="bilinear",
                    )
                gt_title = "Original / No GT" if gt_count is None else f"GT\ncount={gt_count}"
                axes[0].set_title(gt_title)
                axes[0].axis("off")

                # Subplots 1 to N: Heatmap for each scale
                for i, (scale, density_map) in enumerate(scale_density_maps, start=1):
                    count = density_map.sum()
                    axes[i].imshow(img)
                    axes[i].imshow(
                        density_map,
                        cmap="magma",
                        alpha=0.55,
                        extent=(0, img.width, img.height, 0),
                        interpolation="bilinear",
                    )
                    axes[i].set_title(f"Scale: {scale}x\ncount={count:.1f}")
                    axes[i].axis("off")

                fig.tight_layout(pad=0.2)
                out = viz_dir / f"{image_id}_tta_compare.png"
                fig.savefig(out, dpi=150)
                plt.close(fig)
                
                print(f"Saved TTA heatmap to {out}")
            return
    
    # ── Compare mode (two models) ───────────────────────────────────────────
    if args.weights_a is not None and args.weights_b is not None:
        model_a = load_model(device, weights_path=args.weights_a)
        model_b = load_model(device, weights_path=args.weights_b)
        model_a.eval()
        model_b.eval()

        window = int(args.window)
        stride = int(args.stride)
        preview_size = int(args.resize)

        scan_ids = all_image_ids[: max(1, int(args.max_images))]
        rows = []

        for image_id in scan_ids:
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                continue

            gt_count = load_gt_count(image_id)

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                preview = img.resize((preview_size, preview_size), Image.Resampling.BILINEAR)
                img_tensor = NORMALIZE(image_to_tensor(preview))

            density_a = predict_density(model_a, img_tensor, device, window, stride)
            density_b = predict_density(model_b, img_tensor, device, window, stride)
            count_a = float(density_a.sum().item())
            count_b = float(density_b.sum().item())
            # abs_diff = abs(count_a - count_b)
            abs_diff = count_a - count_b
            rows.append((abs_diff, image_id, count_a, count_b, gt_count))

        rows.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, int(args.top_k))
        top = rows[:top_k]

        print(f"Scanned {len(rows)} images. Top-{len(top)} by |Δcount|:")
        for rank, (abs_diff, image_id, count_a, count_b, gt_count) in enumerate(top, start=1):
            gt_str = "-" if gt_count is None else str(gt_count)
            err_a = "-" if gt_count is None else f"{abs(gt_count - count_a):.1f}"
            err_b = "-" if gt_count is None else f"{abs(gt_count - count_b):.1f}"
            print(
                f"{rank:02d}. {image_id}: GT={gt_str} | {args.label_a}={count_a:.1f} (err={err_a}) | {args.label_b}={count_b:.1f} (err={err_b}) | |Δ|={abs_diff:.1f}"
            )

        for abs_diff, image_id, _, _, _ in top:
            image_path = image_dir / f"{image_id}.jpg"
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                orig_w, orig_h = img.size
                preview = img.resize((preview_size, preview_size), Image.Resampling.BILINEAR)
                img_tensor = NORMALIZE(image_to_tensor(preview))

                density_a = predict_density(model_a, img_tensor, device, window, stride)
                density_b = predict_density(model_b, img_tensor, device, window, stride)

                gt_map, gt_count = build_gt_density_preview(
                    image_id=image_id,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    out_w=preview_size,
                    out_h=preview_size,
                )

                out = save_model_comparison(
                    img=img,
                    image_id=image_id,
                    gt_map=gt_map,
                    gt_count=gt_count,
                    density_a=density_a,
                    density_b=density_b,
                    label_a=args.label_a,
                    label_b=args.label_b,
                    viz_dir=viz_dir,
                )
                print(f"Saved compare heatmap (|Δ|={abs_diff:.1f}) to {out}")
        return

    # ── Single-model mode (original behavior) ──────────────────────────────
    model = load_model(device)

    image_ids = all_image_ids[: max(1, args.num_viz)]
    model.eval()
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            preview = img.resize((224, 224), Image.Resampling.BILINEAR)
            img_tensor = NORMALIZE(image_to_tensor(preview))
            density = predict_density(model, img_tensor, device, 224, 224)
            density_map = density.squeeze().detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.imshow(
                density_map,
                cmap="magma",
                alpha=0.55,
                extent=(0, img.width, img.height, 0),
                interpolation="bilinear",
            )
            ax.set_title(f"{image_id} heatmap")
            ax.axis("off")
            fig.tight_layout(pad=0)

            output_path = viz_dir / f"{image_id}_heatmap.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()