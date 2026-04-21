import torch
import torch.nn.functional as F

from utils.eval_utils import sliding_window_predict  # CLIP-EBC utility (in sys.path after clip_ebc import)

WINDOW_SIZE = 224
STRIDE = 224


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
    density = sliding_window_predict(model, img_tensor.unsqueeze(0).to(device), window, stride)
    return density.sum().item()
