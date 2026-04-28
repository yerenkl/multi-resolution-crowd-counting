import sys
import importlib.util

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from src.settings import settings
from src.logger import get_logger

logger = get_logger(__name__)

# ── CLIP-EBC bootstrap ────────────────────────────────────────────────
# Append (not insert) CLIP_EBC_DIR so that project-root packages (especially
# our own `datasets/`) take precedence in sys.path. CLIP-EBC's models, utils,
# and losses packages don't conflict with anything in our project root.
sys.path.append(str(settings.CLIP_EBC_DIR))
from models import get_model  # noqa: E402  (CLIP-EBC's models package)

# Load generate_density_map via spec_from_file_location so it is registered
# in sys.modules under a private name and does NOT shadow our top-level
# `datasets/` package with CLIP-EBC's `datasets/`.
_spec = importlib.util.spec_from_file_location(
    "_clip_ebc_dataset_utils",
    settings.CLIP_EBC_DIR / "datasets" / "utils.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_generate_density_map = _mod.generate_density_map
del _spec, _mod

# ── Model config ──────────────────────────────────────────────────────
MODEL_CFG = dict(
    backbone="clip_vit_b_16",
    input_size=224,
    reduction=8,
    bins=[[0, 0], [1, 1], [2, 2], [3, 3], [4, float("inf")]],
    anchor_points=[0, 1, 2, 3, 4.21931],
    prompt_type="word",
    num_vpt=32,
    vpt_drop=0.0,
    deep_vpt=True,
)

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_model(device):
    """Build CLIP-EBC from raw CLIP pretrained weights (no CLIP-EBC checkpoint)."""
    model = get_model(**MODEL_CFG)
    model.to(device)
    logger.info("Built CLIP-EBC from raw CLIP weights (no pretrained checkpoint)")
    return model


def load_model(device):
    """Build CLIP-EBC and load the authors' pretrained checkpoint."""
    model = get_model(**MODEL_CFG)
    ckpt = torch.load(settings.CLIP_EBC_WEIGHTS, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    logger.info(f"Loaded pretrained weights from {settings.CLIP_EBC_WEIGHTS}")
    return model


def make_density_map(points, h, w):
    return _generate_density_map(points, h, w, sigma=None)
