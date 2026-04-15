"""
Transform definitions for multi-resolution crowd counting.

Augmentation strategy is TBD. This module is the single place to define
and compose transforms for NWPU training and ZoomPairs evaluation.

Transforms must output a (C, H, W) tensor so that NWPU.__getitem__ can
read the new spatial dimensions via img.shape for point coordinate rescaling.
"""

import torchvision.transforms as T


# Placeholder — replace with proper augmentation strategy when decided.
nwpu_train_transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
])

nwpu_val_transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
])

# No fixed resize for ZoomPairs — the HR/LR spatial ratio is the measurement
# of interest and should be preserved for evaluation.
zoom_pairs_transform = T.Compose([
    T.ToTensor(),
])
