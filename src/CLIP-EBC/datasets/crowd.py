import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms.functional as TF
import json
import os
from glob import glob
from PIL import Image
import numpy as np
from typing import Optional, Callable, Union, Tuple
from warnings import warn

from .utils import get_id, generate_density_map

curr_dir = os.path.dirname(os.path.abspath(__file__))

available_datasets = [
    "shanghaitech_a", "sha",
    "shanghaitech_b", "shb",
    "ucf_qnrf", "qnrf", "ucf-qnrf",
    "nwpu", "nwpu_crowd", "nwpu-crowd",
    "jhu", "jhu_crowd", "jhu_crowd_v2"
]


def standardize_dataset_name(dataset: str) -> str:
    assert dataset.lower() in available_datasets, f"Dataset {dataset} is not available."
    if dataset.lower() in ["shanghaitech_a", "sha"]:
        return "sha"
    elif dataset.lower() in ["shanghaitech_b", "shb"]:
        return "shb"
    elif dataset.lower() in ["ucf_qnrf", "qnrf", "ucf-qnrf"]:
        return "qnrf"
    elif dataset.lower() in ["nwpu", "nwpu_crowd", "nwpu-crowd"]:
        return "nwpu"
    else:  # dataset.lower() in ["jhu", "jhu_crowd", "jhu_crowd_v2"]
        return "jhu"


def _calc_resize_shape(
    img_w: int,
    img_h: int,
    min_size: int,
    max_size: int,
    base: int = 32
) -> Tuple[int, int]:
    """
    This function generates a new size for an image while keeping the aspect ratio. The new size should be within the given range (min_size, max_size).

    Args:
        img_w (int): The width of the image.
        img_h (int): The height of the image.
        min_size (int): The minimum size of the edges of the image.
        max_size (int): The maximum size of the edges of the image.
    """
    assert min_size % base == 0, f"min_size ({min_size}) must be a multiple of {base}"
    if max_size != float("inf"):
        assert max_size % base == 0, f"max_size ({max_size}) must be a multiple of {base} if provided"

    assert min_size <= max_size, f"min_size ({min_size}) must be less than or equal to max_size ({max_size})"

    aspect_ratios = (img_w / img_h, img_h / img_w)
    if min_size / max_size <= min(aspect_ratios) <= max(aspect_ratios) <= max_size / min_size:  # possible to resize and preserve the aspect ratio
        if min_size <= min(img_w, img_h) <= max(img_w, img_h) <= max_size:  # already within the range, no need to resize
            ratio = 1.
        elif min(img_w, img_h) < min_size:  # smaller than the minimum size, resize to the minimum size
            ratio = min_size / min(img_w, img_h)
        else:  # larger than the maximum size, resize to the maximum size
            ratio = max_size / max(img_w, img_h)

        new_w, new_h = int(round(img_w * ratio / base) * base), int(round(img_h * ratio / base) * base)
        new_w = max(min_size, min(max_size, new_w))
        new_h = max(min_size, min(max_size, new_h))
        return new_w, new_h

    # If constraints are impossible while preserving aspect ratio, relax max_size.
    warn(
        f"Impossible to resize {img_w}x{img_h} within ({min_size}, {max_size}) while preserving aspect ratio. "
        "Ignoring max_size for this sample."
    )
    return _calc_resize_shape(img_w, img_h, min_size, float("inf"), base)


class Crowd(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
        resize_min_size: Optional[int] = 448,
        resize_max_size: Optional[int] = 3072,
        resize_base: int = 32,
    ) -> None:
        """
        Dataset for crowd counting.
        """
        assert dataset.lower() in available_datasets, f"Dataset {dataset} is not available."
        assert split in ["train", "val"], f"Split {split} is not available."
        assert num_crops > 0, f"num_crops should be positive, got {num_crops}."
        assert (resize_min_size is None) == (resize_max_size is None), "resize_min_size and resize_max_size must be both set or both None."
        if resize_min_size is not None and resize_max_size is not None:
            assert resize_min_size > 0 and resize_max_size > 0, "resize_min_size and resize_max_size must be positive."
            assert resize_min_size <= resize_max_size, "resize_min_size must be <= resize_max_size."
            assert resize_min_size % resize_base == 0, f"resize_min_size ({resize_min_size}) must be a multiple of resize_base ({resize_base})."
            assert resize_max_size % resize_base == 0, f"resize_max_size ({resize_max_size}) must be a multiple of resize_base ({resize_base})."

        self.dataset = standardize_dataset_name(dataset)
        self.split = split

        self.__find_root__()
        self.__make_dataset__()
        self.__check_sanity__()
        self.indices = list(range(len(self.image_names)))

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename
        self.num_crops = num_crops
        self.resize_min_size = resize_min_size
        self.resize_max_size = resize_max_size
        self.resize_base = resize_base

        

    def __find_root__(self) -> None:
        self.root = "/dtu/blackhole/02/137570/MultiRes/NWPU_crowd"

    def __make_dataset__(self) -> None:
        with open(os.path.join(self.root, f"{self.split}.txt"), "r") as f:
                    self.image_paths = [line.strip().split()[0] for line in f.readlines()]

        image_names = [os.path.basename(image_path) for image_path in self.image_paths]
        label_names = [image_name.replace("images", "jsons").replace(".jpg", ".json") for image_name in image_names]
        image_names.sort(key=get_id)
        label_names.sort(key=get_id)
        image_ids = tuple([get_id(image_name) for image_name in image_names])
        label_ids = tuple([get_id(label_name) for label_name in label_names])

        assert image_ids == label_ids, "image_ids and label_ids do not match."
        self.image_names = tuple(image_names)
        self.label_names = tuple(label_names)

    def __check_sanity__(self) -> None:
        if self.dataset == "sha":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 300, f"ShanghaiTech_A train split should have 300 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 182, f"ShanghaiTech_A val split should have 182 images, but found {len(self.image_names)}."
        elif self.dataset == "shb":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 400, f"ShanghaiTech_B train split should have 400 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 316, f"ShanghaiTech_B val split should have 316 images, but found {len(self.image_names)}."
        elif self.dataset == "nwpu":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 3109, f"NWPU train split should have 3109 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 500, f"NWPU val split should have 500 images, but found {len(self.image_names)}."
        elif self.dataset == "qnrf":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 1201, f"UCF_QNRF train split should have 1201 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 334, f"UCF_QNRF val split should have 334 images, but found {len(self.image_names)}."
        else:  # self.dataset == "jhu"
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 2772, f"JHU train split should have 2772 images, but found {len(self.image_names)}."
            else:
                assert len(self.image_names) == len(self.label_names) == 1600, f"JHU val split should have 1600 images, but found {len(self.image_names)}."

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]

        image_path = os.path.join(self.root, "images", image_name+".jpg")
        label_path = os.path.join(self.root, "jsons", label_name+".json")

        with open(image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
        image = self.to_tensor(image)

        with open(label_path, "r") as f:
            annotation = json.load(f)

        label = torch.tensor(annotation["points"], dtype=torch.float32)

        if self.resize_min_size is not None and self.resize_max_size is not None:
            old_h, old_w = image.shape[-2], image.shape[-1]
            new_h, new_w = _calc_resize_shape(old_w, old_h, self.resize_min_size, self.resize_max_size, self.resize_base)
            if (new_h, new_w) != (old_h, old_w):
                image = TF.resize(
                    image,
                    [new_h, new_w],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True,
                )
                if label.numel() > 0:
                    label[:, 0] = label[:, 0] * (new_w / old_w)
                    label[:, 1] = label[:, 1] * (new_h / old_h)
                    label[:, 0] = label[:, 0].clamp(min=0, max=new_w - 1)
                    label[:, 1] = label[:, 1].clamp(min=0, max=new_h - 1)

        if self.transforms is not None:
            images_labels = [self.transforms(image.clone(), label.clone()) for _ in range(self.num_crops)]
            images, labels = zip(*images_labels)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            labels = [label.clone() for _ in range(self.num_crops)]

        images = [self.normalize(img) for img in images]
        if idx in self.indices:
            density_maps = torch.stack([generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in zip(images, labels)], 0)
        else:
            labels = None
            density_maps = None

        image_names = [image_name] * len(images)
        images = torch.stack(images, 0)

        if self.return_filename:
            return images, labels, density_maps, image_names
        else:
            return images, labels, density_maps


class NWPUTest(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
    ) -> None:
        """
        The test set of NWPU-Crowd dataset. The test set is not labeled, so only images are returned.
        """
        self.root = "/dtu/blackhole/02/137570/MultiRes/NWPU_crowd"

        image_npys = glob(os.path.join(self.root, "test", "images", "*.npy"))
        if len(image_npys) > 0:
            self.image_type = "npy"
            image_names = image_npys
        else:
            self.image_type = "jpg"
            image_names = glob(os.path.join(self.root, "test", "images", "*.jpg"))

        image_names = [os.path.basename(image_name) for image_name in image_names]
        assert len(image_names) == 1500, f"NWPU test split should have 1500 images, but found {len(image_names)}."
        image_names.sort(key=get_id)
        self.image_names = tuple(image_names)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, str]]:
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, "test", "images", image_name)

        if self.image_type == "npy":
            with open(image_path, "rb") as f:
                image = np.load(f)
            image = torch.from_numpy(image).float() / 255.
        else:
            with open(image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
            image = self.to_tensor(image)
        
        label = torch.tensor([], dtype=torch.float)  # dummy label
        image, _ = self.transforms(image, label) if self.transforms is not None else (image, label)
        image = self.normalize(image)

        if self.return_filename:
            return image, image_name
        else:
            return image
