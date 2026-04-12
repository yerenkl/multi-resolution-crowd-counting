import json
import os
import random
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as standard_transforms

warnings.filterwarnings('ignore')


class NWPU(Dataset):
    def __init__(self, transform=None, split="train",
                 path="/home/user/src_wsl/multi-resolution-crowd-counting/PET-main/data/NWPU-Crowd_2048"):
        self.dataset_path = path
        self.train = split == "train"
        self.flip = True

        with open(os.path.join(self.dataset_path, f"{split}.txt"), "r") as f:
            self.image_paths = [line.strip().split()[0] for line in f.readlines()]

        self.nSamples = len(self.image_paths)

        self.transform = transform
        self.patch_size = 256

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_id = self.image_paths[index]
        img_path = os.path.join(self.dataset_path, "images", image_id + ".jpg")
        label_path = os.path.join(self.dataset_path, "jsons", image_id + ".json")

        # I am lazy to clean the dataset, so I add a fallback mechanism to handle missing files
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            fallback_id = "0001"
            img_path = os.path.join(self.dataset_path, "images", fallback_id + ".jpg")
            label_path = os.path.join(self.dataset_path, "jsons", fallback_id + ".json")

        img, points = load_data((img_path, label_path))
        points = points.astype(float)
        points = torch.Tensor(points)
        if points.numel() == 0:
            points = points.reshape(0, 2)
            print(img_path)

        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # random crop patch
        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        density = self.compute_density(points)

        target = {'points': points, 'labels': torch.ones([points.shape[0]]).long(), 'density': density,}

        return img, target

    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        dist = torch.cdist(points, points, p=2)
        if points.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density


def load_data(img_gt_path):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with open(gt_path, "r") as f:
        annotation = json.load(f)
    return img, np.array(annotation['points'])


def build_nwpu_datasets():
    # build dataset
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = NWPU(split="train", transform=transform,
                         path="/home/user/src_wsl/multi-resolution-crowd-counting/PET-main/data/NWPU-Crowd_2048")
    dataset_val = NWPU(split="val", transform=transform,
                       path="/home/user/src_wsl/multi-resolution-crowd-counting/PET-main/data/NWPU-Crowd_2048")
    return  dataset_train, dataset_val


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size

    # compute crop coordinates
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w

    # crop image (always)
    result_img = img[:, start_h:end_h, start_w:end_w]

    # empty points case
    if points.shape[0] == 0:
        return result_img, points

    # filter points
    idx = (
        (points[:, 0] >= start_h) & (points[:, 0] <= end_h) &
        (points[:, 1] >= start_w) & (points[:, 1] <= end_w)
    )

    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w

    # resize to patch size
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW

    result_img = torch.nn.functional.interpolate(
        result_img.unsqueeze(0),
        (patch_h, patch_w)
    ).squeeze(0)

    result_points[:, 0] *= fH
    result_points[:, 1] *= fW

    return result_img, result_points