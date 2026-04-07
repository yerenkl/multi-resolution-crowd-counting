import os
import json
import torch
from PIL import Image
import torchvision.transforms as T

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(labels)

class NWPU(torch.utils.data.Dataset):
    def __init__(self, transform, split="train"):
        self.dataset_path = "/dtu/blackhole/02/137570/MultiRes/NWPU_crowd"

        with open(os.path.join(self.dataset_path, f"{split}.txt"), "r") as f:
            self.image_paths = [line.strip().split()[0] for line in f.readlines()]

        self.transforms = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_id = self.image_paths[idx]
        img_path = os.path.join(self.dataset_path, "images", image_id + ".jpg")
        label_path = os.path.join(self.dataset_path, "jsons", image_id + ".json")

        # Load annotations
        with open(label_path, "r") as f:
            annotation = json.load(f)

        labels = torch.tensor(annotation["points"], dtype=torch.float32)

        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image
        if self.transforms is not None:
            img = self.transforms(img)
            # Scale labels to match resized image
            _, new_h, new_w = img.shape
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            
            if labels.numel() > 0:
                labels[:, 0] *= scale_x
                labels[:, 1] *= scale_y

        return img, labels


if __name__ == "__main__":
    # Example usage
    dataset = NWPU(split="train", transform=T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor()
    ]))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    imgs, labels = next(iter(dataloader))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10), dpi=150)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        if label.numel() > 0:
            plt.scatter(label[:, 0], label[:, 1], c='red', s=10)
        plt.axis('off')

    plt.savefig("/work3/s252653/multi-resolution-crowd-counting/results/sample_image.png")