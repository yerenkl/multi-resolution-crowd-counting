import os
import random

import numpy as np
import torchvision.transforms as tf
from PIL import ImageFilter, Image


class MethodWeights:

    def __init__(self, bilinear: float, bicubic: float, nearest: float, lanczos: float):
        total = bilinear + bicubic + nearest + lanczos
        self.bilinear = bilinear
        self.bicubic = bicubic
        self.nearest = nearest
        self.lanczos = lanczos

    @staticmethod
    def bilinear_only():
        return MethodWeights(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def bicubic_only():
        return MethodWeights(0.0, 1.0, 0.0, 0.0)

    @staticmethod
    def nearest_only():
        return MethodWeights(0.0, 0.0, 1.0, 0.0)

    @staticmethod
    def lanczos_only():
        return MethodWeights(0.0, 0.0, 0.0, 1.0)

    def as_list(self):
        return [self.bilinear, self.bicubic, self.nearest, self.lanczos]


def gaussian_blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize(img, scale, method='bilinear'):
    methods = {
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'nearest': Image.Resampling.NEAREST,
        'lanczos': Image.Resampling.LANCZOS
    }

    new_size = (int(img.width * scale), int(img.height * scale))

    return img.resize(new_size, methods[method])


def transform(img, pre_downsampling_blur: bool = True, downsample_factor: int = 4,
              method_weights: MethodWeights = MethodWeights(0.25, 0.25, 0.25, 0.25), upsample: bool = False,
              add_noise: bool = False):
    if pre_downsampling_blur:
        img = gaussian_blur(img, sigma=random.uniform(0.3, 1.3))

    methods = ['bilinear', 'bicubic', 'nearest', 'lanczos']
    method = random.choices(methods, weights=method_weights.as_list(), k=1)[0]
    img = resize(img, scale=1 / downsample_factor, method=method)

    if upsample:
        img = resize(img, downsample_factor, method="bilinear")

    if add_noise:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 1.5, arr.shape)
        img = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # img = tf.ToTensor()(img)

    return img


if __name__ == "__main__":
    img = Image.open("0040.jpg")
    img = img.convert("RGB")

    methods_weights = [
        MethodWeights.bilinear_only(),
        MethodWeights.bicubic_only(),
        MethodWeights.nearest_only(),
        MethodWeights.lanczos_only()
    ]

    base_name = "output_image"

    for i, method in enumerate(methods_weights):
        for blur in [True]:
            for downsample_factor in [4]:
                for upsample in [False]:
                    for add_noise in [False, True]:
                        out_img = transform(
                            img,
                            method_weights=method,
                            upsample=upsample,
                            pre_downsampling_blur=blur,
                            downsample_factor=downsample_factor,
                            add_noise=add_noise
                        )
                        out_img = tf.ToPILImage()(out_img)

                        blur_tag = "blur" if blur else "noblur"
                        method_tag = "bilinear" if method.bilinear == 1.0 else "bicubic" if method.bicubic == 1.0 else "nearest" if method.nearest == 1.0 else "lanczos"
                        upsample_tag = "upsampled" if upsample else "downsampled"
                        add_noise_tag = "noise" if add_noise else "nonoise"

                        filename = f"{base_name}_x{downsample_factor}_{method_tag}_{blur_tag}_{upsample_tag}_{add_noise_tag}.png"
                        save_path = os.path.join("output", filename)

                        out_img.save(save_path)
                        print(f"Saved: {save_path}")
