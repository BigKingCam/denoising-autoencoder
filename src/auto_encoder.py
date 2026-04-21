# import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
from os import listdir
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt

# from keras import layers, models, optimizers

# from keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose

# from dataset import Dataset
# from noise import add_noise

from download_dataset import get_cbsd68_path
from download_dataset import get_bsds500_path

from noise import get_noise_fn
from noise import make_denoising_dataset

from dataset import Dataset

BASE_DIR: Path = Path(__file__).resolve().parents[2]

cbsd68_img_folder: Path = get_cbsd68_path()  # CBSD68/
cbsd_ground_truth: Path = cbsd68_img_folder / "original_png"

bsds500_img_folder: Path = get_bsds500_path()  # BSDS500/
bsd500_train: Path = bsds500_img_folder / "train"
bsd500_val: Path = bsds500_img_folder / "val"
bsd500_test: Path = bsds500_img_folder / "test"


def build_image_set(folder: Path) -> list[str]:
    """Builds the image set"""

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    image_paths: list[str] = []

    for file in listdir(folder):
        path: str = join(folder, file)

        if isfile(path) and file.lower().endswith(".jpg"):
            image_paths.append(path)

    return image_paths


def visualize_noise(dataset, noise_fn, num_images=4):
    """
    Shows original vs noisy images from dataset.

    Args:
        dataset: tf.data.Dataset yielding images (x) or (x, y)
        noise_fn: function(x) -> noisy_x
        num_images: number of samples to display
    """
    # Get one batch
    batch = next(iter(dataset))

    # Handle (x, y) or x
    if isinstance(batch, tuple):
        x = batch[0]
    else:
        x = batch

    x = x[:num_images]
    noisy = noise_fn(x)

    x = tf.clip_by_value(x, 0.0, 1.0)
    noisy = tf.clip_by_value(noisy, 0.0, 1.0)

    fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))

    for i in range(num_images):
        axes[0, i].imshow(x[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(noisy[i])
        axes[1, i].set_title("Noisy")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_noise_levels(dataset, base_config, sigmas=[0.05, 0.1, 0.2], num_images=3):
    batch = next(iter(dataset))
    x = batch[0] if isinstance(batch, tuple) else batch
    x = x[:num_images]

    fig, axes = plt.subplots(len(sigmas)+1, num_images, figsize=(3*num_images, 3*(len(sigmas)+1)))

    # Original row
    for i in range(num_images):
        axes[0, i].imshow(x[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

    # Noise rows
    for row, sigma in enumerate(sigmas, start=1):
        config = dict(base_config)
        config["noise"] = dict(base_config["noise"])
        config["noise"]["sigma"] = sigma

        noise_fn = get_noise_fn(config)
        noisy = noise_fn(x)

        for col in range(num_images):
            axes[row, col].imshow(tf.clip_by_value(noisy[col], 0, 1))
            axes[row, col].set_title(f"σ={sigma}")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# dataset init: train
train_imgs: list[str] = build_image_set(bsd500_train)

train_ds = Dataset(
    image_paths=train_imgs,
    patch_size=64,
    sigma=25,          # this becomes 25/255 internally
    batch_size=32,
    training=True
)

# ex config
test_config = {"noise": {"type": "gaussian", "sigma": 0.2}}

noise_fn = get_noise_fn(test_config)
visualize_noise(train_ds, noise_fn)
visualize_noise_levels(train_ds, test_config)


# def load_datasets():
#     train_imgs: list[str] = build_image_set(bsd500_train)
#     val_imgs: list[str] = build_image_set(bsd500_val)
#     test_imgs: list[str] = build_image_set(bsd500_test)
#     return train_imgs, val_imgs, test_imgs


if __name__ == "__main__":
    train_imgs: list[str] = build_image_set(bsd500_train)
    val_imgs: list[str] = build_image_set(bsd500_val)
    test_imgs: list[str] = build_image_set(bsd500_test)
