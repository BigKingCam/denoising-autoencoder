# import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
from os import listdir
from os.path import isfile, join
from pathlib import Path

from keras import layers, Model, optimizers, Sequential

from keras.layers import Dense, Input, Reshape, Flatten, Conv2D, Conv2DTranspose

from keras.constraints import max_norm

from dataset import Dataset
# from noise import add_noise

from download_dataset import get_cbsd68_path
from download_dataset import get_bsds500_path

BASE_DIR: Path = Path(__file__).resolve().parents[2]

cbsd68_img_folder: Path = get_cbsd68_path()
cbsd_ground_truth: Path = cbsd68_img_folder / "original_png"

bsds500_img_folder: Path = get_bsds500_path()
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
        if isfile(path):
            image_paths.append(path)
    return image_paths


# The model architecture that we implemented is referenced from:
# Francesco Franco: "Building an Image Denoiser with a Keras Autoencoder"
# https://pub.aimind.so/building-an-image-denoiser-with-a-keras-autoencoder-ead8d55e047f
# Found on: 4/21/2026

#NOTE: This code has been adapted and modified for this project

def conv_model(input_shape=None) -> Model:

    max_norm_value = 2.0 #2.0 indicates the weight constraint, moderate regularization. smaller value = stricter constraint, larger value = looser constraint

    if input_shape is None:
        input_shape=(Dataset.patch_size, Dataset.patch_size, 3)

    # Create the  model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2DTranspose(32, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2DTranspose(64, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
                    #3 is for RGB channels, was originally '1' for greyscale 
    model.add(Conv2D(3, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))

    model.compile(
        optimizer='adam',   # adam is standard for autoencoders
        loss='mse'          # mean squared error — standard for image reconstruction
    )

    model.summary()

    return model
  
def train_model(model, noisy_patch, clean_patch) -> Model:
    model.fit( #fit is for training the model with the given inputs
        x = noisy_patch,
        y = clean_patch,
        batch_size = Dataset.batch_size,
        epochs = 3, #idk how many is good???
        verbose = 2, #0:no output, 1:progress bar, 2:shows one line per epoch
    )
      
    return model


if __name__ == "__main__":
    training_imgs: list[str] = build_image_set(bsd500_train)
    validation_imgs: list[str] = build_image_set(bsd500_val)

    dataset = Dataset(training_imgs)
    noisy, clean = Dataset.get_patches()

    model = conv_model()
    trained_model = train_model(model, noisy, clean)