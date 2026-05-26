"""Train Xception (transfer learning) and Custom SE-ResNet CNN on the Brain Tumor dataset."""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR  = os.path.join(os.path.dirname(__file__), 'data', 'brain_tumor_dataset', 'Training')
TEST_DIR   = os.path.join(os.path.dirname(__file__), 'data', 'brain_tumor_dataset', 'Testing')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
BATCH_SIZE = 32

os.makedirs(MODELS_DIR, exist_ok=True)


def make_generators(image_size: tuple, augment: bool = True):
    aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        validation_split=0.15,
    ) if augment else ImageDataGenerator(rescale=1.0 / 255, validation_split=0.15)
    test_gen_cfg = ImageDataGenerator(rescale=1.0 / 255)

    kw = dict(target_size=image_size, batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
    train_gen = aug.flow_from_directory(TRAIN_DIR, subset='training',  shuffle=True,  **kw)
    val_gen   = aug.flow_from_directory(TRAIN_DIR, subset='validation', shuffle=False, **kw)
    test_gen  = test_gen_cfg.flow_from_directory(TEST_DIR, shuffle=False, **kw)
    print('Class indices:', train_gen.class_indices)
    return train_gen, val_gen, test_gen
