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
