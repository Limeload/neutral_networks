import numpy as np
import tensorflow as tf
from PIL import Image

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


def load_and_preprocess(image_path: str, target_size: tuple) -> np.ndarray:
    img = Image.open(image_path).convert('RGB').resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0
