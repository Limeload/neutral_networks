import numpy as np
import tensorflow as tf
from PIL import Image

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


def load_and_preprocess(image_path: str, target_size: tuple) -> np.ndarray:
    img = Image.open(image_path).convert('RGB').resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


def predict(model: tf.keras.Model, image_path: str, target_size: tuple) -> dict:
    img   = load_and_preprocess(image_path, target_size)
    probs = model.predict(img[np.newaxis, ...], verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        'class':        CLASSES[idx],
        'confidence':   float(probs[idx]),
        'probabilities': {c: float(p) for c, p in zip(CLASSES, probs)},
        'image_array':  img,
    }
