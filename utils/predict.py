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


def compute_saliency(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    tensor = tf.Variable(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        preds     = model(tensor)
        top_class = tf.argmax(preds[0])
        loss      = preds[:, top_class]
    grads    = tape.gradient(loss, tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
