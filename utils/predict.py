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
    """Return a normalised saliency map for *image* w.r.t. the model's top prediction.

    Preconditions:
    - image: np.ndarray, shape (H, W, 3), single RGB image without a batch dimension
    - dtype: float32 or float64; values in [0.0, 1.0] (i.e. rescaled, not raw uint8)
    - H and W must match the spatial dimensions the model expects (model.input_shape[1:3])
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f'image must be a numpy ndarray, got {type(image).__name__}.'
        )
    if image.ndim != 3:
        raise ValueError(
            f'image must be 3-D (H, W, C), got shape {image.shape}. '
            'Pass a single image without a batch dimension.'
        )
    if image.shape[-1] != 3:
        raise ValueError(
            f'image must have 3 channels (RGB), got {image.shape[-1]}.'
        )
    expected_h, expected_w = model.input_shape[1], model.input_shape[2]
    if image.shape[0] != expected_h or image.shape[1] != expected_w:
        raise ValueError(
            f'image spatial dimensions {image.shape[:2]} do not match the model\'s '
            f'expected input size {(expected_h, expected_w)}. '
            'Resize the image before calling compute_saliency.'
        )
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(
            f'image values must be in [0.0, 1.0] '
            f'(got min={image.min():.4f}, max={image.max():.4f}). '
            'Rescale with image / 255.0 before calling compute_saliency.'
        )

    tensor = tf.Variable(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        preds     = model(tensor)
        top_class = tf.argmax(preds[0])
        loss      = preds[:, top_class]
    grads    = tape.gradient(loss, tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
