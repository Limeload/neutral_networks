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


# ── Xception (Challenge 2 — transfer learning, target ≥99%) ──────────────────

def _build_xception(image_size: tuple) -> tuple:
    base = Xception(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
    base.trainable = False

    inputs = keras.Input(shape=(*image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    return keras.Model(inputs, outputs), base


def train_xception(quick: bool = False) -> keras.Model:
    print('\n' + '=' * 60)
    print('Training Xception — transfer learning (target ≥99%)')
    print('=' * 60)
    image_size = (299, 299)
    save_path  = os.path.join(MODELS_DIR, 'xception_brain_tumor.keras')

    train_gen, val_gen, test_gen = make_generators(image_size)
    model, base = _build_xception(image_size)

    cbs = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    epochs_frozen = 3 if quick else 10
    print(f'\nPhase 1 — frozen base ({epochs_frozen} epochs)…')
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_frozen, callbacks=cbs,
              workers=4, use_multiprocessing=False)

    epochs_ft = 3 if quick else 20
    print(f'\nPhase 2 — fine-tune top-30 Xception layers ({epochs_ft} epochs)…')
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_ft, callbacks=cbs,
              workers=4, use_multiprocessing=False)

    loss, acc = model.evaluate(test_gen)
    print(f'\nXception test accuracy: {acc:.4f}')
    model.save(save_path)
    print(f'Saved → {save_path}')
    return model


# ── Custom SE-ResNet CNN (Challenge 1 — target ≥98%) ─────────────────────────

def _se_block(x, ratio: int = 16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def _conv_bn_relu(x, filters: int, kernel_size: int = 3, strides: int = 1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)


def _build_se_resnet(image_size: tuple) -> keras.Model:
    inputs = keras.Input(shape=(*image_size, 3))
    x = _conv_bn_relu(inputs, 64, 7, strides=2)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    for filters, repeats in [(64, 2), (128, 2), (256, 3), (512, 3)]:
        for _ in range(repeats):
            x = _se_residual_block(x, filters)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(0.1)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def train_custom_cnn(quick: bool = False) -> keras.Model:
    print('\n' + '=' * 60)
    print('Training SE-ResNet CNN (target ≥98%)')
    print('=' * 60)
    image_size = (224, 224)
    save_path  = os.path.join(MODELS_DIR, 'custom_cnn_brain_tumor.keras')

    train_gen, val_gen, test_gen = make_generators(image_size)
    model  = _build_se_resnet(image_size)
    epochs = 5 if quick else 80

    total_steps = len(train_gen) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=total_steps, alpha=1e-6,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
    )

    cbs = [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, verbose=1),
    ]
    print(f'\nTraining for up to {epochs} epochs with cosine decay…')
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cbs,
              workers=4, use_multiprocessing=False)


def _se_residual_block(x, filters: int):
    shortcut = x
    x = _conv_bn_relu(x, filters)
    x = _conv_bn_relu(x, filters)
    x = _se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)
