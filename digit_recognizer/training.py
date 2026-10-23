"""Train the digit CNN on MNIST and save it for the web and desktop apps."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
from keras import layers, ops

from .config import MODEL_PATH
from .model import build_model, metadata_path


def load_mnist():
    """Return MNIST as float32 images in [0, 1] shaped (N, 28, 28, 1), plus integer labels."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype(np.float32) / 255, -1)
    x_test = np.expand_dims(x_test.astype(np.float32) / 255, -1)
    return (x_train, y_train), (x_test, y_test)


class RandomAffine(layers.Layer):
    """Randomly rotate, zoom and shift images during training.

    Teaches the model to cope with how people really draw. All three changes
    are combined into one affine warp, which is several times faster than
    chaining Keras' RandomRotation, RandomZoom and RandomTranslation layers.
    """

    def __init__(self, rotation: float = 15.0, zoom: float = 0.1, shift: float = 0.1, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rotation = math.radians(rotation)
        self.zoom = zoom
        self.shift = shift
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, images, training=None):
        if not training:
            return images
        batch = ops.shape(images)[0]
        height, width = images.shape[1], images.shape[2]

        def uniform(limit):
            return keras.random.uniform((batch,), -limit, limit, seed=self.seed_generator)

        angle = uniform(self.rotation)
        scale = 1.0 + uniform(self.zoom)
        tx, ty = uniform(self.shift) * width, uniform(self.shift) * height

        # affine_transform maps each output pixel back to an input pixel, so build the
        # inverse of "rotate and scale about the centre, then shift".
        cos, sin = ops.cos(angle) / scale, ops.sin(angle) / scale
        cx, cy = (width - 1) / 2, (height - 1) / 2
        zeros = ops.zeros_like(cos)
        transform = ops.stack(
            [
                cos, sin, cx - cos * (cx + tx) - sin * (cy + ty),
                -sin, cos, cy + sin * (cx + tx) - cos * (cy + ty),
                zeros, zeros,
            ],
            axis=1,
        )  # fmt: skip
        return ops.image.affine_transform(images, transform, fill_mode="constant", fill_value=0)


def train(
    output: str | Path = MODEL_PATH,
    epochs: int = 15,
    batch_size: int = 128,
    validation_split: float = 0.1,
    augment: bool = True,
    seed: int = 42,
) -> dict:
    """Train a fresh model, save it to ``output`` and return its metadata (test accuracy etc.)."""
    keras.utils.set_random_seed(seed)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Augmentation wraps the classifier rather than living inside it, so the saved
    # model stays small, inference-only and free of custom layers.
    classifier = build_model()
    trainer_layers = [keras.Input(shape=x_train.shape[1:])]
    if augment:
        trainer_layers.append(RandomAffine(seed=seed))
    trainer = keras.Sequential([*trainer_layers, classifier], name="trainer")
    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    classifier.summary()

    history = trainer.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ],
    )

    test_loss, test_accuracy = trainer.evaluate(x_test, y_test, verbose=0)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(output)
    metadata = {
        "test_accuracy": round(float(test_accuracy), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs": len(history.history["loss"]),
        "augmented": augment,
        "parameters": classifier.count_params(),
        "keras_version": keras.__version__,
        "backend": keras.backend.backend(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metadata_path(output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
