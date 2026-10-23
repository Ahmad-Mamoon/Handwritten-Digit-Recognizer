"""The CNN architecture and a thread-safe wrapper for loading and running it."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .config import IMAGE_SIZE, MODEL_PATH, NUM_CLASSES


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file doesn't exist yet."""


def build_model():
    """Return the uncompiled convolutional network that classifies 28x28 digits.

    Three conv blocks (28x28 -> 14x14 -> 7x7 -> 3x3) followed by a small dense
    head. It reaches about 99.4% on MNIST while staying quick to train on a CPU.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="digit_cnn")


def metadata_path(model_path: str | Path) -> Path:
    """Training details (accuracy, date, ...) are stored in a JSON file next to the model."""
    return Path(model_path).with_suffix(".json")


class DigitClassifier:
    """Loads the trained Keras model on first use and predicts digits with it.

    Keras is imported lazily so the web app starts instantly, and can report a
    missing model, without paying the deep-learning import cost up front.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return self._model is not None or self.model_path.is_file()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise ModelNotFoundError(
                        f"No trained model at {self.model_path}. Run `python train.py` to create one."
                    )
                import keras

                self._model = keras.models.load_model(self.model_path, compile=False)
        return self._model

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (N, 10) for images of shape (N, 28, 28) or (28, 28)."""
        model = self.load()
        batch = np.asarray(images, dtype=np.float32).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        with self._lock:
            probabilities = model.predict_on_batch(batch)
        return np.asarray(probabilities, dtype=np.float32)

    def warm_up(self) -> None:
        """Load the model and run one prediction so the first real request is fast."""
        self.predict(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

    def metadata(self) -> dict:
        try:
            return json.loads(metadata_path(self.model_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
