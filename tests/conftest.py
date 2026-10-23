import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def draw_seven():
    """Return a function that draws a "7" with the given colours, size and position."""

    def draw(canvas=280, ink=255, background=0, mode="L", left=60, top=50, size=160):
        image = Image.new(mode, (canvas, canvas), background)
        pen = ImageDraw.Draw(image)
        width = max(2, size // 9)
        pen.line([(left, top), (left + size * 0.7, top)], fill=ink, width=width)
        pen.line([(left + size * 0.7, top), (left + size * 0.3, top + size)], fill=ink, width=width)
        return image

    return draw


class FakeClassifier:
    """Stands in for the Keras model so web tests don't need a trained model."""

    model_path = "fake.keras"
    is_available = True
    is_loaded = True

    def __init__(self):
        self.received = []

    def predict(self, images):
        images = np.asarray(images).reshape(-1, 28, 28)
        self.received.append(images)
        probabilities = np.full((len(images), 10), 0.01, dtype=np.float32)
        probabilities[:, 7] = 0.91
        return probabilities

    def metadata(self):
        return {"test_accuracy": 0.995, "parameters": 1234, "keras_version": "3.x", "backend": "fake"}


@pytest.fixture
def classifier():
    return FakeClassifier()
