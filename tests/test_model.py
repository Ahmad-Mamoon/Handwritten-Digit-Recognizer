import numpy as np
import pytest

import digit_recognizer  # noqa: F401  (selects a Keras backend before keras is imported)

keras = pytest.importorskip("keras")

from digit_recognizer.model import DigitClassifier, ModelNotFoundError, build_model  # noqa: E402
from digit_recognizer.training import RandomAffine  # noqa: E402


def test_model_outputs_a_probability_per_digit():
    model = build_model()
    probabilities = np.asarray(model.predict_on_batch(np.zeros((2, 28, 28, 1), np.float32)))
    assert probabilities.shape == (2, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_saved_model_round_trips_through_classifier(tmp_path):
    model = build_model()
    path = tmp_path / "model.keras"
    model.save(path)

    classifier = DigitClassifier(path)
    assert classifier.is_available and not classifier.is_loaded
    images = np.random.default_rng(0).random((3, 28, 28), dtype=np.float32)
    np.testing.assert_allclose(
        classifier.predict(images), model.predict_on_batch(images[..., None]), atol=1e-5
    )
    assert classifier.is_loaded


def test_missing_model_raises_a_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train.py"):
        DigitClassifier(tmp_path / "nope.keras").load()


def test_augmentation_only_runs_while_training():
    images = np.zeros((4, 28, 28, 1), np.float32)
    images[:, 10:18, 12:16] = 1.0
    layer = RandomAffine(seed=0)

    np.testing.assert_array_equal(np.asarray(layer(images, training=False)), images)
    augmented = np.asarray(layer(images, training=True))
    assert augmented.shape == images.shape
    assert not np.allclose(augmented, images)
    assert augmented.sum() > 0.5 * images.sum()  # the digit is moved, not lost
