"""Handwritten digit recognition with a convolutional neural network trained on MNIST."""

import importlib.util
import os

__version__ = "2.0.0"


def _select_keras_backend() -> None:
    """Point Keras at an installed backend when the user hasn't picked one.

    Keras defaults to TensorFlow, which doesn't publish wheels for the newest
    Python releases. When TensorFlow is missing we fall back to JAX (the
    backend in requirements.txt) or PyTorch, whichever is installed.
    """
    if "KERAS_BACKEND" in os.environ or importlib.util.find_spec("tensorflow"):
        return
    for backend in ("jax", "torch"):
        if importlib.util.find_spec(backend):
            os.environ["KERAS_BACKEND"] = backend
            return


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's C++ start-up chatter
_select_keras_backend()
