"""Project-wide paths and constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

#: Where the trained model lives. Override with the DIGIT_MODEL_PATH environment variable.
MODEL_PATH = Path(os.environ.get("DIGIT_MODEL_PATH", MODELS_DIR / "digit_model.keras"))

#: The static website (also what gets deployed to GitHub Pages).
WEB_DIR = PROJECT_ROOT / "web"
#: The model exported for the browser: model.json (architecture) + weights.bin.
WEB_MODEL_DIR = WEB_DIR / "model"

IMAGE_SIZE = 28  # MNIST images are 28x28 pixels...
DIGIT_BOX = 20  # ...with each digit scaled to fit a 20x20 box inside the frame.
NUM_CLASSES = 10
