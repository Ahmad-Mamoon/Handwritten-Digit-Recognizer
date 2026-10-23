"""Turn arbitrary digit images into MNIST-style model input.

MNIST digits are light ink on a black background, scaled to fit a 20x20 box
and centred by centre of mass inside a 28x28 frame. The network only performs
well on images prepared the same way. Squashing a whole drawing canvas down
to 28x28 gives a tiny, off-centre digit and poor predictions, so every image
goes through :func:`preprocess` first.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from .config import DIGIT_BOX, IMAGE_SIZE

INK_THRESHOLD = 0.2  # intensity (0-1) that counts as ink when cropping to the digit
NOISE_FLOOR = 0.1  # fainter pixels are zeroed to drop paper texture and JPEG noise
MAX_INPUT_SIDE = 512  # bigger uploads are shrunk first to keep preprocessing fast


def preprocess(image: Image.Image) -> np.ndarray | None:
    """Convert a digit image into a (28, 28) float32 array in [0, 1] for the model.

    Accepts light-on-dark drawings, dark-on-light scans or photos, and images
    with transparent backgrounds. Returns ``None`` when there is no digit.
    """
    return center_digit(to_grayscale(image))


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Return ``image`` as floats in [0, 1] with light ink on a black background."""
    image = ImageOps.exif_transpose(image)  # respect phone camera orientation
    image.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE))

    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255
        gray = np.asarray(rgba.convert("L"), dtype=np.float32) / 255
        # Transparent pixels are background: fill them with whatever contrasts with the ink.
        opaque = alpha > 0.5
        ink_is_dark = opaque.any() and gray[opaque].mean() < 0.5
        gray = gray * alpha + (1.0 - alpha) * (1.0 if ink_is_dark else 0.0)
    else:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255

    # The border is almost all background; if it's light, the ink is dark, so invert.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    if background > 0.5:
        gray = 1.0 - gray
        background = 1.0 - background

    # Stretch contrast so the background becomes 0 and the strongest ink 1.
    peak = float(gray.max())
    if peak - background < 1e-3:
        return np.zeros_like(gray)
    gray = np.clip((gray - background) / (peak - background), 0.0, 1.0)
    gray[gray < NOISE_FLOOR] = 0.0
    return gray


def center_digit(gray: np.ndarray) -> np.ndarray | None:
    """Crop the ink, fit it into a 20x20 box and centre it by mass in a 28x28 frame."""
    ink = gray > INK_THRESHOLD
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        return None
    digit = gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    # Scale the longest side to DIGIT_BOX pixels, keeping the aspect ratio (a "1" stays thin).
    height, width = digit.shape
    scale = DIGIT_BOX / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(np.uint8(np.round(digit * 255))).resize(new_size, Image.Resampling.LANCZOS)
    digit = np.asarray(resized, dtype=np.float32) / 255

    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    top = (IMAGE_SIZE - digit.shape[0]) // 2
    left = (IMAGE_SIZE - digit.shape[1]) // 2
    frame[top : top + digit.shape[0], left : left + digit.shape[1]] = digit

    total = frame.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(frame.shape)
    centre = (IMAGE_SIZE - 1) / 2
    dy = round(centre - (ys * frame).sum() / total)
    dx = round(centre - (xs * frame).sum() / total)
    return _shift(frame, dy, dx)


def _shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate ``image`` by (dy, dx) pixels, filling the uncovered area with zeros."""
    height, width = image.shape
    shifted = np.zeros_like(image)
    shifted[max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)] = image[
        max(-dy, 0) : height + min(-dy, 0), max(-dx, 0) : width + min(-dx, 0)
    ]
    return shifted
