"""Local web server: serves the website in web/ and a JSON prediction API.

The website itself runs the model in the browser, so it works on any static
host such as GitHub Pages. This server is for local use and for programs that
want predictions over HTTP (``POST /api/predict``).
"""

from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import re

import numpy as np
from flask import Blueprint, Flask, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from .config import MODEL_PATH, WEB_DIR
from .model import DigitClassifier, ModelNotFoundError
from .preprocessing import preprocess

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 25_000_000  # a small file can still decode to a huge image
DATA_URL_PREFIX = re.compile(r"^data:[^,]*;base64,", re.IGNORECASE)

# Windows can map .js to text/plain in the registry, and browsers refuse to run
# module scripts served with that type.
mimetypes.add_type("text/javascript", ".js")

api = Blueprint("api", __name__, url_prefix="/api")


def create_app(classifier: DigitClassifier | None = None) -> Flask:
    """Build the Flask app. Pass ``classifier`` to use a different (or fake) model."""
    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.extensions["digit_classifier"] = classifier or DigitClassifier(MODEL_PATH)
    app.add_url_rule("/", "index", lambda: app.send_static_file("index.html"))
    app.register_blueprint(api)
    return app


class ApiError(Exception):
    """An error reported to the client as ``{"error": message}`` with an HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _classifier() -> DigitClassifier:
    return current_app.extensions["digit_classifier"]


@api.get("/health")
def health():
    classifier = _classifier()
    return jsonify(
        status="ok",
        model_available=classifier.is_available,
        model_loaded=classifier.is_loaded,
        model=classifier.metadata(),
    )


@api.post("/predict")
def predict():
    """Predict the digit in an image.

    Send either JSON ``{"image": "<data URL or base64>"}``, a form field named
    ``image`` with the same content, or a multipart file upload named ``file``.
    """
    classifier = _classifier()
    if not classifier.is_available:
        raise ApiError("The model hasn't been trained yet. Run `python train.py` first.", 503)

    digit_image = preprocess(_read_image())
    if digit_image is None:
        raise ApiError("No digit found in the image. Draw something first!", 422)

    try:
        probabilities = classifier.predict(digit_image)[0]
    except ModelNotFoundError as exc:
        raise ApiError(str(exc), 503) from exc

    digit = int(np.argmax(probabilities))
    return jsonify(
        digit=digit,
        confidence=round(float(probabilities[digit]), 4),
        probabilities=[round(float(p), 4) for p in probabilities],
        input_image=_to_data_url(digit_image),
    )


@api.errorhandler(Exception)
def handle_api_error(exc: Exception):
    if isinstance(exc, ApiError):
        return jsonify(error=exc.message), exc.status
    if isinstance(exc, HTTPException):  # e.g. 413 when the upload is too large
        return jsonify(error=exc.description), exc.code
    current_app.logger.exception("Unhandled error in %s", request.path)
    return jsonify(error="Something went wrong on the server."), 500


def _read_image() -> Image.Image:
    upload = request.files.get("file") or request.files.get("image")
    if upload:
        data = upload.read()
    else:
        payload = request.get_json(silent=True) if request.is_json else request.form
        encoded = payload.get("image") if isinstance(payload, dict) else None
        if not encoded or not isinstance(encoded, str):
            raise ApiError('No image provided. Send JSON like {"image": "data:image/png;base64,..."} or upload a file.')
        try:
            data = base64.b64decode(DATA_URL_PREFIX.sub("", encoded.strip()), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ApiError("The image isn't valid base64.") from exc

    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise ApiError("That image is too large. Please use one under 25 megapixels.", 413)
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ApiError("Couldn't read that image. Try a PNG, JPEG, GIF, BMP or WebP file.") from exc
    return image


def _to_data_url(image: np.ndarray) -> str:
    """Encode the 28x28 model input as a PNG data URL so clients can show it."""
    buffer = io.BytesIO()
    Image.fromarray(np.uint8(np.round(image * 255))).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
