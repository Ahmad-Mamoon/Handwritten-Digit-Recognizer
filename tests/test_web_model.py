"""The in-browser code (web/js) must behave exactly like the Python/Keras version."""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

import digit_recognizer  # noqa: F401  (selects a Keras backend before keras is imported)
from digit_recognizer.config import MODEL_PATH, WEB_MODEL_DIR
from digit_recognizer.model import metadata_path
from digit_recognizer.preprocessing import preprocess

NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("web_harness.mjs")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the browser code")


def run_js(job: dict) -> dict:
    completed = subprocess.run(
        [NODE, str(HARNESS)], input=json.dumps(job), capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def rgba_job(image: Image.Image) -> dict:
    rgba = image.convert("RGBA")
    return {"width": rgba.width, "height": rgba.height, "rgba": base64.b64encode(rgba.tobytes()).decode()}


def seven_with_faint_line(draw_seven):
    # 51/255 is exactly the 0.2 ink threshold in float32: it must not count as ink in either language.
    image = draw_seven()
    ImageDraw.Draw(image).line([(240, 30), (240, 260)], fill=51, width=3)
    return image


@needs_node
def test_js_preprocessing_matches_python(draw_seven):
    images = [
        seven_with_faint_line(draw_seven),
        draw_seven(),
        draw_seven(left=10, top=10, size=60),
        draw_seven(left=150, top=150, size=90),
        draw_seven(canvas=301, left=33, top=71, size=137),
        draw_seven().filter(ImageFilter.GaussianBlur(4)),  # soft, anti-aliased edges
        draw_seven(ink=0, background=255),
        draw_seven(ink=(40, 40, 50), background=(200, 195, 190), mode="RGB"),
        draw_seven(ink=(0, 0, 0, 255), background=(0, 0, 0, 0), mode="RGBA"),
        Image.new("L", (280, 280), 0),
    ]
    results = run_js({"images": [rgba_job(image) for image in images]})["preprocessed"]

    for image, js in zip(images, results):
        expected = preprocess(image)
        if expected is None:
            assert js is None
        else:
            # Pillow resizes with fixed-point maths, so allow a few levels of rounding difference.
            np.testing.assert_allclose(np.reshape(js, (28, 28)), expected, atol=4 / 255)


@pytest.fixture
def keras():
    return pytest.importorskip("keras")


def randomise_batchnorm(model, keras, rng):
    """Give BatchNorm layers non-trivial statistics so folding them is really tested."""
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            n = layer.moving_mean.shape[0]
            layer.moving_mean.assign(rng.normal(0, 0.5, n).astype("float32"))
            layer.moving_variance.assign(rng.uniform(0.5, 2.0, n).astype("float32"))
            layer.gamma.assign(rng.uniform(0.5, 1.5, n).astype("float32"))
            layer.beta.assign(rng.normal(0, 0.2, n).astype("float32"))


def test_export_folds_batchnorm_and_drops_dropout(keras, tmp_path):
    from digit_recognizer.export import export_keras_model
    from digit_recognizer.model import build_model

    manifest_path = export_keras_model(build_model(), tmp_path, {"test_accuracy": 0.5})
    manifest = json.loads(manifest_path.read_text())

    types = [layer["type"] for layer in manifest["layers"]]
    assert types == ["conv2d", "maxpool2d"] * 3 + ["flatten", "dense", "dense"]
    assert [layer.get("activation") for layer in manifest["layers"] if "activation" in layer] == ["relu"] * 4 + ["softmax"]
    assert manifest["metadata"] == {"test_accuracy": 0.5}
    total = sum(layer[name]["length"] for layer in manifest["layers"] for name in ("kernel", "bias") if name in layer)
    assert (tmp_path / "weights.bin").stat().st_size == total * 4


def test_unsupported_layers_are_rejected(keras, tmp_path):
    from digit_recognizer.export import export_keras_model

    model = keras.Sequential([keras.Input((28, 28, 1)), keras.layers.AveragePooling2D(2), keras.layers.Flatten(), keras.layers.Dense(10)])
    with pytest.raises(ValueError, match="AveragePooling2D"):
        export_keras_model(model, tmp_path)


@needs_node
def test_js_model_matches_keras(keras, tmp_path):
    from digit_recognizer.export import export_keras_model
    from digit_recognizer.model import build_model

    rng = np.random.default_rng(0)
    model = build_model()
    randomise_batchnorm(model, keras, rng)
    export_keras_model(model, tmp_path)

    images = rng.random((4, 28, 28), dtype=np.float32)
    images[0] = 0  # an all-black input
    images[1] = images[1] > 0.8  # a sparse, mostly-zero input like a real drawing
    expected = np.asarray(model.predict_on_batch(images[..., None]))
    js = np.array(run_js({"model_dir": str(tmp_path), "inputs": images.reshape(4, -1).tolist()})["probabilities"])

    # Compare log-probabilities: they expose small logit differences that softmax hides.
    np.testing.assert_allclose(np.log(js), np.log(expected), atol=1e-3)


@needs_node
@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="no trained model")
def test_trained_model_runs_identically_in_the_browser(keras, tmp_path, draw_seven):
    from digit_recognizer.export import export_web_model
    from digit_recognizer.model import DigitClassifier

    export_web_model(MODEL_PATH, tmp_path)
    digit = preprocess(draw_seven())
    expected = DigitClassifier(MODEL_PATH).predict(digit)[0]
    js = run_js({"model_dir": str(tmp_path), "inputs": [digit.ravel().tolist()]})["probabilities"][0]
    np.testing.assert_allclose(js, expected, atol=1e-4)
    assert int(np.argmax(js)) == 7


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and (WEB_MODEL_DIR / "model.json").is_file()),
    reason="needs both the trained model and the website export",
)
def test_website_model_is_up_to_date():
    """Catch a retrained model that wasn't re-exported for the website."""
    exported = json.loads((WEB_MODEL_DIR / "model.json").read_text(encoding="utf-8"))["metadata"]
    trained = json.loads(metadata_path(MODEL_PATH).read_text(encoding="utf-8"))
    assert exported == trained, "web/model/ is stale: run `python export_web_model.py`"
