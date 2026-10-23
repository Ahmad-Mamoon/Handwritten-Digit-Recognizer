import base64
import io

import pytest
from PIL import Image

from digit_recognizer.model import DigitClassifier
from digit_recognizer.server import create_app


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(image):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode()


@pytest.fixture
def app(classifier):
    app = create_app(classifier)
    app.testing = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_page_is_the_static_site(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'<canvas id="canvas"' in response.data
    assert b'<script type="module" src="js/app.js">' in response.data


@pytest.mark.parametrize("path", ["/js/app.js", "/js/model.js", "/js/preprocess.js"])
def test_scripts_are_served_as_javascript(client, path):
    # Browsers refuse to run module scripts served as text/plain (a common Windows default).
    response = client.get(path)
    assert response.status_code == 200
    assert response.mimetype == "text/javascript"


def test_stylesheet_is_served(client):
    assert client.get("/css/style.css").status_code == 200


def test_unknown_api_route_is_404(client):
    assert client.get("/api/nope").status_code == 404


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["model"]["test_accuracy"] == 0.995


def test_predict_from_json_data_url(client, classifier, draw_seven):
    response = client.post("/api/predict", json={"image": data_url(draw_seven())})
    assert response.status_code == 200
    body = response.get_json()
    assert body["digit"] == 7
    assert body["confidence"] == pytest.approx(0.91)
    assert len(body["probabilities"]) == 10
    assert body["input_image"].startswith("data:image/png;base64,")
    # The model must receive one normalised 28x28 image, not the raw canvas.
    (images,) = classifier.received
    assert images.shape == (1, 28, 28)
    assert 0.9 < images.max() <= 1.0


def test_predict_from_raw_base64_form_field(client, draw_seven):
    encoded = base64.b64encode(png_bytes(draw_seven())).decode()
    response = client.post("/api/predict", data={"image": encoded})
    assert response.status_code == 200
    assert response.get_json()["digit"] == 7


def test_predict_from_file_upload(client, draw_seven):
    upload = (io.BytesIO(png_bytes(draw_seven(ink=0, background=255))), "seven.png")
    response = client.post("/api/predict", data={"file": upload}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.get_json()["digit"] == 7


def test_blank_canvas_is_rejected(client):
    response = client.post("/api/predict", json={"image": data_url(Image.new("RGB", (280, 280)))})
    assert response.status_code == 422
    assert "No digit" in response.get_json()["error"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"image": ""}, {"image": 123}, ["not", "an", "object"]],
)
def test_missing_image_is_a_bad_request(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_base64_is_a_bad_request(client):
    response = client.post("/api/predict", json={"image": "data:image/png;base64,???"})
    assert response.status_code == 400
    assert "base64" in response.get_json()["error"]


def test_non_image_data_is_a_bad_request(client):
    response = client.post("/api/predict", json={"image": base64.b64encode(b"hello world").decode()})
    assert response.status_code == 400
    assert "Couldn't read" in response.get_json()["error"]


def test_oversized_upload_returns_json_413(app, client):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    upload = (io.BytesIO(b"0" * 4096), "big.png")
    response = client.post("/api/predict", data={"file": upload}, content_type="multipart/form-data")
    assert response.status_code == 413
    assert "error" in response.get_json()


def test_huge_image_dimensions_are_rejected(client):
    tiny_file_huge_image = png_bytes(Image.new("1", (6000, 6000)))
    response = client.post("/api/predict", json={"image": base64.b64encode(tiny_file_huge_image).decode()})
    assert response.status_code == 413
    assert "too large" in response.get_json()["error"]


def test_unexpected_errors_return_json_500(app, client, classifier, draw_seven):
    def explode(images):
        raise RuntimeError("boom")

    classifier.predict = explode
    response = client.post("/api/predict", json={"image": data_url(draw_seven())})
    assert response.status_code == 500
    assert response.get_json() == {"error": "Something went wrong on the server."}


def test_missing_model_is_reported(tmp_path, draw_seven):
    app = create_app(DigitClassifier(tmp_path / "missing.keras"))
    client = app.test_client()

    assert client.get("/").status_code == 200  # the website itself doesn't need the Keras model
    assert client.get("/api/health").get_json()["model_available"] is False

    response = client.post("/api/predict", json={"image": data_url(draw_seven())})
    assert response.status_code == 503
    assert "train.py" in response.get_json()["error"]
