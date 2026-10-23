# Handwritten Digit Recognizer

Draw a digit and a convolutional neural network (CNN) trained on
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) recognises it in real time.
The network runs **entirely in your browser**, so the site can be hosted for free on **GitHub Pages**.
It's trained with **Keras 3** and includes an optional **Flask** API and a **pygame** desktop app.

**Live demo:**  [here](https://ahmad-mamoon.github.io/Handwritten-Digit-Recognizer/)

## Features

- **Draw with a mouse, finger or stylus** on desktop and mobile
- **Live predictions** that update while you draw (a prediction takes ~6 ms), with a confidence bar for every digit
- **See what the model sees**: a preview of the 28×28 image the network receives
- **Upload, drag-and-drop or paste** a photo or scan of a handwritten digit (dark ink on paper works too)
- **No server needed**: the CNN runs in plain JavaScript (no ML library), so the site works on any static host, and drawings never leave the device
- Undo, clear, keyboard shortcuts, and light and dark themes
- **99.4% accuracy** on the MNIST test set, and training takes about 10 minutes on a laptop CPU
- Optional **JSON API** (`POST /api/predict`), a **desktop app**, a test suite and GitHub Actions CI

## Quick start

You need **Python 3.10 – 3.14**.

```bash
git clone https://github.com/Ahmad-Mamoon/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python train.py                 # once, ~10 min on CPU (skip if web/model/ already exists)
python app.py                   # then open http://127.0.0.1:5000
```

> **Windows with several Python versions?** Create the environment with a specific one,
> for example `py -3.13 -m venv .venv`.

`train.py` saves the Keras model to `models/` and exports it for the website to `web/model/`.

The website is just the static files in `web/`, so any web server can host it:

```bash
python -m http.server 8000 --directory web
```

(Opening `web/index.html` straight from disk doesn't work, because browsers block loading the
model from `file://` pages.)

To try it on your phone, run `python app.py --host 0.0.0.0` and open
`http://<your-computer's-IP>:5000` on a device on the same Wi-Fi network.


## Project structure

```
Handwritten-Digit-Recognizer/
├── web/                       # The website (deployed to GitHub Pages as-is)
│   ├── index.html
│   ├── css/style.css
│   ├── js/
│   │   ├── app.js             # Drawing pad and UI
│   │   ├── preprocess.js      # MNIST-style preprocessing (port of preprocessing.py)
│   │   └── model.js           # Runs the CNN in plain JavaScript
│   └── model/                 # Exported model: model.json + weights.bin
├── app.py                     # Start the local server (website + JSON API)
├── train.py                   # Train the model on MNIST (and export it for the website)
├── export_web_model.py        # Re-export an existing model for the website
├── desktop.py                 # Start the pygame desktop app
├── digit_recognizer/          # The Python package
│   ├── config.py              # Paths and constants
│   ├── model.py               # CNN architecture + DigitClassifier (loads the model, predicts)
│   ├── preprocessing.py       # Turns any digit image into MNIST-style 28×28 input
│   ├── training.py            # Training loop and data augmentation
│   ├── export.py              # Keras model -> web/model (folds BatchNorm into the convolutions)
│   ├── server.py              # Flask: serves web/ and the JSON API
│   └── desktop.py             # pygame app
├── models/                    # digit_model.keras + digit_model.json (training results)
├── tests/                     # pytest suite (plus a Node harness for the browser code)
├── .github/workflows/         # tests.yml (CI) and pages.yml (GitHub Pages deployment)
├── requirements.txt           # Runtime dependencies
├── requirements-desktop.txt   # ...plus pygame
└── requirements-dev.txt       # ...plus pytest
```

## How it works

1. **Drawing.** The canvas records white strokes on black, the same polarity as MNIST.
2. **Preprocessing** (`web/js/preprocess.js`, and `digit_recognizer/preprocessing.py` for Python).
   MNIST digits were scaled to fit a 20×20 box and centred by centre of mass in a 28×28 frame.
   Every image gets the same treatment: it's cropped to the ink, scaled while keeping its aspect
   ratio, and shifted to the centre. That's why it doesn't matter where or how big you draw.
   It also inverts dark-on-light images, handles transparent backgrounds and stretches the
   contrast of photos. Tests check that the JavaScript and Python versions produce the same pixels.
3. **Model** (`digit_recognizer/model.py`). Three convolution blocks (32 → 64 → 128 filters,
   each Conv → BatchNorm → ReLU → MaxPool) and a 128-unit dense layer with dropout, ending in
   softmax over the 10 digits. It has about 243k parameters.
4. **Training** (`digit_recognizer/training.py`). Random rotations (±15°), zooms (±10%) and shifts
   (±10%) make the network robust to real handwriting. Training uses Adam with early stopping
   and learning-rate decay. The test accuracy is saved to `models/digit_model.json`.
5. **In the browser** (`digit_recognizer/export.py` → `web/js/model.js`). The exporter folds each
   BatchNorm into the convolution before it and drops dropout, which inference doesn't use. What's
   left is five simple operations (convolution, max-pooling, flatten, dense, relu/softmax), which
   `model.js` implements in about 100 lines. Tests check that it matches Keras's output.

### Training options

```bash
python train.py --epochs 5                       # quicker, slightly less accurate
python train.py --no-augment                     # faster, but worse on real handwriting
python train.py --output models/experiment.keras --no-export   # don't touch the website's model
python train.py --help
```

## API (local server only)

`python app.py` also serves a JSON API, for other programs that want predictions over HTTP.

### `POST /api/predict`

Send the image in any of these forms:

| Format    | Body                                                    |
| --------- | ------------------------------------------------------- |
| JSON      | `{"image": "data:image/png;base64,..."}` (raw base64 also works) |
| Form      | field `image` with the same content                     |
| Multipart | file field `file`                                       |

```bash
curl -F "file=@my_digit.png" http://127.0.0.1:5000/api/predict
```

```json
{
  "digit": 7,
  "confidence": 0.9981,
  "probabilities": [0.0, 0.0001, 0.0012, 0.0, 0.0, 0.0, 0.0, 0.9981, 0.0, 0.0006],
  "input_image": "data:image/png;base64,..."
}
```

`input_image` is the 28×28 image the model actually received.
Errors return `{"error": "..."}` with these status codes:

| Status | Meaning                    |
| ------ | -------------------------- |
| `400`  | Missing or unreadable image |
| `413`  | Upload larger than 5 MB    |
| `422`  | No digit found (blank image) |
| `503`  | Model not trained yet      |

### `GET /api/health`

Returns `{"status": "ok", "model_available": true, "model_loaded": true, "model": {...training metadata}}`.

To host the API in production, use any WSGI server with the `app` object in `app.py`:

```bash
pip install waitress && waitress-serve --port=8000 app:app    # Windows / any OS
pip install gunicorn && gunicorn -w 2 -b 0.0.0.0:8000 app:app  # Linux / macOS
```

## Desktop app

```bash
pip install -r requirements-desktop.txt
python desktop.py
```

Draw with the mouse and the prediction updates after every stroke.
Press <kbd>C</kbd> to clear and <kbd>Esc</kbd> to quit.

## Configuration

| Environment variable | Default                           | Purpose                       |
| -------------------- | --------------------------------- | ----------------------------- |
| `DIGIT_MODEL_PATH`   | `models/digit_model.keras`        | Keras model to load and save  |
| `KERAS_BACKEND`      | auto: `tensorflow` → `jax` → `torch` | Which Keras backend to use |

Keras 3 runs on TensorFlow, JAX or PyTorch. `requirements.txt` installs JAX because it works
on every OS and supported Python version. If TensorFlow is installed (`pip install tensorflow`),
it's used automatically. Models trained with one backend load fine with another.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests that compare the browser code with Python need [Node.js](https://nodejs.org/) on your
`PATH`, and they're skipped without it.

## Troubleshooting

- **"Couldn't load the model" on the website**: `web/model/` is missing. Run `python train.py`
  (or `python export_web_model.py`). On GitHub Pages, commit `web/model/` and push.
- **The Pages workflow fails at "Check the exported model is committed"**: same fix, commit `web/model/`.
- **The Pages workflow fails at "configure-pages" or "deploy"**: Pages isn't enabled yet.
  Go to Settings → Pages → Source → **GitHub Actions**, then re-run the workflow.
- **`pip install` fails on `h5py` or can't find `tensorflow`**: your Python is newer than these
  packages support. Use Python 3.10 – 3.14. On Windows, run `py install 3.13`, then
  `py -3.13 -m venv .venv`.
- **Wrong predictions**: draw one digit, reasonably large, and check the
  *What the model sees* preview. A messy or cut-off digit is hard for any model.
