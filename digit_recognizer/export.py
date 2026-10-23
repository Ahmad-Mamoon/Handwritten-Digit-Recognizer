"""Export the trained Keras model so it can run in the browser (web/js/model.js).

The static website has no Python server behind it (GitHub Pages only hosts
files), so the network runs in plain JavaScript. The exporter writes:

* ``model.json``: the layer list, where each layer's weights sit in the binary file, and training metadata
* ``weights.bin``: every weight as little-endian float32

BatchNormalization layers are folded into the convolution or dense layer in
front of them, which leaves the browser only five simple operations to
implement: conv2d, maxpool2d, flatten, dense and the relu/softmax activations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import MODEL_PATH, WEB_MODEL_DIR
from .model import ModelNotFoundError, metadata_path

FORMAT = "digit-recognizer-web/1"
ACTIVATIONS = {"linear", "relu", "softmax"}


def export_web_model(model_path: str | Path = MODEL_PATH, output_dir: str | Path = WEB_MODEL_DIR) -> Path:
    """Convert a saved ``.keras`` model into ``output_dir``; returns the path of model.json."""
    import keras

    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelNotFoundError(f"No trained model at {model_path}. Run `python train.py` first.")
    return export_keras_model(keras.models.load_model(model_path, compile=False), output_dir, _read_metadata(model_path))


def export_keras_model(model, output_dir: str | Path, metadata: dict | None = None) -> Path:
    layers = _convert_layers(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks, offset = [], 0
    for layer in layers:
        for name in ("kernel", "bias"):
            if name in layer:
                values = np.ascontiguousarray(layer[name], dtype="<f4")
                layer[name] = {"offset": offset, "length": int(values.size), "shape": list(values.shape)}
                chunks.append(values.tobytes())
                offset += values.nbytes
    (output_dir / "weights.bin").write_bytes(b"".join(chunks))

    manifest = {
        "format": FORMAT,
        "input_shape": [int(d) for d in model.input_shape[1:]],
        "weights": "weights.bin",
        "layers": layers,
        "metadata": metadata or {},
    }
    manifest_path = output_dir / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest_path


def _convert_layers(model) -> list[dict]:
    import keras

    converted: list[dict] = []

    def previous_linear_layer(layer) -> dict:
        if not converted or converted[-1]["type"] not in ("conv2d", "dense") or converted[-1]["activation"] != "linear":
            raise ValueError(f"{layer.name}: can only follow a Conv2D or Dense layer without an activation.")
        return converted[-1]

    for layer in model.layers:
        kind = type(layer).__name__
        if kind in ("InputLayer", "Dropout"):
            continue  # nothing to do at inference time

        if kind == "Conv2D":
            if layer.padding != "same" or tuple(layer.strides) != (1, 1) or tuple(layer.dilation_rate) != (1, 1):
                raise ValueError(f"{layer.name}: only stride-1 'same' convolutions are supported.")
            kernel, *rest = layer.get_weights()
            converted.append({
                "type": "conv2d",
                "activation": layer.activation.__name__,
                "kernel": kernel,
                "bias": rest[0] if rest else np.zeros(kernel.shape[-1], np.float32),
            })
        elif kind == "Dense":
            kernel, *rest = layer.get_weights()
            converted.append({
                "type": "dense",
                "activation": layer.activation.__name__,
                "kernel": kernel,
                "bias": rest[0] if rest else np.zeros(kernel.shape[-1], np.float32),
            })
        elif kind == "BatchNormalization":
            target = previous_linear_layer(layer)
            if layer.axis not in (-1, target["kernel"].ndim - 1):
                raise ValueError(f"{layer.name}: only BatchNormalization over the channel axis is supported.")
            to_numpy = keras.ops.convert_to_numpy
            mean, variance = to_numpy(layer.moving_mean), to_numpy(layer.moving_variance)
            gamma = to_numpy(layer.gamma) if layer.scale else np.ones_like(mean)
            beta = to_numpy(layer.beta) if layer.center else np.zeros_like(mean)
            # BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta, applied per output channel.
            scale = gamma / np.sqrt(variance + layer.epsilon)
            target["kernel"] = target["kernel"] * scale
            target["bias"] = (target["bias"] - mean) * scale + beta
        elif kind == "Activation":
            previous_linear_layer(layer)["activation"] = layer.activation.__name__
        elif kind == "MaxPooling2D":
            if tuple(layer.pool_size) != tuple(layer.strides) or layer.padding != "valid":
                raise ValueError(f"{layer.name}: only non-overlapping 'valid' max pooling is supported.")
            converted.append({"type": "maxpool2d", "pool_size": [int(s) for s in layer.pool_size]})
        elif kind == "Flatten":
            converted.append({"type": "flatten"})
        else:
            raise ValueError(f"{layer.name}: {kind} layers aren't supported by the browser runtime (web/js/model.js).")

    for layer in converted:
        if layer.get("activation", "linear") not in ACTIVATIONS:
            raise ValueError(f"Activation {layer['activation']!r} isn't supported by the browser runtime.")
    return converted


def _read_metadata(model_path: Path) -> dict:
    try:
        return json.loads(metadata_path(model_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
