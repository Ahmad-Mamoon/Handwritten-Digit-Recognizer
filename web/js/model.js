// Run the exported CNN in plain JavaScript, with no ML library needed.
//
// The model is exported by digit_recognizer/export.py as model.json (the layer
// list) plus weights.bin (float32 weights). BatchNorm is already folded into the
// conv/dense weights, so only these layer types exist: conv2d ("same" padding,
// stride 1), maxpool2d, flatten and dense, with relu/softmax activations.
// Tensors are flat Float32Arrays in height x width x channels order, like Keras.
//
// No imports and no DOM access, so the module also runs under Node for the tests.

export const FORMAT = "digit-recognizer-web/1";

/** Fetch model.json and weights.bin from `baseUrl` (a URL ending in "/"). */
export async function loadModel(baseUrl) {
    const manifestResponse = await fetch(new URL("model.json", baseUrl));
    if (!manifestResponse.ok) throw new Error(`model.json: HTTP ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();

    const weightsResponse = await fetch(new URL(manifest.weights, baseUrl));
    if (!weightsResponse.ok) throw new Error(`${manifest.weights}: HTTP ${weightsResponse.status}`);
    return createModel(manifest, await weightsResponse.arrayBuffer());
}

/** Build a model from a parsed manifest and the weights' ArrayBuffer. */
export function createModel(manifest, buffer) {
    if (manifest.format !== FORMAT) throw new Error(`Unsupported model format: ${manifest.format}`);
    const tensor = ({ offset, length }) => new Float32Array(buffer, offset, length);
    const layers = manifest.layers.map((layer) => ({
        ...layer,
        kernel: layer.kernel && tensor(layer.kernel),
        bias: layer.bias && tensor(layer.bias),
        kernelShape: layer.kernel && layer.kernel.shape,
    }));
    const inputShape = manifest.input_shape;

    return {
        metadata: manifest.metadata || {},
        inputShape,

        /** Class probabilities (Float32Array of 10) for one image (Float32Array of 28*28). */
        predict(input) {
            let x = Float32Array.from(input);
            let [height, width, channels] = inputShape;
            for (const layer of layers) {
                if (layer.type === "conv2d") {
                    x = conv2d(x, height, width, channels, layer);
                    channels = layer.kernelShape[3];
                } else if (layer.type === "maxpool2d") {
                    x = maxPool(x, height, width, channels, layer.pool_size);
                    height = Math.floor(height / layer.pool_size[0]);
                    width = Math.floor(width / layer.pool_size[1]);
                } else if (layer.type === "flatten") {
                    [height, width, channels] = [1, 1, x.length]; // already flat in HWC order
                } else if (layer.type === "dense") {
                    x = dense(x, layer);
                    [height, width, channels] = [1, 1, x.length];
                } else {
                    throw new Error(`Unsupported layer type: ${layer.type}`);
                }
            }
            return x;
        },
    };
}

function conv2d(input, height, width, channels, { kernel, bias, kernelShape, activation }) {
    const [kh, kw, , filters] = kernelShape;
    const padTop = (kh - 1) >> 1;
    const padLeft = (kw - 1) >> 1;
    const out = new Float32Array(height * width * filters);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const o = (y * width + x) * filters;
            out.set(bias, o);
            for (let ky = 0; ky < kh; ky++) {
                const iy = y + ky - padTop;
                if (iy < 0 || iy >= height) continue;
                for (let kx = 0; kx < kw; kx++) {
                    const ix = x + kx - padLeft;
                    if (ix < 0 || ix >= width) continue;
                    const i = (iy * width + ix) * channels;
                    const k = (ky * kw + kx) * channels * filters;
                    for (let c = 0; c < channels; c++) {
                        const v = input[i + c];
                        if (v === 0) continue; // most inputs are zero (black pixels, ReLU outputs)
                        const kc = k + c * filters;
                        for (let f = 0; f < filters; f++) out[o + f] += v * kernel[kc + f];
                    }
                }
            }
        }
    }
    return activate(out, activation);
}

function maxPool(input, height, width, channels, [ph, pw]) {
    const outHeight = Math.floor(height / ph);
    const outWidth = Math.floor(width / pw);
    const out = new Float32Array(outHeight * outWidth * channels).fill(-Infinity);
    for (let y = 0; y < outHeight * ph; y++) {
        for (let x = 0; x < outWidth * pw; x++) {
            const i = (y * width + x) * channels;
            const o = ((Math.floor(y / ph) * outWidth) + Math.floor(x / pw)) * channels;
            for (let c = 0; c < channels; c++) out[o + c] = Math.max(out[o + c], input[i + c]);
        }
    }
    return out;
}

function dense(input, { kernel, bias, kernelShape, activation }) {
    const units = kernelShape[1];
    const out = Float32Array.from(bias);
    for (let i = 0; i < input.length; i++) {
        const v = input[i];
        if (v === 0) continue;
        const row = i * units;
        for (let j = 0; j < units; j++) out[j] += v * kernel[row + j];
    }
    return activate(out, activation);
}

function activate(x, activation) {
    if (activation === "relu") {
        for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
    } else if (activation === "softmax") {
        let max = -Infinity;
        for (const v of x) max = Math.max(max, v);
        let sum = 0;
        for (let i = 0; i < x.length; i++) sum += x[i] = Math.exp(x[i] - max);
        for (let i = 0; i < x.length; i++) x[i] /= sum;
    } else if (activation !== "linear") {
        throw new Error(`Unsupported activation: ${activation}`);
    }
    return x;
}
