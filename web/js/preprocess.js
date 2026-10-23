// Turn a drawing or photo into MNIST-style model input, in the browser.
//
// This is a line-by-line port of digit_recognizer/preprocessing.py (keep the two in
// sync; tests/test_web_model.py checks they agree). MNIST digits are light ink on
// black, scaled to fit a 20x20 box and centred by centre of mass in a 28x28 frame.
// The network only works well on images prepared the same way.
//
// No imports and no DOM access, so the module also runs under Node for the tests.

export const IMAGE_SIZE = 28;
export const DIGIT_BOX = 20;
// NumPy does this maths in float32, so Math.fround mirrors it. Without that, pixels that sit
// exactly on a threshold (MNIST's 51/255 is 0.2) land on different sides in JS and Python.
const f32 = Math.fround;
const INK_THRESHOLD = f32(0.2); // intensity (0-1) that counts as ink when cropping to the digit
const NOISE_FLOOR = f32(0.1); // fainter pixels are zeroed to drop paper texture and JPEG noise

/**
 * Convert RGBA pixels ({data, width, height}, e.g. from getImageData) into a
 * Float32Array of 28*28 values in [0, 1], or null if the image has no digit.
 */
export function preprocess(imageData) {
    return centerDigit(toGrayscale(imageData));
}

/** RGBA pixels -> {pixels, width, height} floats in [0, 1], light ink on a black background. */
export function toGrayscale({ data, width, height }) {
    const n = width * height;
    const gray = new Float32Array(n);
    const alpha = new Float32Array(n);
    let opaqueSum = 0;
    let opaqueCount = 0;
    for (let i = 0; i < n; i++) {
        const r = data[4 * i];
        const g = data[4 * i + 1];
        const b = data[4 * i + 2];
        gray[i] = ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16) / 255; // same formula as Pillow's "L"
        alpha[i] = data[4 * i + 3] / 255;
        if (alpha[i] > 0.5) {
            opaqueSum += gray[i];
            opaqueCount++;
        }
    }

    // Transparent pixels are background: fill them with whatever contrasts with the ink.
    const fill = opaqueCount > 0 && opaqueSum / opaqueCount < 0.5 ? 1 : 0;
    for (let i = 0; i < n; i++) gray[i] = f32(gray[i] * alpha[i]) + f32(f32(1 - alpha[i]) * fill);

    // The border is almost all background; if it's light, the ink is dark, so invert.
    const border = [];
    for (let x = 0; x < width; x++) border.push(gray[x], gray[(height - 1) * width + x]);
    for (let y = 0; y < height; y++) border.push(gray[y * width], gray[y * width + width - 1]);
    let background = f32(median(border));
    if (background > 0.5) {
        for (let i = 0; i < n; i++) gray[i] = 1 - gray[i];
        background = 1 - background;
    }

    // Stretch contrast so the background becomes 0 and the strongest ink 1.
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, gray[i]);
    if (peak - background < 1e-3) return { pixels: new Float32Array(n), width, height };
    const range = f32(peak - background);
    for (let i = 0; i < n; i++) {
        const v = Math.min(1, Math.max(0, f32(f32(gray[i] - f32(background)) / range)));
        gray[i] = v < NOISE_FLOOR ? 0 : v;
    }
    return { pixels: gray, width, height };
}

/** Crop the ink, fit it into a 20x20 box and centre it by mass in a 28x28 frame. */
export function centerDigit({ pixels, width, height }) {
    let top = height;
    let bottom = -1;
    let left = width;
    let right = -1;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (pixels[y * width + x] > INK_THRESHOLD) {
                top = Math.min(top, y);
                bottom = Math.max(bottom, y);
                left = Math.min(left, x);
                right = Math.max(right, x);
            }
        }
    }
    if (bottom < 0) return null;

    const cropWidth = right - left + 1;
    const cropHeight = bottom - top + 1;
    const crop = new Float32Array(cropWidth * cropHeight);
    for (let y = 0; y < cropHeight; y++) {
        for (let x = 0; x < cropWidth; x++) {
            crop[y * cropWidth + x] = roundHalfEven(f32(pixels[(top + y) * width + left + x] * 255));
        }
    }

    // Scale the longest side to DIGIT_BOX pixels, keeping the aspect ratio (a "1" stays thin).
    const scale = DIGIT_BOX / Math.max(cropWidth, cropHeight);
    const newWidth = Math.max(1, roundHalfEven(cropWidth * scale));
    const newHeight = Math.max(1, roundHalfEven(cropHeight * scale));
    const digit = resizeLanczos(crop, cropWidth, cropHeight, newWidth, newHeight);

    const frame = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
    const offsetY = Math.floor((IMAGE_SIZE - newHeight) / 2);
    const offsetX = Math.floor((IMAGE_SIZE - newWidth) / 2);
    let total = 0;
    let sumY = 0;
    let sumX = 0;
    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const v = f32(digit[y * newWidth + x] / 255);
            frame[(offsetY + y) * IMAGE_SIZE + offsetX + x] = v;
            total += v;
            sumY += (offsetY + y) * v;
            sumX += (offsetX + x) * v;
        }
    }
    if (total <= 0) return null;

    // Shift so the centre of mass sits in the middle of the frame, as in MNIST.
    const centre = (IMAGE_SIZE - 1) / 2;
    return shift(frame, roundHalfEven(centre - sumY / total), roundHalfEven(centre - sumX / total));
}

function shift(image, dy, dx) {
    const shifted = new Float32Array(image.length);
    for (let y = Math.max(dy, 0); y < IMAGE_SIZE + Math.min(dy, 0); y++) {
        for (let x = Math.max(dx, 0); x < IMAGE_SIZE + Math.min(dx, 0); x++) {
            shifted[y * IMAGE_SIZE + x] = image[(y - dy) * IMAGE_SIZE + x - dx];
        }
    }
    return shifted;
}

// Separable Lanczos-3 resampling of 0-255 values, matching Pillow's Image.resize(LANCZOS).
function resizeLanczos(src, width, height, newWidth, newHeight) {
    const horizontal = resample(src, width, height, newWidth, true);
    return resample(horizontal, newWidth, height, newHeight, false);
}

function resample(src, width, height, outSize, horizontal) {
    const inSize = horizontal ? width : height;
    const lines = horizontal ? height : width;
    const out = new Float32Array(horizontal ? outSize * height : width * outSize);
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1);
    const support = 3 * filterScale;

    for (let o = 0; o < outSize; o++) {
        const centre = (o + 0.5) * scale;
        const start = Math.max(0, Math.trunc(centre - support + 0.5));
        const end = Math.min(inSize, Math.trunc(centre + support + 0.5));
        const weights = [];
        let weightSum = 0;
        for (let i = start; i < end; i++) {
            const w = lanczos((i - centre + 0.5) / filterScale);
            weights.push(w);
            weightSum += w;
        }
        for (let line = 0; line < lines; line++) {
            let value = 0;
            for (let i = start; i < end; i++) {
                const index = horizontal ? line * width + i : i * width + line;
                value += src[index] * weights[i - start];
            }
            value = Math.min(255, Math.max(0, Math.round(weightSum ? value / weightSum : 0)));
            out[horizontal ? line * outSize + o : o * width + line] = value;
        }
    }
    return out;
}

function lanczos(x) {
    if (x <= -3 || x >= 3) return 0;
    return sinc(x) * sinc(x / 3);
}

function sinc(x) {
    if (x === 0) return 1;
    const px = Math.PI * x;
    return Math.sin(px) / px;
}

function median(values) {
    const sorted = Float64Array.from(values).sort();
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Python's round(): halves go to the nearest even number.
function roundHalfEven(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
