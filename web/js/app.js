import { loadModel } from "./model.js";
import { IMAGE_SIZE, preprocess } from "./preprocess.js";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const els = {
    banner: document.getElementById("modelBanner"),
    pad: document.getElementById("pad"),
    padHint: document.getElementById("padHint"),
    live: document.getElementById("liveToggle"),
    undo: document.getElementById("undoBtn"),
    clear: document.getElementById("clearBtn"),
    predict: document.getElementById("predictBtn"),
    fileInput: document.getElementById("fileInput"),
    resultCard: document.getElementById("resultCard"),
    digit: document.getElementById("resultDigit"),
    status: document.getElementById("resultStatus"),
    note: document.getElementById("resultNote"),
    modelInputFigure: document.getElementById("modelInputFigure"),
    modelInput: document.getElementById("modelInput"),
    modelInfo: document.getElementById("modelInfo"),
    probRows: [...document.querySelectorAll("#probabilities .prob")],
};

// Stroke width relative to the canvas, roughly matching MNIST pen thickness.
const BRUSH_WIDTH = canvas.width * 0.065;
const MAX_UPLOAD_SIDE = 512;
const LIVE_MIN_INTERVAL_MS = 100;
const LIVE_STORAGE_KEY = "digit-recognizer:live";

let model = null;
let strokes = []; // every stroke is an array of {x, y} points in canvas pixels
let activeStroke = null;
let activePointer = null;
let nextLivePrediction = 0;
let requestSeq = 0; // lets us drop results that arrive after a newer request or a clear

// ---------- Model ----------

const modelReady = loadModel(new URL("../model/", import.meta.url)).then(
    (loaded) => {
        model = loaded;
        const { parameters, test_accuracy: accuracy } = loaded.metadata;
        const facts = ["Convolutional neural network"];
        if (parameters) facts[0] = `CNN with ${parameters.toLocaleString("en-US")} parameters`;
        if (accuracy) facts.push(`${(accuracy * 100).toFixed(2)}% accuracy on the MNIST test set`);
        els.modelInfo.textContent = facts.join(" · ");
        return loaded;
    },
    (error) => {
        console.warn("Couldn't load the model:", error);
        els.banner.hidden = false;
        throw error;
    },
);
modelReady.catch(() => {}); // failures are reported through the banner

// ---------- Drawing ----------

function resetCanvas() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.lineWidth = BRUSH_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}

function drawDot(point) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
    ctx.fill();
}

function drawSegment(from, to) {
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
}

function redraw() {
    resetCanvas();
    for (const stroke of strokes) {
        drawDot(stroke[0]);
        for (let i = 1; i < stroke.length; i++) drawSegment(stroke[i - 1], stroke[i]);
    }
    updateControls();
}

function toCanvasPoint(event) {
    // The canvas is scaled with CSS, so map from screen pixels to canvas pixels.
    const rect = canvas.getBoundingClientRect();
    return {
        x: ((event.clientX - rect.left) * canvas.width) / rect.width,
        y: ((event.clientY - rect.top) * canvas.height) / rect.height,
    };
}

canvas.addEventListener("pointerdown", (event) => {
    if (activeStroke || event.button !== 0) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    activePointer = event.pointerId;
    nextLivePrediction = performance.now() + LIVE_MIN_INTERVAL_MS;

    const point = toCanvasPoint(event);
    activeStroke = [point];
    strokes.push(activeStroke);
    drawDot(point);
    updateControls();
});

canvas.addEventListener("pointermove", (event) => {
    if (!activeStroke || event.pointerId !== activePointer) return;
    // Coalesced events give every point the browser recorded, for smoother lines.
    const coalesced = event.getCoalescedEvents ? event.getCoalescedEvents() : [];
    for (const e of coalesced.length ? coalesced : [event]) {
        const point = toCanvasPoint(e);
        drawSegment(activeStroke[activeStroke.length - 1], point);
        activeStroke.push(point);
    }
    if (els.live.checked && model && performance.now() >= nextLivePrediction) livePredict();
});

function endStroke(event) {
    if (!activeStroke || event.pointerId !== activePointer) return;
    activeStroke = null;
    activePointer = null;
    if (els.live.checked) predictCanvas();
}

canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

function undo() {
    if (activeStroke || !strokes.length) return;
    strokes.pop();
    redraw();
    if (!strokes.length) resetResult();
    else if (els.live.checked) predictCanvas();
}

function clearAll() {
    if (activeStroke) return;
    strokes = [];
    redraw();
    resetResult();
}

function updateControls() {
    const empty = strokes.length === 0;
    els.undo.disabled = empty;
    els.predict.disabled = empty;
    els.clear.disabled = empty && els.digit.classList.contains("is-placeholder");
    els.padHint.hidden = !empty;
}

// ---------- Predictions ----------

const canvasPixels = () => ctx.getImageData(0, 0, canvas.width, canvas.height);

function predictCanvas() {
    if (strokes.length) return predict(canvasPixels);
}

// While drawing, predict as often as the device comfortably allows (~5x the time a prediction takes).
function livePredict() {
    const start = performance.now();
    runPrediction(canvasPixels());
    nextLivePrediction = performance.now() + Math.max(LIVE_MIN_INTERVAL_MS, (performance.now() - start) * 5);
}

async function predict(getPixels, source = null) {
    const seq = ++requestSeq;
    if (!model) {
        els.status.textContent = "Loading the model…";
        els.resultCard.setAttribute("aria-busy", "true");
        try {
            await modelReady;
        } catch {
            if (seq === requestSeq) showError("The model couldn't be loaded, so predictions aren't available.");
            return;
        } finally {
            els.resultCard.setAttribute("aria-busy", "false");
        }
        if (seq !== requestSeq) return;
    }
    runPrediction(getPixels(), source);
}

function runPrediction(pixels, source = null) {
    const input = preprocess(pixels);
    if (!input) {
        showError("No digit found in the image.");
        return;
    }
    showResult(model.predict(input), input, source);
}

async function predictFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        showError("That file isn't an image.");
        return;
    }
    let pixels;
    try {
        pixels = await decodeImage(file);
    } catch {
        showError("Couldn't read that image. Try a PNG, JPEG, GIF, BMP or WebP file.");
        return;
    }
    // The result will describe the uploaded image, so don't leave an unrelated drawing on screen.
    strokes = [];
    redraw();
    await predict(() => pixels, file.name || "pasted image");
}

function decodeImage(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const image = new Image();
        image.onload = () => {
            URL.revokeObjectURL(url);
            // Shrink big photos first; the digit ends up 20x20 pixels anyway.
            const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
            const work = document.createElement("canvas");
            work.width = Math.max(1, Math.round(image.naturalWidth * scale));
            work.height = Math.max(1, Math.round(image.naturalHeight * scale));
            const workCtx = work.getContext("2d", { willReadFrequently: true });
            workCtx.imageSmoothingQuality = "high";
            workCtx.drawImage(image, 0, 0, work.width, work.height);
            resolve(workCtx.getImageData(0, 0, work.width, work.height));
        };
        image.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error("Couldn't decode the image"));
        };
        image.src = url;
    });
}

const percent = (p) => `${(p * 100).toFixed(1)}%`;

function showResult(probabilities, input, source) {
    let digit = 0;
    for (let d = 1; d < probabilities.length; d++) if (probabilities[d] > probabilities[digit]) digit = d;
    const confidence = probabilities[digit];

    els.digit.textContent = digit;
    els.digit.classList.remove("is-placeholder");
    els.status.classList.remove("is-error");
    els.status.textContent = `It's ${digit === 8 ? "an" : "a"} ${digit} (${percent(confidence)} sure)`;

    const ranked = [...probabilities.keys()].sort((a, b) => probabilities[b] - probabilities[a]);
    const runnerUp = ranked[1];
    const notes = [];
    if (confidence < 0.8) notes.push(`Second guess: ${runnerUp} (${percent(probabilities[runnerUp])}). Try drawing bigger or neater.`);
    if (source) notes.push(`From ${source}`);
    els.note.textContent = notes.join(" · ");

    drawModelInput(input);
    els.modelInputFigure.hidden = false;

    els.probRows.forEach((row, d) => {
        row.querySelector(".prob-fill").style.width = `${probabilities[d] * 100}%`;
        row.querySelector(".prob-value").textContent = percent(probabilities[d]);
        row.classList.toggle("is-top", d === digit);
    });
    updateControls();
}

function drawModelInput(input) {
    const previewCtx = els.modelInput.getContext("2d");
    const image = previewCtx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    input.forEach((v, i) => {
        image.data.fill(Math.round(v * 255), 4 * i, 4 * i + 3);
        image.data[4 * i + 3] = 255;
    });
    previewCtx.putImageData(image, 0, 0);
}

function showError(message) {
    els.status.textContent = message;
    els.status.classList.add("is-error");
    els.note.textContent = "";
}

function resetResult() {
    requestSeq++; // drop any prediction still waiting for the model
    els.resultCard.setAttribute("aria-busy", "false");
    els.digit.textContent = "?";
    els.digit.classList.add("is-placeholder");
    els.status.textContent = "Draw a digit to get started.";
    els.status.classList.remove("is-error");
    els.note.textContent = "";
    els.modelInputFigure.hidden = true;
    els.probRows.forEach((row) => {
        row.querySelector(".prob-fill").style.width = "0";
        row.querySelector(".prob-value").textContent = "–";
        row.classList.remove("is-top");
    });
    updateControls();
}

// ---------- Controls ----------

els.undo.addEventListener("click", undo);
els.clear.addEventListener("click", clearAll);
els.predict.addEventListener("click", predictCanvas);

els.fileInput.addEventListener("change", () => {
    predictFile(els.fileInput.files[0]);
    els.fileInput.value = ""; // allow picking the same file again
});

els.pad.addEventListener("dragover", (event) => {
    event.preventDefault();
    els.pad.classList.add("is-dragging");
});
els.pad.addEventListener("dragleave", () => els.pad.classList.remove("is-dragging"));
els.pad.addEventListener("drop", (event) => {
    event.preventDefault();
    els.pad.classList.remove("is-dragging");
    predictFile(event.dataTransfer.files[0]);
});

document.addEventListener("paste", (event) => {
    const item = [...(event.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) predictFile(item.getAsFile());
});

document.addEventListener("keydown", (event) => {
    if (event.altKey) return;
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === "z") {
        event.preventDefault();
        undo();
    } else if (event.ctrlKey || event.metaKey) {
        return;
    } else if (key === "enter" && !(event.target instanceof HTMLButtonElement) && event.target !== els.fileInput) {
        event.preventDefault();
        predictCanvas();
    } else if (key === "escape") {
        clearAll();
    }
});

try {
    els.live.checked = localStorage.getItem(LIVE_STORAGE_KEY) !== "off";
} catch {
    // Storage can be unavailable (private mode, blocked cookies); keep the default.
}
els.live.addEventListener("change", () => {
    try {
        localStorage.setItem(LIVE_STORAGE_KEY, els.live.checked ? "on" : "off");
    } catch {
        // Ignore: remembering the toggle is only a convenience.
    }
    if (els.live.checked) predictCanvas();
});

resetCanvas();
updateControls();
