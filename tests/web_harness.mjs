// Runs the browser modules (web/js) under Node for tests/test_web_model.py.
// Reads a JSON job on stdin and prints the results as JSON.
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const jsDir = new URL("../web/js/", import.meta.url);
// Data-URL imports are always ES modules, whatever the Node version or package.json says.
const load = async (name) =>
    import(`data:text/javascript;base64,${(await readFile(new URL(name, jsDir))).toString("base64")}`);

const { createModel } = await load("model.js");
const { preprocess } = await load("preprocess.js");

let input = "";
for await (const chunk of process.stdin) input += chunk;
const job = JSON.parse(input);
const result = {};

if (job.model_dir) {
    const manifest = JSON.parse(await readFile(join(job.model_dir, "model.json"), "utf8"));
    const bytes = await readFile(join(job.model_dir, manifest.weights));
    const model = createModel(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    result.probabilities = job.inputs.map((x) => Array.from(model.predict(Float32Array.from(x))));
}

if (job.images) {
    result.preprocessed = job.images.map(({ width, height, rgba }) => {
        const digit = preprocess({ width, height, data: new Uint8ClampedArray(Buffer.from(rgba, "base64")) });
        return digit && Array.from(digit);
    });
}

process.stdout.write(JSON.stringify(result));
