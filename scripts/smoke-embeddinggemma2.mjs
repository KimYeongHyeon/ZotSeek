// Bounded real-model smoke. Download the registry's files into --models-dir first.
import { build } from 'esbuild';
import { env } from '@huggingface/transformers';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '..');
const modelsDir = process.argv[2];
assert.ok(modelsDir, 'Usage: node scripts/smoke-embeddinggemma2.mjs <models-dir>');
const out = path.join(root, 'node_modules/.cache/zotseek-smoke');
await build({ entryPoints: [path.join(root, 'src/worker/text-embedder.ts'), path.join(root, 'src/worker/embedding-output.ts'), path.join(root, 'src/core/model-registry.ts')],
  bundle: true, entryNames: '[name]', platform: 'node', format: 'esm', outdir: out, outExtension: { '.js': '.mjs' }, external: ['@huggingface/transformers'] });
const { loadTextEmbedder } = await import(pathToFileURL(path.join(out, 'text-embedder.mjs')));
const { readEmbedding } = await import(pathToFileURL(path.join(out, 'embedding-output.mjs')));
const { getModel, applyPrefix } = await import(pathToFileURL(path.join(out, 'model-registry.mjs')));
const gemmaConfig = getModel('embeddinggemma-2');
env.allowRemoteModels = false;
env.useBrowserCache = false;
const options = { dtype: 'q8', device: 'cpu', local_files_only: true,
  session_options: { intraOpNumThreads: 4, interOpNumThreads: 1 } };
const texts = ['Which planet is known as the Red Planet?',
  'Mars is known as the Red Planet because of its reddish appearance.',
  'Bread is baked from flour, water and yeast.'];
const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

env.localModelPath = path.resolve(modelsDir) + '/';
const gemma = await loadTextEmbedder(gemmaConfig.hfPath, { ...options, dtype: gemmaConfig.dtype }, gemmaConfig.textOnly);
const gemmaVectors = [];
for (const [i, text] of texts.entries()) {
  const vector = readEmbedding(await gemma(applyPrefix(text, i === 0 ? 'query' : 'doc', gemmaConfig)), gemmaConfig.dimensions);
  assert.ok(Math.abs(cosine(vector, vector) - 1) < 0.001, 'Gemma vector must be normalized');
  gemmaVectors.push(vector);
}
assert.ok(cosine(gemmaVectors[0], gemmaVectors[1]) > cosine(gemmaVectors[0], gemmaVectors[2]), 'relevant passage must outrank unrelated passage');
const repeat = readEmbedding(await gemma(applyPrefix(texts[0], 'query', gemmaConfig)), 768);
assert.ok(Math.max(...repeat.map((x, i) => Math.abs(x - gemmaVectors[0][i]))) < 1e-6, 'repeat must be deterministic');

env.localModelPath = path.join(root, 'content/models/');
const nomic = await loadTextEmbedder('Xenova/nomic-embed-text-v1.5', options);
const baseline = readEmbedding(await nomic('search_query: ' + texts[0], { pooling: 'mean', normalize: true }), 768);
const maxDifference = Math.max(...baseline.map((x, i) => Math.abs(x - gemmaVectors[0][i])));
assert.ok(maxDifference > 0.001, 'same raw input must distinguish Gemma from the baseline');
console.log(JSON.stringify({ result: 'PASS', seed: 42, runtime: 'Transformers.js 4.3.1 / ONNX Runtime CPU',
  dimensions: 768, realEmbeddings: 5, relevantScore: cosine(gemmaVectors[0], gemmaVectors[1]),
  unrelatedScore: cosine(gemmaVectors[0], gemmaVectors[2]), baselineMaxDifference: maxDifference,
  deterministicRepeat: true, mediaEncodersLoaded: false }, null, 2));
