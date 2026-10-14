import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { getModel, applyPrefix } from '../src/core/model-registry';
import { findModelLocation } from '../src/core/model-download';
import { readEmbedding } from '../src/worker/embedding-output';

test('EmbeddingGemma 2 declares only text FP16 artifacts and official retrieval prompts', () => {
  const model = getModel('embeddinggemma-2')!;
  assert.equal(model.dimensions, 768);
  assert.equal(model.textOnly, true);
  assert.equal(model.dtype, 'fp16');
  assert.equal(model.runtime, 'onnx');
  assert.ok(model.files.includes('onnx/model_fp16.onnx_data'));
  assert.equal(model.files.at(-1), model.onnxFile, 'graph must be written after its external weights');
  assert.ok(!model.files.some(f => /audio|vision|quantized/.test(f)));
  assert.equal(applyPrefix('cats', 'query', model), 'task: search result | query: cats');
  assert.equal(applyPrefix('cats', 'doc', model), 'title: none | text: cats');
});

test('a graph without its external weights is not a downloaded model', async () => {
  const model = getModel('embeddinggemma-2')!;
  const files = new Map<string, number>();
  const globals = globalThis as any;
  globals.Zotero = { Profile: { dir: '/profile' }, DataDirectory: { dir: '/data' } };
  globals.PathUtils = { join: path.join };
  globals.IOUtils = {
    exists: async (p: string) => files.has(p),
    stat: async (p: string) => ({ size: files.get(p) }),
  };
  try {
    const root = '/profile/zotseek-models/' + model.hfPath;
    files.set(path.join(root, model.onnxFile), 425780);
    assert.equal(await findModelLocation(model), null);
    files.set(path.join(root, 'onnx/model_fp16.onnx_data'), 0);
    assert.equal(await findModelLocation(model), null);
    files.set(path.join(root, 'onnx/model_fp16.onnx_data'), 542085120);
    assert.equal(await findModelLocation(model), null, 'missing tokenizer/config files must not pass');
    for (const file of ['config.json', 'tokenizer.json', 'tokenizer_config.json']) files.set(path.join(root, file), 1);
    assert.equal(await findModelLocation(model), 'profile');
  } finally {
    delete globals.Zotero;
    delete globals.PathUtils;
    delete globals.IOUtils;
  }
});

test('bad output dimensions, NaN and all-zero embeddings fail before indexing and release tensors', () => {
  for (const [dims, data] of [
    [[1, 512], new Float32Array(512).fill(1)],
    [[1, 768], new Float32Array(768).fill(NaN)],
    [[1, 768], new Float32Array(768)],
    [[1, 768], new Float32Array(10).fill(1)],
  ]) {
    let disposed = false;
    assert.throws(() => readEmbedding({ dims, data, dispose: () => { disposed = true; } }, 768));
    assert.equal(disposed, true);
  }
  assert.throws(() => readEmbedding(undefined, 768));
  assert.equal(readEmbedding({ dims: [1, 768], data: new Float32Array(768).fill(1) }, 768).length, 768);
});
