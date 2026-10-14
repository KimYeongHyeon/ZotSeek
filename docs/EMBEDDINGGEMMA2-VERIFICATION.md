# EmbeddingGemma 2 ONNX — v1.22.6.1

This fork adds **text-only EmbeddingGemma 2** as a fifth curated model. It uses
`onnx-community/embeddinggemma-2-ONNX`, FP16 ONNX, the projected/normalized
768-dimensional `sentence_embedding`, and the official retrieval prefixes.
The existing chunker and Float32/base64 database representation are unchanged.
Model ID `embeddinggemma-2` gives it a separate index from E5/Nomic.

## Why FP16

The official Q8 graph failed in the real browser WASM worker with
`Could not find an implementation for GatherBlockQuantized(1)`.
Node's native ONNX runtime supports that graph, but passing a Node smoke is not
proof it runs inside ZotSeek. No fallback to a different model is implemented.
FP16 is the tested WASM path. Download size is about **575 MB**, including the
32 MB tokenizer: graph 425,780 bytes + external weights 542,085,120 bytes.
Image and audio encoders are neither downloaded nor loaded.

## Verification (2026-10-14)

- **PASS:** 282 Node tests, including absent/empty external weights, missing
  tokenizer/config, wrong output dimensions, NaN and all-zero output rejection.
- **PASS:** typecheck reports no new errors; the upstream baseline contains 16
  known errors. Release version/compatibility manifests agree.
- **PASS:** real FP16 ONNX CPU smoke: 768 finite, normalized values; deterministic
  repeat; the same raw query differs from Nomic (maximum coordinate difference
  0.336952). Relevant/unrelated passage cosine scores: 0.834464 / 0.586828.
- **PASS:** headed Chromium runs the actual built worker and bundled ORT-Web WASM:
  init → ready → embedding, including Nomic regression smoke. Relevant/unrelated
  scores: 0.834424 / 0.586805; same raw query baseline difference 0.340874.
  The harness substitutes only the `chrome://` WASM asset base with a loopback
  HTTP base and uses relative local model URLs to emulate `resource://` loading.
  This proves browser WASM execution, not Zotero privileged-protocol behavior.
- **PASS:** independent read-only review: GO for experimental prerelease, no blocking/high-priority findings. XPI SHA-256: `ed75f612152b454fe6b67cdbd2b75815621d23075c9674a1940015baf5ab1cc8`; all 8 packaged WASM assets and worker match the tested build.
- **UNVERIFIED:** installation and visible model picker inside native Zotero,
  `resource://`/ChromeWorker behavior, real-library indexing/persistence and
  retrieval quality over the user's library. No existing installation or index
  was modified. This is an experimental prerelease.

The tiny ranking smoke is a functional check, not a retrieval benchmark.

## Reproduce

Use Node >=20.9, `npm ci`, `npm test`, `npm run typecheck`, `npm run build`, and
`npm run check:versions`. Download the model registry's required FP16 files into
`<models-dir>/onnx-community/embeddinggemma-2-ONNX/`, then run
`node scripts/smoke-embeddinggemma2.mjs <models-dir>`.
Select the model in ZotSeek and use **Update Index** to build its own index;
existing model indexes are retained. Rebuild is not needed just to switch models.
