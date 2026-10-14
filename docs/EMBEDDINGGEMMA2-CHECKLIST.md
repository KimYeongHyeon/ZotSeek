# EmbeddingGemma 2 ONNX support

- [x] Inspect upstream model format, prompts, runtime, download and index identity.
- [x] Add text-only FP16 model and external weights to the curated registry.
- [x] Load with Transformers.js 4.3.1 without downloading media encoders.
- [x] Reject incomplete downloads and invalid vectors; test missing external weights.
- [x] Run tests, typecheck and build the installable XPI.
- [x] Run deterministic real-model smoke and compare the same input with Nomic.
- [x] Inspect packaged worker/WASM identity and obtain independent review.
- [x] Save patch, XPI and verification report; fork/push if GitHub authentication permits.

Scope: local ONNX text embeddings for the existing ZotSeek text/PDF chunking flow, 768 dimensions. Preserve installed ZotSeek and user indexes. No whole-library indexing is authorized. Runtime smoke uses fixed inputs (seed 42 where applicable), bounded to a few embeddings and durable logs.
