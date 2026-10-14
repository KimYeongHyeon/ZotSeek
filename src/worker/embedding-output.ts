/** Reject bad outputs before they cross into the index, disposing the WASM tensor even on failure. */
export function readEmbedding(output: any, dimensions: number): number[] {
  try {
    if (output?.dims?.length !== 2 || output.dims[0] !== 1 || output.dims[1] !== dimensions) {
      throw new Error(`Expected one ${dimensions}-dimensional embedding, got ${output?.dims}`);
    }
    const values = Array.from(output.data as Float32Array);
    if (values.length !== dimensions || !values.every(Number.isFinite) || !values.some(v => v !== 0)) {
      throw new Error('Embedding contains invalid, missing or all-zero values');
    }
    return values;
  } finally {
    if (typeof output?.dispose === 'function') output.dispose();
  }
}
