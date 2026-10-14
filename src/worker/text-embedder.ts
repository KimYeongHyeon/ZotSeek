import { AutoConfig, AutoModel, AutoTokenizer, pipeline } from '@huggingface/transformers';

/** Keep multimodal encoders out of ZotSeek's text-only indexing flow. */
export async function loadTextEmbedder(hfPath: string, options: any, textOnly = false): Promise<any> {
  if (!textOnly) return pipeline('feature-extraction', hfPath, options);

  const config: any = await AutoConfig.from_pretrained(hfPath, options);
  config.vision_config = null;
  config.audio_config = null;
  const tokenizer = await AutoTokenizer.from_pretrained(hfPath, options);
  const model = await AutoModel.from_pretrained(hfPath, { ...options, config });

  return async (text: string) => {
    const inputs = tokenizer(text, { padding: true, truncation: true, max_length: 8192 });
    const outputs = await model(inputs);
    // The exported graph includes the trained 512→768 projection and normalization.
    // Pooling last_hidden_state again would bypass that projection.
    const embedding = outputs.sentence_embedding;
    for (const value of Object.values(outputs)) {
      if (value !== embedding && typeof (value as any)?.dispose === 'function') (value as any).dispose();
    }
    if (!embedding) throw new Error('Text model returned no sentence_embedding');
    return embedding;
  };
}
