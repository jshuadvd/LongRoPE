# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

## Introduction
The paper introduces LongRoPE, a method to extend the context window of large language models (LLMs) beyond 2 million tokens.

The key ideas are:

- Identify and exploit two forms of non-uniformities in positional embeddings to minimize information loss during interpolation. This enables 8x context extension without fine-tuning.

- Use an efficient progressive extension strategy with 256k fine-tuning to reach 2048k context, instead of directly fine-tuning an extremely large context.

- Adjust embeddings for shorter contexts to recover performance within original window size.

The method is applied to LLaMA2 and Mistral. Experiments across various tasks demonstrate LongRoPE's effectiveness in maintaining performance from 4k to 2048k context lengths.

Potential implementations:
- Enable in-context learning with more examples to boost LLM reasoning
- Build LLM agents that leverage longer context for tasks like dialog and question answering
- Summarize very long documents by utilizing the full document context
- Improve few-shot learning by providing more contextual examples to models

https://arxiv.org/pdf/2402.13753.pdf
