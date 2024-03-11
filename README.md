# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

## Introduction
The paper introduces LongRoPE, a method to extend the context window of large language models (LLMs) beyond 2 million tokens.

The key ideas are:

- Identify and exploit two forms of non-uniformities in positional embeddings to minimize information loss during interpolation. This enables 8x context extension without fine-tuning.

- Use an efficient progressive extension strategy with 256k fine-tuning to reach 2048k context, instead of directly fine-tuning an extremely large context.

- Adjust embeddings for shorter contexts to recover performance within original window size.

The method is applied to LLaMA2 and Mistral. Experiments across various tasks demonstrate LongRoPE's effectiveness in maintaining performance from 4k to 2048k context lengths.

### Potential implementations:
- Enable in-context learning with more examples to boost LLM reasoning
- Build LLM agents that leverage longer context for tasks like dialog and question answering
- Summarize very long documents by utilizing the full document context
- Improve few-shot learning by providing more contextual examples to models

## Installation
Instructions for setting up the LongRoPE environment, including required libraries and frameworks.

## Usage
Comprehensive examples demonstrating how to leverage LongRoPE for various applications, from text analysis to generating extensive documents.

```python
# Example usage
data_path = "path/to/your/dataset"
d_model = 512
n_heads = 8
num_layers = 6
base_length = 4096
target_length = 2048 * 1024

data = load_data(data_path)
model = LongRoPEModel(d_model, n_heads, num_layers, base_length)
model = model.extend_context(data, target_length)

input_ids = torch.randn(2, target_length, d_model)
output = model(input_ids)
print(output.shape)  # Expected shape: (batch_size, target_length, d_model)
```

## Model Architecture
An in-depth look at the structural modifications and their implications for model performance. This section would ideally cover the technical design and the rationale behind each decision.

## Implementation Highlights
Insights into the coding and operational specifics that enable LongRoPE's functionality. This may include snippets or pseudocode illustrating key components.

For more detailed information, please refer to the [paper](https://arxiv.org/pdf/2402.13753.pdf).

## Citation

```bibtex
@article{ding2024longrope,
  title={LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens},
  author={Ding, Yiran and Zhang, Li Lyna and Zhang, Chengruidong and Xu, Yuanyuan and Shang, Ning and Xu, Jiahang and Yang, Fan and Yang, Mao},
  journal={arXiv preprint arXiv:2402.13753},
  year={2024}
}
```
