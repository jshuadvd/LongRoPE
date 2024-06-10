import pytest
import torch
from transformers import GPT2Tokenizer
from src.main import (
    LongRoPEModel,
    load_data,
    non_uniform_interpolation,
    RoPEPositionalEncoding,
    progressive_extension,
    short_context_recovery,
)


def test_load_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = load_data("data/raw/enwik8.gz", tokenizer, 65536)
    assert len(data) > 0
    assert isinstance(data[0], torch.Tensor)
