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


def test_non_uniform_interpolation():
    pos_embed = torch.randn(1, 100, 512)
    lambda_factors = torch.ones(256)
    n_hat = 50
    interpolated = non_uniform_interpolation(pos_embed, 2.0, lambda_factors, n_hat)
    assert interpolated.shape == pos_embed.shape
