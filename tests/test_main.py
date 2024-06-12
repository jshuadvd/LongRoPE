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

# Testing the load_data function
def test_load_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = load_data("data/raw/enwik8.gz", tokenizer, 65536)
    assert len(data) > 0
    assert isinstance(data[0], torch.Tensor)
    assert data[0].shape[0] <= 65536


# Testing the non_uniform_interpolation function
def test_non_uniform_interpolation():
    pos_embed = torch.randn(1, 100, 512)
    lambda_factors = torch.ones(256)
    n_hat = 50
    interpolated = non_uniform_interpolation(pos_embed, 2.0, lambda_factors, n_hat)
    assert interpolated.shape == pos_embed.shape
    assert not torch.equal(pos_embed, interpolated)


# Testing the RoPEPositionalEncoding class
def test_rope_positional_encoding():
    rope = RoPEPositionalEncoding(d_model=512, max_len=100)
    positions = torch.arange(100).unsqueeze(0)
    pos_embeddings = rope(positions)
    assert pos_embeddings.shape == (1, 100, 512)
    assert not torch.equal(positions, pos_embeddings)


# Testing the LongRoPEModel class
def test_longrope_model_initialization():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    assert model.d_model == 512
    assert model.n_heads == 8
    assert model.num_layers == 6
    assert model.vocab_size == 50257
    assert model.max_len == 65536


def test_longrope_model_embedding():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    input_ids = torch.randint(0, 50257, (2, 1024))
    embeddings = model.embedding(input_ids)
    assert embeddings.shape == (2, 1024, 512)
    assert not torch.equal(input_ids, embeddings)


def test_longrope_model_transformers():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    input_ids = torch.randint(0, 50257, (2, 1024))
    embeddings = model.embedding(input_ids)
    for transformer in model.transformers:
        embeddings = transformer(embeddings)
    assert embeddings.shape == (2, 1024, 512)


def test_longrope_model_forward():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    input_ids = torch.randint(0, 50257, (2, 1024))
    output = model(input_ids)
    assert output.shape == (2, 1024, 512)
    assert not torch.equal(input_ids, output)


def test_extend_context():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    extended_model = model.extend_context(
        data_path="data/raw/enwik8.gz",
        target_length=2048000,
        max_sequence_length=65536,
        tokenizer=tokenizer,
        population_size=64,
        num_mutations=16,
        num_crossovers=16,
        max_iterations=10,
    )
    assert extended_model is not None
    assert extended_model.max_len == 2048000


def test_recover_short_context():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    recovered_model = model.recover_short_context(
        data_path="data/raw/enwik8.gz",
        max_sequence_length=65536,
        tokenizer=tokenizer,
    )
    assert recovered_model is not None
    assert recovered_model.max_len == 65536


def test_progressive_extension():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    data = [torch.randint(0, 50257, (65536,)) for _ in range(10)]
    (
        extended_model,
        lambda_factors,
        n_hat,
        lambda_factors_base,
        n_hat_base,
    ) = progressive_extension(
        model,
        data,
        base_length=65536,
        target_length=2048000,
        population_size=64,
        num_mutations=16,
        num_crossovers=16,
        max_iterations=10,
    )
    assert extended_model is not None
    assert lambda_factors is not None
    assert n_hat is not None
    assert lambda_factors_base is not None
    assert n_hat_base is not None
    assert extended_model.max_len == 2048000


def test_short_context_recovery():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    short_context_recovery(model, tokenizer)


def test_longrope_model_forward_with_extended_context():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=2048000
    )
    input_ids = torch.randint(0, 50257, (2, 2048))
    output = model(input_ids)
    assert output.shape == (2, 2048, 512)
    assert not torch.equal(input_ids, output)


def test_longrope_model_forward_with_short_context():
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=4096
    )
    input_ids = torch.randint(0, 50257, (2, 1024))
    output = model(input_ids)
    assert output.shape == (2, 1024, 512)
    assert not torch.equal(input_ids, output)
