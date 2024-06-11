import pytest
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from train import CustomDataset, collate_fn, preprocess_data, validate_targets, train
from src.main import LongRoPEModel


def test_custom_dataset():
    sequences = [[1, 2, 3], [4, 5, 6]]
    targets = [[2, 3, 4], [5, 6, 7]]
    dataset = CustomDataset(sequences, targets)
    assert len(dataset) == 2
    assert dataset[0] == (sequences[0], targets[0])


def test_collate_fn():
    batch = [([1, 2, 3], [2, 3, 4]), ([4, 5], [5, 6])]
    inputs, targets = collate_fn(batch)
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)


def test_preprocess_data():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = "This is a test."
    sequences = preprocess_data(data, tokenizer, max_length=10, overlap=5)
    assert len(sequences) > 0


def test_validate_targets():
    targets = [[1, 2, 3], [4, 5, 6]]
    vocab_size = 10
    assert validate_targets(targets, vocab_size) == True


def test_train():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sequences = [[1, 2, 3], [4, 5, 6]]
    targets = [[2, 3, 4], [5, 6, 7]]
    dataset = CustomDataset(sequences, targets)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    device = torch.device("cpu")
    model = LongRoPEModel(
        d_model=512, n_heads=8, num_layers=6, vocab_size=50257, max_len=65536
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs=1)