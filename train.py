# Train a LongRoPE model on a given dataset

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gzip
import io

from longrope import LongRoPEModel, RoPEPositionalEncoding


def load_data(filename):
    with gzip.open(filename, "rt") as f:
        data = f.read()
    return data


# Load the dataset
data = load_data("/data/raw/enwiki8.gz")

# Tokenize the dataset
tokenizer = torch.load("/data/tokenizer/enwiki8.pt")
tokenized_data = tokenizer.encode(data)

# Split the dataset into training and validation sets
train_data = tokenized_data[: int(len(tokenized_data) * 0.8)]
val_data = tokenized_data[int(len(tokenized_data) * 0.8) :]

# Define the model architecture
d_model = 512
n_heads = 8
num_layers = 6  # Number of transformer layers
max_len = 4096  # Maximum sequence length
base_length = 4096  # Base context window length
target_length = 2048 * 1024  # Target context window length

model = LongRoPEModel(d_model, n_heads, num_layers, max_len)
model = model.extend_context(train_data, target_length)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
