# Train a LongRoPE model on a given dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import gzip
from transformers import GPT2Tokenizer
from src.main import LongRoPEModel, RoPEPositionalEncoding


class CustomDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_data(filename):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        data = f.read()
    return data


def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(
        [torch.tensor(seq) for seq in inputs], batch_first=True, padding_value=0
    )
    padded_targets = pad_sequence(
        [torch.tensor(tgt) for tgt in targets], batch_first=True, padding_value=-1
    )  # Assuming -1 is an ignore index for your loss function
    return padded_inputs, padded_targets


def preprocess_data(data, tokenizer, max_length=1024):
    tokenized_data = tokenizer.encode(data)
    sequences = []
    targets = []

    # Segment the tokenized data into chunks of max_length
    for i in range(0, len(tokenized_data), max_length):
        end = i + max_length
        # Ensure not to exceed the length of tokenized_data
        if end < len(tokenized_data):
            sequences.append(tokenized_data[i:end])
            targets.append(tokenized_data[i + 1 : end + 1])
        else:
            sequences.append(tokenized_data[i:])
            targets.append(tokenized_data[i + 1 :] + [tokenizer.eos_token_id])

    return sequences, targets


def validate_targets(targets, vocab_size):
    for target_batch in targets:
        if any(t >= vocab_size for t in target_batch):
            raise ValueError("Target index out of vocabulary size range.")


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch

            # Add the debugging print statement here
            print(
                f"Batch input shape: {inputs.shape}, Batch target max: {max(targets.view(-1))}"
            )

            try:
                inputs = inputs.to(device)
                targets = targets.to(device)
            except ValueError:
                print("Error unpacking batch. Check the dataset loader.")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)

            # Permute the outputs to match [batch_size, num_classes, sequence_length]
            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader)}")
        model.train()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size  # Get the vocabulary size from the tokenizer

    # Load data
    # data = load_data('../data/raw/enwik8.gz')
    # Preprocess and segment data
    sequences, targets = preprocess_data(
        data, tokenizer
    )  # Ensure sequences do not exceed 1024 tokens

    # Validate targets
    validate_targets(targets, vocab_size)  # Corrected usage

    # Create dataset
    dataset = CustomDataset(sequences, targets)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Setup DataLoader with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize the model
    adjusted_vocab_size = 130110  # max(target label) + 1
    model = LongRoPEModel(
        d_model=1024,
        n_heads=16,
        num_layers=6,
        max_len=4096,
        vocab_size=adjusted_vocab_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Test model output with a single batch
    test_inputs, test_targets = next(iter(train_loader))
    test_outputs = model(test_inputs.to(device))
    test_loss = criterion(test_outputs.permute(0, 2, 1), test_targets.to(device))
    print(f"Test loss: {test_loss.item()}")

    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, device)


if __name__ == "__main__":
    main()

# def load_data(filename):
#     with gzip.open(filename, "rt") as f:
#         data = f.read()
#     return data


# # Load the dataset
# data = load_data("/data/raw/enwiki8.gz")

# # Tokenize the dataset
# tokenizer = torch.load("/data/tokenizer/enwiki8.pt")
# tokenized_data = tokenizer.encode(data)

# # Split the dataset into training and validation sets
# train_data = tokenized_data[: int(len(tokenized_data) * 0.8)]
# val_data = tokenized_data[int(len(tokenized_data) * 0.8) :]

# # Define the model architecture
# d_model = 512
# n_heads = 8
# num_layers = 6  # Number of transformer layers
# max_len = 4096  # Maximum sequence length
# base_length = 4096  # Base context window length
# target_length = 2048 * 1024  # Target context window length

# model = LongRoPEModel(d_model, n_heads, num_layers, max_len)
# model = model.extend_context(train_data, target_length)

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
