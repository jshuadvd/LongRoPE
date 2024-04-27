# Train a LongRoPE model on a given dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import gzip
from transformers import GPT2Tokenizer
from src.main import (
    LongRoPEModel,
    RoPEPositionalEncoding,
    short_context_recovery,
    progressive_extension,
    load_data,
)


class CustomDataset(Dataset):
    """Custom dataset for handling sequences and targets."""

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_data(filename):
    """Load data from a gzip file."""
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        data = f.read()
    return data


def collate_fn(batch):
    """Custom collate function to pad data batches."""
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(
        [torch.tensor(seq) for seq in inputs], batch_first=True, padding_value=0
    )
    padded_targets = pad_sequence(
        [torch.tensor(tgt) for tgt in targets], batch_first=True, padding_value=-1
    )
    return padded_inputs, padded_targets


def create_sliding_window_chunks(tokenized_data, max_length=8192, overlap=512):
    """Create sliding window chunks from tokenized data."""
    sequences = []
    start = 0
    while start + max_length < len(tokenized_data):
        end = start + max_length
        sequences.append(tokenized_data[start:end])
        start = end - overlap
    if start < len(tokenized_data):
        sequences.append(
            tokenized_data[start : min(start + max_length, len(tokenized_data))]
        )
    return sequences


def validate_targets(targets, vocab_size):
    """Validate that all target indices are within the vocabulary size."""
    for target_batch in targets:
        if any(t >= vocab_size for t in target_batch):
            raise ValueError("Target index out of vocabulary size range.")
    print("All targets are within the vocabulary size.")


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    """Training loop for the model."""
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            print(f"Input shape: {inputs.shape}")
            print(f"Target shape: {targets.shape}")

            if inputs.size(1) > model.rope.max_len:
                print(
                    f"Warning: Batch with input size {inputs.size(1)} exceeds the maximum length of {model.rope.max_len}."
                )
                continue  # Skip this batch

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.permute(0, 2, 1), targets)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.permute(0, 2, 1), targets)
                val_loss += loss.item()
        print(
            f"Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}"
        )
        model.train()


def main():
    """Main function to setup and run training."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data = load_data("../data/raw/enwik8.gz")
    tokenized_data = tokenizer.encode(data)
    sequences = create_sliding_window_chunks(
        tokenized_data, max_length=8192, overlap=512
    )

    targets = [seq[1:] + [tokenizer.eos_token_id] for seq in sequences]

    validate_targets(targets, tokenizer.vocab_size)

    print(f"Validated: {validate_targets(targets, tokenizer.vocab_size)}")

    dataset = CustomDataset(sequences, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongRoPEModel(
        d_model=1024,
        n_heads=16,
        num_layers=6,
        vocab_size=tokenizer.vocab_size,
        max_len=8192,
    ).to(device)

    extended_model = model.extend_context(
        data_path="../data/raw/enwik8.gz",
        target_length=16384,
        max_sequence_length=8192,
        tokenizer=tokenizer,
    )

    recovered_model = extended_model.recover_short_context(
        data_path="../data/raw/enwik8.gz",
        max_sequence_length=8192,
        tokenizer=tokenizer,
    )

    optimizer = optim.Adam(recovered_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train(recovered_model, train_loader, val_loader, optimizer, criterion, device)


if __name__ == "__main__":
    main()
