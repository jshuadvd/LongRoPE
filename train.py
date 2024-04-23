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


def preprocess_data(data, tokenizer, max_length=8192, overlap=512):
    tokenized_data = tokenizer.encode(data)
    sequences = []
    targets = []

    step = max_length - overlap
    for i in range(0, len(tokenized_data) - overlap, step):
        end = i + max_length
        sequences.append(tokenized_data[i:end])
        targets.append(
            tokenized_data[i + 1 : end + 1]
            if end < len(tokenized_data)
            else tokenized_data[i + 1 :] + [tokenizer.eos_token_id]
        )

    return sequences, targets


def create_sliding_window_chunks(tokenized_data, max_length=8192, overlap=512):
    sequences = []
    start = 0
    while start + max_length < len(tokenized_data):
        end = start + max_length
        sequences.append(tokenized_data[start:end])
        start = end - overlap
    if start < len(tokenized_data):
        sequences.append(tokenized_data[start:])
    return sequences


def validate_targets(targets, vocab_size):
    for target_batch in targets:
        if any(t >= vocab_size for t in target_batch):
            raise ValueError("Target index out of vocabulary size range.")


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            inputs, targets = batch

            # Ensure no input batch exceeds the maximum allowed sequence length
            if inputs.shape[1] > 8192:
                raise ValueError(
                    f"Batch {i} has input length {inputs.shape[1]}, which exceeds 8192."
                )

            inputs = inputs.to(device)
            targets = targets.to(device)

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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_special_tokens=False)
    vocab_size = tokenizer.vocab_size

    # Load data
    data = load_data("../data/raw/enwik8.gz")
    tokenized_data = tokenizer.encode(data)
    print("Total length of tokenized data:", len(tokenized_data))  # Debugging line

    # Create sliding window chunks
    max_length = 8192
    overlap = 512
    sequences = create_sliding_window_chunks(tokenized_data, max_length, overlap)
    print(
        "Max sequence length after chunking:", max(len(seq) for seq in sequences)
    )  # Debugging line
    print("Number of sequences:", len(sequences))  # Debugging line

    # Ensure all sequences are within the maximum allowed length
    max_allowed_len = 8192  # or whatever your model's max_len is
    sequences = [seq for seq in sequences if len(seq) <= max_allowed_len]
    if any(len(seq) > max_allowed_len for seq in sequences):
        print("Warning: Some sequences exceed the maximum allowed length.")

    # Assuming a simple target creation where the target is the next token
    targets = [seq[1:] + [tokenizer.eos_token_id] for seq in sequences]

    # Validate targets
    validate_targets(targets, vocab_size)

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
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    # Initialize the model
    adjusted_vocab_size = vocab_size  # Adjusted to use the tokenizer's vocab size
    model = LongRoPEModel(
        d_model=1024,
        n_heads=16,
        num_layers=6,
        max_len=max_length,  # Ensure this matches the max_length used for segmentation
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
