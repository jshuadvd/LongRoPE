# Train a LongRoPE model on a given dataset
# %%
from src.main import LongRoPEModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import gzip
from transformers import GPT2Tokenizer
from importlib import reload
import src.main

reload(src.main)


# %%
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


def create_sliding_window_chunks(tokenized_data, max_length=65536, overlap=4096):
    """Create sliding window chunks from tokenized data."""
    sequences = []
    start = 0
    while start < len(tokenized_data):
        end = start + max_length
        if end >= len(tokenized_data):
            # If the remaining sequence is shorter than max_length, append it as is
            sequences.append(tokenized_data[start:])
        else:
            # Split the sequence into chunks of max_length with overlap
            while start < end:
                chunk_end = min(start + max_length, end)
                sequences.append(tokenized_data[start:chunk_end])
                start += max_length - overlap
    return sequences


def validate_targets(targets, vocab_size):
    """Validate that all target indices are within the vocabulary size."""
    for target_batch in targets:
        if any(t >= vocab_size for t in target_batch):
            raise ValueError("Target index out of vocabulary size range.")
    return True


def preprocess_data(data, tokenizer, max_length, overlap):
    """
    Preprocess the input data by tokenizing it in chunks and creating sliding window sequences.

    Args:
        data (str): Input data as a string.
        tokenizer: Tokenizer object for encoding the data.
        max_length (int): Maximum sequence length for each chunk.
        overlap (int): Overlap size between consecutive chunks.

    Returns:
        list: List of preprocessed sequences.
    """
    sequences = []
    start = 0
    while start < len(data):
        end = start + max_length
        chunk = data[start:end]
        tokenized_chunk = tokenizer.encode(chunk)

        # Create sliding window sequences from the tokenized chunk
        chunk_sequences = create_sliding_window_chunks(
            tokenized_chunk, max_length=max_length, overlap=overlap
        )
        sequences.extend(chunk_sequences)

        start = end - overlap

    return sequences


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


# %%
def main():
    """Main function to setup and run training."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = (
        2048000  # Set the maximum sequence length for the tokenizer
    )
    data = load_data("../data/raw/enwik8.gz")

    max_length = 65536
    overlap = 4096
    sequences = preprocess_data(data, tokenizer, max_length, overlap)

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
        d_model=4096,
        n_heads=32,
        num_layers=6,
        vocab_size=tokenizer.vocab_size,
        max_len=2048000,  # Set max_len to 2048k tokens
    ).to(device)

    extended_model = model.extend_context(
        data_path="../data/raw/enwik8.gz",
        target_length=2048000,  # Set target_length to 2048k tokens
        max_sequence_length=65536,
        tokenizer=tokenizer,
        population_size=64,
        num_mutations=16,
        num_crossovers=16,
        max_iterations=10,
    )

    recovered_model = model.recover_short_context(
        data_path="../data/raw/enwik8.gz",
        max_sequence_length=48192,
        tokenizer=tokenizer,
    )

    optimizer = optim.Adam(recovered_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # train(recovered_model, train_loader, val_loader, optimizer, criterion, device)


if __name__ == "__main__":
    main()
