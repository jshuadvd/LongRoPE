# Train a LongRoPE model on a given dataset
# %%
from src.main import LongRoPEModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import gzip
from transformers import GPT2Tokenizer
from datasets import load_dataset, concatenate_datasets
from importlib import reload
import src.main
from accelerate import Accelerator
import wandb
import os
import logging
import hashlib
import pickle

from evaluation import evaluate_passkey_retrieval

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

reload(src.main)

# Initialize the accelerator
accelerator = Accelerator()


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
    if not batch:
        return torch.tensor([]), torch.tensor([])
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


def cached_tokenize(text, tokenizer, cache_dir="tokenizer_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{text_hash}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    tokenized = tokenizer.encode(text)

    with open(cache_file, "wb") as f:
        pickle.dump(tokenized, f)

    return tokenized


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
        # tokenized_chunk = tokenizer.encode(chunk)
        # Cache the tokenized chunk
        tokenized_chunk = cached_tokenize(chunk, tokenizer)

        # Create sliding window sequences from the tokenized chunk
        chunk_sequences = create_sliding_window_chunks(
            tokenized_chunk, max_length=max_length, overlap=overlap
        )
        sequences.extend(chunk_sequences)

        start = end - overlap

    return sequences


def compute_perplexity(loss):
    return torch.exp(loss)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    tokenizer,
    epochs=10,
    gradient_accumulation_steps=4,
    resume_from_checkpoint=None,
    max_steps=None,
):
    """
    Train the LongRoPE model.

    Args:
        model (nn.Module): The LongRoPE model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.
        scheduler (LRScheduler): Learning rate scheduler.
        tokenizer: Tokenizer for encoding/decoding text.
        epochs (int): Number of training epochs.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        resume_from_checkpoint (str): Path to a checkpoint to resume training from.
        max_steps (int): Maximum number of steps to train. If None, train for full epochs.

    Returns:
        None
    """
    # Initialize the gradient scaler for mixed precision training
    scaler = GradScaler()

    # Variables for early stopping
    best_val_loss = float("inf")
    patience = 0
    max_patience = 3
    start_epoch = 0
    global_step = 0

    # Check if resuming from a checkpoint
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = accelerator.load_state(resume_from_checkpoint)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(
            f"Resumed training from {resume_from_checkpoint} at epoch {start_epoch}, step {global_step}"
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            if max_steps and global_step >= max_steps:
                break

            # Move data to the appropriate device (CPU or GPU)
            inputs, targets = (
                inputs.to(accelerator.device),
                targets.to(accelerator.device),
            )

            # Use mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.permute(0, 2, 1), targets)
                # Normalize the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Backpropagate and accumulate gradients
            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                # Update weights and reset gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item()

            if max_steps and global_step >= max_steps:
                break

        # Calculate average training loss and perplexity
        avg_train_loss = total_loss / len(train_loader)
        train_perplexity = compute_perplexity(avg_train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = (
                    inputs.to(accelerator.device),
                    targets.to(accelerator.device),
                )
                outputs = model(inputs)
                loss = criterion(outputs.permute(0, 2, 1), targets)
                val_loss += loss.item()

        # Calculate average validation loss and perplexity
        avg_val_loss = val_loss / len(val_loader)
        val_perplexity = compute_perplexity(avg_val_loss)

        # Update learning rate
        scheduler.step()

        # Evaluate passkey retrieval at the end of each epoch and log results
        passkey_accuracies = evaluate_passkey_retrieval(model, tokenizer, model.max_len)
        for length, accuracy in passkey_accuracies.items():
            wandb.log({f"passkey_retrieval_{length}": accuracy})
            logger.info(
                f"Passkey retrieval accuracy at {length} tokens: {accuracy:.2f}"
            )

        # Log gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        wandb.log({"gradient_norm": total_norm})
        logger.info(f"Gradient norm: {total_norm:.4f}")

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": avg_train_loss,
                "train_perplexity": train_perplexity,
                "val_loss": avg_val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Log epoch results
        logger.info(
            f"Epoch {epoch+1}, Global Step {global_step}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}"
        )

        # Save checkpoint
        accelerator.save_state(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            },
            f"checkpoint_epoch_{epoch}_step_{global_step}.pt",
        )

        # Save latest checkpoint
        accelerator.save_state(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            },
            "checkpoint_latest.pt",
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            # Save best model
            accelerator.save_state(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
                "best_model.pt",
            )
        else:
            patience += 1
            if patience >= max_patience:
                logger.info("Early stopping triggered")
                break

        if max_steps and global_step >= max_steps:
            break


# %%
def main():
    """
    Main function to set up and run the LongRoPE model training process.
    """

    # Initialize Weights & Biases for experiment tracking
    wandb.init(project="longrope", entity="your-entity-name")

    # Load and configure the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 2048000  # Set maximum sequence length to 2048k tokens

    # Load the PG19 dataset
    pg19_dataset = load_dataset("pg19", split="train")

    # Define sequence lengths for progressive training
    sequence_lengths = [2048, 128000, 256000, 2048000]

    for length in sequence_lengths:
        logger.info(f"Training on sequence length: {length}")

        # Set parameters for data preprocessing
        max_length = min(length, 65536)
        overlap = 4096

        # Preprocess the data into sequences
        logger.info(f"Preprocessing PG19 dataset for length {length}...")
        sequences = []
        for item in pg19_dataset:
            text = item["text"]
            sequences.extend(preprocess_data(text, tokenizer, max_length, overlap))
        logger.info(f"Total sequences after preprocessing: {len(sequences)}")

        # Create target sequences (shifted by one token)
        targets = [seq[1:] + [tokenizer.eos_token_id] for seq in sequences]

        # Validate that all target indices are within the vocabulary size
        validate_targets(targets, tokenizer.vocab_size)

        # Create a custom dataset from sequences and targets
        dataset = CustomDataset(sequences, targets)

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders for training and validation
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

        # Initialize or extend the LongRoPE model based on the current sequence length
        if length == 2048:
            # Initialize the base LongRoPE model
            model = LongRoPEModel(
                d_model=4096,
                n_heads=32,
                num_layers=6,
                vocab_size=tokenizer.vocab_size,
                max_len=length,
            )
        else:
            # Extend the context window of the model
            model = model.extend_context(
                data=sequences,
                target_length=length,
                max_sequence_length=max_length,
                tokenizer=tokenizer,
                population_size=64,
                num_mutations=16,
                num_crossovers=16,
                max_iterations=10,
            )

        # Set up optimizer, loss function, and learning rate scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        # Prepare model, optimizer, data loaders, and scheduler for distributed training
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

        # Check for the latest checkpoint specific to this sequence length
        latest_checkpoint = f"checkpoint_latest_{length}.pt"
        if os.path.exists(latest_checkpoint):
            logger.info(f"Found checkpoint for length {length}: {latest_checkpoint}")
            resume_from_checkpoint = latest_checkpoint
        else:
            logger.info(
                f"No checkpoint found for length {length}, starting training from scratch"
            )
            resume_from_checkpoint = None

        # Perform training or fine-tuning based on the current sequence length
        if length in [128000, 256000]:
            # Fine-tuning for specific steps as mentioned in the LongRoPE paper
            fine_tune_steps = 400 if length == 128000 else 600
            train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                scheduler,
                tokenizer,
                epochs=1,
                gradient_accumulation_steps=fine_tune_steps // len(train_loader),
                resume_from_checkpoint=resume_from_checkpoint,
                max_steps=fine_tune_steps,
            )
        else:
            # Regular training for other sequence lengths
            train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                scheduler,
                tokenizer,
                resume_from_checkpoint=resume_from_checkpoint,
            )

        # Recover performance on shorter contexts after 256k extension
        if length == 256000:
            model = model.recover_short_context(
                data=sequences,
                max_sequence_length=48192,
                tokenizer=tokenizer,
            )

        # Add a simple validation step after short context recovery
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(inputs), targets).item()
                for inputs, targets in val_loader
            ) / len(val_loader)
        logger.info(f"Validation loss after short context recovery: {val_loss:.4f}")
        wandb.log({"short_context_val_loss": val_loss})

    # Save the final model
    accelerator.save_state("final_model.pt")
    wandb.save("final_model.pt")

    # Finish logging and close the Weights & Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
