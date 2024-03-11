import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gzip
import io


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.theta = torch.tensor(
            [base ** (-2 * (i // 2) / d_model) for i in range(d_model)]
        )

    def forward(self, positions):
        angles = positions.unsqueeze(-1) * self.theta
        return torch.stack([angles.cos(), angles.sin()], dim=-1).flatten(-2)


def non_uniform_interpolation(pos_embed, extension_ratio, lambda_factors, n_hat):
    d_model = pos_embed.shape[-1]
    interpolated_pos = pos_embed.clone()

    for i in range(d_model // 2):
        mask = torch.arange(pos_embed.shape[-2]) < n_hat
        scale = torch.where(
            mask, torch.ones_like(pos_embed[..., 0]), 1 / lambda_factors[i]
        )
        interpolated_pos[..., i * 2] *= scale
        interpolated_pos[..., i * 2 + 1] *= scale

    return interpolated_pos


def search_lambda_factors(
    model,
    data,
    extension_ratio,
    population_size,
    num_mutations,
    num_crossovers,
    max_iterations,
):
    population = initialize_population(population_size, extension_ratio)

    for i in range(max_iterations):
        perplexities = evaluate_population(model, data, population)
        parents = select_topk(population, perplexities, k=population_size // 2)
        population = mutate(parents, num_mutations) + crossover(parents, num_crossovers)

    return min(population, key=lambda x: evaluate_individual(model, data, x))


def progressive_extension(model, data, base_length, target_length):
    curr_model = model
    curr_length = base_length

    while curr_length < target_length:
        lambda_factors, n_hat = search_lambda_factors(
            curr_model, data, curr_length / base_length
        )
        curr_model = fine_tune(curr_model, data, curr_length, lambda_factors, n_hat)
        curr_length *= 2

    lambda_factors_base, _ = search_lambda_factors(
        curr_model, data, curr_length / base_length, max_length=base_length
    )

    return curr_model, lambda_factors, lambda_factors_base


class LongRoPEModel(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.rope = RoPEPositionalEncoding(d_model, max_len)
        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
                for _ in range(num_layers)
            ]
        )
        self.lambda_factors = None
        self.lambda_factors_base = None

    def forward(self, input_ids):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        pos_embeddings = self.rope(positions)

        if self.lambda_factors is not None:
            pos_embeddings = non_uniform_interpolation(
                pos_embeddings, self.extension_ratio, self.lambda_factors, self.n_hat
            )

        input_embeddings = input_ids + pos_embeddings

        for transformer in self.transformers:
            input_embeddings = transformer(input_embeddings)

        return input_embeddings

    def extend_context(self, data_path, target_length, max_sequence_length, tokenizer):
        self.extension_ratio = target_length / self.rope.max_len

        data = load_data(data_path, tokenizer, max_sequence_length)
        model, lambda_factors, lambda_factors_base = progressive_extension(
            self, data, self.rope.max_len, target_length
        )

        self.lambda_factors = lambda_factors
        self.lambda_factors_base = lambda_factors_base
        self.n_hat = self.rope.max_len // 2

        return model


def load_data(data_path, tokenizer, max_sequence_length):
    # Load and preprocess the dataset using the specified tokenizer
    if data_path.endswith(".gz"):
        with gzip.open(data_path, "rt", encoding="utf-8") as file:
            text_data = file.read()
    else:
        with open(data_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    tokenized_data = tokenizer.encode(text_data)

    # Split the tokenized data into sequences of max_sequence_length
    sequences = [
        tokenized_data[i : i + max_sequence_length]
        for i in range(0, len(tokenized_data), max_sequence_length)
    ]

    # Convert sequences to tensors
    tensor_data = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    return tensor_data


def initialize_population(population_size, extension_ratio):
    population = []

    # Add PI factors
    population.append(torch.ones(512) * extension_ratio)

    # Add NTK factors
    ntk_factors = torch.tensor([extension_ratio ** (2 * i / 512) for i in range(512)])
    population.append(ntk_factors)

    # Add YaRN factors
    yarn_factors = torch.ones(512)
    yarn_factors[:128] = 1.0
    yarn_factors[128:256] = extension_ratio ** (1 / 3)
    yarn_factors[256:] = extension_ratio
    population.append(yarn_factors)

    # Generate random mutations
    for _ in range(population_size - 3):
        factors = torch.ones(512)
        for i in range(512):
            if random.random() < 0.1:
                factors[i] = random.uniform(1, extension_ratio)
        population.append(factors)

    return population


def evaluate_individual(model, data, individual):
    model.lambda_factors = individual
    perplexities = []

    for seq in data:
        input_ids = seq.unsqueeze(0)
        output = model(input_ids)
        # Calculate perplexity based on the model output
        # Implement your perplexity calculation logic here
        perplexity = torch.exp(torch.mean(output))
        perplexities.append(perplexity.item())

    return np.mean(perplexities)


def evaluate_population(model, data, population):
    perplexities = []
    for individual in population:
        perplexity = evaluate_individual(model, data, individual)
        perplexities.append(perplexity)
    return perplexities


def select_topk(population, perplexities, k):
    indices = np.argsort(perplexities)[:k]
    return [population[i] for i in indices]


def mutate(parents, num_mutations):
    mutated_population = []
    for _ in range(num_mutations):
        parent = random.choice(parents)
        child = parent.clone()
        for i in range(512):
            if random.random() < 0.1:
                child[i] *= random.uniform(0.8, 1.2)
        mutated_population.append(child)
    return mutated_population


def crossover(parents, num_crossovers):
    crossover_population = []
    for _ in range(num_crossovers):
        parent1, parent2 = random.sample(parents, 2)
        child = parent1.clone()
        for i in range(512):
            if random.random() < 0.5:
                child[i] = parent2[i]
        crossover_population.append(child)
    return crossover_population


def fine_tune(model, data, target_length, lambda_factors, n_hat, num_epochs=3):
    model.lambda_factors = lambda_factors
    model.n_hat = n_hat
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for seq in data:
            optimizer.zero_grad()

            # Create mixed-length sequences for fine-tuning
            seq_len = seq.size(0)
            if seq_len <= target_length:
                input_ids = seq.unsqueeze(0)
            else:
                start_idx = random.randint(0, seq_len - target_length)
                input_ids = seq[start_idx : start_idx + target_length].unsqueeze(0)

            output = model(input_ids)
            # Calculate loss based on the model output
            # Implement your loss calculation logic here
            loss = torch.mean(output)

            loss.backward()
            optimizer.step()

    return model


# Example usage
data_path = "path/to/your/dataset"
d_model = 512
n_heads = 8
num_layers = 6
base_length = 4096
target_length = 2048 * 1024

data = load_data(data_path)
model = LongRoPEModel(d_model, n_heads, num_layers, base_length)
model = model.extend_context(data, target_length)

input_ids = torch.randn(2, target_length, d_model)
output = model(input_ids)
print(output.shape)  # Expected shape: (batch_size, target_length, d_model)
