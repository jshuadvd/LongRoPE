import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gzip
import io


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Encoding (RoPE) module.
    """

    def __init__(self, d_model, max_len=1000000, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.theta = torch.tensor(
            [base ** (-2 * (i // 2) / d_model) for i in range(d_model)]
        )

    def forward(self, positions):
        angles = positions.unsqueeze(-1) * self.theta
        sin_cos = torch.stack([angles.cos(), angles.sin()], dim=-1)
        return sin_cos.view(*sin_cos.shape[:-2], -1)


def non_uniform_interpolation(pos_embed, extension_ratio, lambda_factors, n_hat):
    """
    Perform non-uniform interpolation on position embeddings.

    Args:
        pos_embed (torch.Tensor): Position embeddings.
        extension_ratio (float): Extension ratio for context window.
        lambda_factors (list): Lambda factors for interpolation.
        n_hat (int): Threshold for applying interpolation.

    Returns:
        torch.Tensor: Interpolated position embeddings.
    """
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
    """
    Search for optimal lambda factors using evolutionary search.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        extension_ratio (float): Extension ratio for context window.
        population_size (int): Size of the population for evolutionary search.
        num_mutations (int): Number of mutations per iteration.
        num_crossovers (int): Number of crossovers per iteration.
        max_iterations (int): Maximum number of iterations for evolutionary search.

    Returns:
        list: Optimal lambda factors found by the search.
    """
    population = initialize_population(population_size, extension_ratio)

    for i in range(max_iterations):
        perplexities = evaluate_population(model, data, population)
        parents = select_topk(population, perplexities, k=population_size // 2)
        population = mutate(parents, num_mutations) + crossover(parents, num_crossovers)

    return min(population, key=lambda x: evaluate_individual(model, data, x))


def progressive_extension(model, data, base_length, target_length):
    """
    Progressively extend the context window of the model.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        base_length (int): Base context window length.
        target_length (int): Target context window length.

    Returns:
        tuple: (Extended model, lambda factors, base lambda factors)
    """
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


def short_context_recovery(model, data, base_length, lambda_factors_base, n_hat_base):
    """
    Recover performance on shorter context lengths.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        base_length (int): Base context window length.
        lambda_factors_base (list): Base lambda factors.
        n_hat_base (int): Base n_hat.

    Returns:
        nn.Module: Recovered LongRoPE model.
    """
    short_lengths = [base_length // 2, base_length // 4]

    for length in short_lengths:
        extension_ratio = length / base_length
        lambda_factors, n_hat = search_lambda_factors(
            model, data, extension_ratio, max_length=length
        )
        model = fine_tune(model, data, length, lambda_factors, n_hat)

    model.lambda_factors_base = lambda_factors_base
    model.n_hat_base = n_hat_base

    return model


class LongRoPEModel(nn.Module):
    """
    Long Range Rotary Position Encoding (LongRoPE) model.

    This model extends the context window of transformer-based models beyond the
    typical limit by using non-uniform interpolation of rotary position embeddings.
    It enables the model to handle longer input sequences while maintaining the
    ability to capture long-range dependencies.

    Attributes:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_len (int): Maximum sequence length.
        rope (RoPEPositionalEncoding): Rotary Position Encoding (RoPE) module.
        transformers (nn.ModuleList): List of transformer encoder layers.
        lambda_factors (list): Lambda factors for non-uniform interpolation.
        lambda_factors_base (list): Lambda factors for the base model.
        extension_ratio (float): Extension ratio for the context window.
        n_hat (int): Threshold for applying interpolation.

    Methods:
        forward(input_ids):
            Perform forward pass on the input sequence.

            Args:
                input_ids (torch.Tensor): Input sequence tensor.

            Returns:
                torch.Tensor: Output embeddings from the model.

        extend_context(data_path, target_length, max_sequence_length, tokenizer):
            Extend the context window of the model.

            Args:
                data_path (str): Path to the input data file.
                target_length (int): Target context window length.
                max_sequence_length (int): Maximum sequence length for input data.
                tokenizer: Tokenizer object for encoding input data.

            Returns:
                LongRoPEModel: Extended LongRoPE model.
    """

    def __init__(self, d_model, n_heads, num_layers, vocab_size, max_len=1000000):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size  # Add vocab_size to the parameters
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer
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
        input_embeddings = self.embedding(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(
            0
        )
        pos_embeddings = self.rope(positions)

        if self.lambda_factors is not None:
            pos_embeddings = non_uniform_interpolation(
                pos_embeddings, self.extension_ratio, self.lambda_factors, self.n_hat
            )

        # Ensure pos_embeddings has the same shape as input_embeddings
        pos_embeddings = pos_embeddings[
            :, : input_embeddings.size(1), : self.d_model
        ]  # Adjust to match d_model

        print(f"input_embeddings shape: {input_embeddings.shape}")
        print(f"pos_embeddings shape: {pos_embeddings.shape}")

        embeddings = input_embeddings + pos_embeddings

        for transformer in self.transformers:
            embeddings = transformer(embeddings)

        return embeddings

    def extend_context(self, data_path, target_length, max_sequence_length, tokenizer):
        """
        Extend the context window of the model.

        Args:
            data_path (str): Path to the input data file.
            target_length (int): Target context window length.
            max_sequence_length (int): Maximum sequence length for input data.
            tokenizer: Tokenizer object for encoding input data.

        Returns:
            LongRoPEModel: Extended LongRoPE model.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for extending context.")

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
    """
    Load and preprocess the input data.

    Args:
        data_path (str): Path to the input data file.
        tokenizer: Tokenizer object for encoding input data.
        max_sequence_length (int): Maximum sequence length for input data.

    Returns:
        list: List of preprocessed input sequences.
    """
    if data_path is None or tokenizer is None:
        raise ValueError("Data path and tokenizer are required for loading data.")

    if data_path.endswith(".gz"):
        with gzip.open(data_path, "rt", encoding="utf-8") as file:
            text_data = file.read()
    else:
        with open(data_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    tokenized_data = tokenizer.encode(text_data)

    sequences = [
        tokenized_data[i : i + max_sequence_length]
        for i in range(0, len(tokenized_data), max_sequence_length)
    ]

    tensor_data = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    return tensor_data


def initialize_population(population_size, extension_ratio):
    """
    Initialize the population for evolutionary search.

    Args:
        population_size (int): Size of the population.
        extension_ratio (float): Extension ratio for context window.

    Returns:
        list: Initialized population.
    """
    population = []

    population.append(torch.ones(512) * extension_ratio)

    ntk_factors = torch.tensor([extension_ratio ** (2 * i / 512) for i in range(512)])
    population.append(ntk_factors)

    yarn_factors = torch.ones(512)
    yarn_factors[:128] = 1.0
    yarn_factors[128:256] = extension_ratio ** (1 / 3)
    yarn_factors[256:] = extension_ratio
    population.append(yarn_factors)

    for _ in range(population_size - 3):
        factors = torch.ones(512)
        for i in range(512):
            if random.random() < 0.1:
                factors[i] = random.uniform(1, extension_ratio)
        population.append(factors)

    return population


def evaluate_individual(model, data, individual):
    """
    Evaluate an individual lambda factor configuration.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        individual (list): Lambda factor configuration.

    Returns:
        float: Perplexity score for the individual.
    """
    model.lambda_factors = individual
    perplexities = []

    for seq in data:
        input_ids = seq.unsqueeze(0)
        output = model(input_ids)
        perplexity = torch.exp(torch.mean(output))
        perplexities.append(perplexity.item())

    return np.mean(perplexities)


def evaluate_population(model, data, population):
    """
    Evaluate the population of lambda factor configurations.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        population (list): Population of lambda factor configurations.

    Returns:
        list: Perplexity scores for each individual in the population.
    """
    perplexities = []
    for individual in population:
        perplexity = evaluate_individual(model, data, individual)
        perplexities.append(perplexity)
    return perplexities


def select_topk(population, perplexities, k):
    """
    Select the top-k individuals from the population based on perplexity scores.

    Args:
        population (list): Population of lambda factor configurations.
        perplexities (list): Perplexity scores for each individual in the population.
        k (int): Number of top individuals to select.

    Returns:
        list: Top-k individuals from the population.
    """
    indices = np.argsort(perplexities)[:k]
    return [population[i] for i in indices]


def mutate(parents, num_mutations):
    """
    Perform mutation on the parent population.

    Args:
        parents (list): Parent population.
        num_mutations (int): Number of mutations to perform.

    Returns:
        list: Mutated population.
    """
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
    """
    Perform crossover on the parent population.

    Args:
        parents (list): Parent population.
        num_crossovers (int): Number of crossovers to perform.

    Returns:
        list: Crossover population.
    """
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
    """
    Fine-tune the LongRoPE model.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        target_length (int): Target context window length.
        lambda_factors (list): Lambda factors for interpolation.
        n_hat (int): Threshold for applying interpolation.
        num_epochs (int, optional): Number of fine-tuning epochs. Defaults to 3.

    Returns:
        nn.Module: Fine-tuned LongRoPE model.
    """
    model.lambda_factors = lambda_factors
    model.n_hat = n_hat
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for seq in data:
            optimizer.zero_grad()

            seq_len = seq.size(0)
            if seq_len <= target_length:
                input_ids = seq.unsqueeze(0)
            else:
                start_idx = random.randint(0, seq_len - target_length)
                input_ids = seq[start_idx : start_idx + target_length].unsqueeze(0)

            output = model(input_ids)
            loss = torch.mean(output)

            loss.backward()
            optimizer.step()

    return model
