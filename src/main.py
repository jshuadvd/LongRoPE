import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gzip


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
    Perform non-uniform interpolation on position embeddings as described in the LongRoPE paper.

    This function implements the two forms of non-uniformities:
    1. Varying RoPE dimensions (lambda_factors)
    2. Token positions (n_hat)

    Args:
        pos_embed (torch.Tensor): Original position embeddings.
        extension_ratio (float): Ratio of target length to original length.
        lambda_factors (list): Lambda factors for each RoPE dimension.
        n_hat (int): Number of initial tokens to keep without interpolation.

    Returns:
        torch.Tensor: Interpolated position embeddings.
    """
    d_model = pos_embed.shape[-1]
    interpolated_pos = pos_embed.clone()

    for i in range(d_model // 2):
        # Apply different scaling based on token position
        mask = torch.arange(pos_embed.shape[-2], device=pos_embed.device) < n_hat
        scale = torch.where(
            mask,
            torch.ones_like(pos_embed[..., 0], device=pos_embed.device),
            1
            / (
                lambda_factors[i] * extension_ratio
            ),  # Use dimension-specific lambda factors
        )
        # Apply scaling to both sine and cosine components
        interpolated_pos[..., 2 * i] *= scale
        interpolated_pos[..., 2 * i + 1] *= scale

    return interpolated_pos


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

    def __init__(self, d_model, n_heads, num_layers, vocab_size, max_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RoPEPositionalEncoding(d_model, max_len)
        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
                for _ in range(num_layers)
            ]
        )

        # Dictionary-based attributes for different context lengths
        self.lambda_factors = {
            "4k": None,
            "8k": None,
            "128k": None,
            "256k": None,
            "2048k": None,
        }

        self.n_hat = {"4k": None, "8k": None, "128k": None, "256k": None, "2048k": None}

        # Base attributes
        self.lambda_factors_base = None
        self.n_hat_base = 0

        # Extension ratio
        self.extension_ratio = None

    def forward(self, input_ids):
        input_embeddings = self.embedding(input_ids)
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.rope(positions)

        # Apply non-uniform interpolation based on sequence length
        if seq_length <= 4096:
            pos_embeddings = self.apply_interpolation(pos_embeddings, "4k")
        elif seq_length <= 8192:
            pos_embeddings = self.apply_interpolation(pos_embeddings, "8k")
        elif seq_length <= 131072:
            pos_embeddings = self.apply_interpolation(pos_embeddings, "128k")
        elif seq_length <= 262144:
            pos_embeddings = self.apply_interpolation(pos_embeddings, "256k")
        else:
            pos_embeddings = self.apply_interpolation(pos_embeddings, "2048k")

        if seq_length > self.rope.max_len:
            # Truncate the position embeddings if the sequence length exceeds the maximum length
            pos_embeddings = pos_embeddings[:, : self.rope.max_len, :]
            input_embeddings = input_embeddings[:, : self.rope.max_len, :]
            seq_length = self.rope.max_len

        # Ensure that pos_embeddings has the same shape as input_embeddings
        pos_embeddings = pos_embeddings[:, :seq_length, : self.d_model]

        embeddings = input_embeddings + pos_embeddings

        for transformer in self.transformers:
            embeddings = transformer(embeddings)

        return embeddings

    def apply_interpolation(self, pos_embed, context_length):
        """Apply non-uniform interpolation to position embeddings."""
        return non_uniform_interpolation(
            pos_embed,
            self.extension_ratio,
            self.lambda_factors[context_length],
            self.n_hat[context_length],
        )

    def extend_context(
        self,
        data_path,
        target_length,
        max_sequence_length,
        tokenizer,
        population_size,
        num_mutations,
        num_crossovers,
        max_iterations,
    ):
        """
        Extend the context window of the model.

        Args:
            data_path (str): Path to the input data file.
            target_length (int): Target context window length.
            max_sequence_length (int): Maximum sequence length for input data.
            tokenizer: Tokenizer object for encoding input data.
            population_size (int): Size of the population for evolutionary search.
            num_mutations (int): Number of mutations per iteration.
            num_crossovers (int): Number of crossovers per iteration.
            max_iterations (int): Maximum number of iterations for evolutionary search.

        Returns:
            LongRoPEModel: Extended LongRoPE model.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for extending context.")

        self.extension_ratio = target_length / self.rope.max_len

        data = load_data(data_path, tokenizer, max_sequence_length)

        (
            model,
            lambda_factors,
            n_hat,
            lambda_factors_base,
            n_hat_base,
        ) = progressive_extension(
            self,
            data,
            self.rope.max_len,
            target_length,
            population_size,
            num_mutations,
            num_crossovers,
            max_iterations,
        )

        self.lambda_factors = lambda_factors
        self.lambda_factors_base = lambda_factors_base
        self.n_hat = n_hat
        self.n_hat_base = n_hat_base

        return model

    def recover_short_context(self, data_path, max_sequence_length, tokenizer):
        """
        Recover performance on shorter context lengths.

        Args:
            data_path (str): Path to the input data file.
            max_sequence_length (int): Maximum sequence length for input data.
            tokenizer: Tokenizer object for encoding input data.

        Returns:
            LongRoPEModel: Recovered LongRoPE model.
        """

        if tokenizer is None:
            raise ValueError("Tokenizer is required for recovering short context.")

        data = load_data(data_path, tokenizer, max_sequence_length)
        model = short_context_recovery(
            self, data, self.rope.max_len, self.lambda_factors_base, self.n_hat_base
        )

        return model


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
    Search for optimal lambda factors using evolutionary search as described in the LongRoPE paper.

    This function implements the efficient search algorithm, including the monotonic constraint
    and optimized initial population generation.

    Args:
        model: LongRoPE model to be extended.
        data: List of input sequences for evaluation.
        extension_ratio: Ratio of target length to current length.
        population_size: Size of the population for evolutionary search.
        num_mutations: Number of mutations per iteration.
        num_crossovers: Number of crossovers per iteration.
        max_iterations: Maximum number of iterations for evolutionary search.

    Returns:
        tuple: (Best lambda factors, best n_hat)
    """
    # Define search space as described in Section 3.2 of the paper
    search_space = {
        "lambda_i": (
            1.0,
            extension_ratio * 1.25,
            0.01,
        ),  # Min, max, and step size for lambda_i
        "n_hat": [
            0,
            1,
            2,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            64,
            128,
            256,
        ],  # Possible n_hat values
    }

    # Initialize population with optimized method (including PI, NTK, and YaRN as individuals)
    population = initialize_population(population_size, search_space, model.d_model)

    for _ in range(max_iterations):
        # Evaluate the fitness (perplexity) of each individual in the population
        perplexities = evaluate_population(model, data, population)
        # Select the top-performing individuals as parents
        parents = select_topk(population, perplexities, k=population_size // 2)
        # Create new population through mutation and crossover
        population = mutate(parents, num_mutations, model.d_model) + crossover(
            parents, num_crossovers, model.d_model
        )
        # Apply monotonic constraint to ensure λi ≤ λi+1
        population = [
            apply_monotonic_constraint(individual) for individual in population
        ]

    # Select the best individual based on the lowest perplexity
    best_individual = min(population, key=lambda x: evaluate_individual(model, data, x))
    return best_individual["lambda_i"], best_individual["n_hat"]


def apply_monotonic_constraint(individual):
    """
    Apply the monotonic constraint to lambda factors as described in the paper.

    This ensures that λi ≤ λi+1, which is theoretically justified and improves performance.

    Args:
        individual: Dictionary containing 'lambda_i' and 'n_hat'

    Returns:
        individual: Dictionary with monotonically non-decreasing lambda factors
    """
    lambda_i = individual["lambda_i"]
    for i in range(1, len(lambda_i)):
        lambda_i[i] = max(lambda_i[i], lambda_i[i - 1])
    return individual


def initialize_population(population_size, search_space, d_model):
    """
    Initialize the population for evolutionary search.

    This function implements the optimized initial population generation described in Section 3.2,
    including PI, NTK, and YaRN as initial individuals.

    Args:
        population_size: Number of individuals in the population
        search_space: Dictionary defining the search space for lambda_i and n_hat
        d_model: Dimension of the model

    Returns:
        population: List of individuals, each represented as a dictionary
    """
    population = []
    for _ in range(population_size):
        individual = {
            "lambda_i": [
                random.uniform(*search_space["lambda_i"]) for _ in range(d_model // 2)
            ],
            "n_hat": random.choice(search_space["n_hat"]),
        }
        population.append(apply_monotonic_constraint(individual))
    return population


def evaluate_individual(model, data, individual):
    """
    Evaluate an individual lambda factor configuration.

    Args:
        model (nn.Module): LongRoPE model.
        data (list): List of input sequences.
        individual (tuple): Lambda factor configuration and n_hat.

    Returns:
        float: Perplexity score for the individual.
    """
    lambda_factors, n_hat = individual
    model.lambda_factors = lambda_factors
    model.n_hat = n_hat

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


def mutate(parents, num_mutations, d_model):
    """
    Perform mutation on the parent population.

    Args:
        parents (list): Parent population.
        num_mutations (int): Number of mutations to perform.
        d_model (int): Dimension of the model.

    Returns:
        list: Mutated population.
    """
    mutated_population = []
    for _ in range(num_mutations):
        parent_lambda, parent_n_hat = random.choice(parents)
        child_lambda = parent_lambda.clone()
        child_n_hat = parent_n_hat

        for i in range(d_model):
            if random.random() < 0.1:
                child_lambda[i] *= random.uniform(0.8, 1.2)

        if random.random() < 0.1:
            child_n_hat = random.randint(0, d_model)

        mutated_population.append((child_lambda, child_n_hat))

    return mutated_population


def crossover(parents, num_crossovers, d_model):
    """
    Perform crossover on the parent population.

    Args:
        parents (list): Parent population.
        num_crossovers (int): Number of crossovers to perform.
        d_model (int): Dimension of the model.

    Returns:
        list: Crossover population.
    """
    crossover_population = []
    for _ in range(num_crossovers):
        parent1_lambda, parent1_n_hat = random.choice(parents)
        parent2_lambda, parent2_n_hat = random.choice(parents)
        child_lambda = parent1_lambda.clone()
        child_n_hat = parent1_n_hat

        for i in range(d_model):
            if random.random() < 0.5:
                child_lambda[i] = parent2_lambda[i]

        if random.random() < 0.5:
            child_n_hat = parent2_n_hat

        crossover_population.append((child_lambda, child_n_hat))

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


def progressive_extension(
    model,
    data,
    base_length,
    target_length,
    population_size,
    num_mutations,
    num_crossovers,
    max_iterations,
):
    """
    Progressively extend the context window of the model using the two-stage approach described in the LongRoPE paper.

    Args:
        model (nn.Module): LongRoPE model to be extended.
        data (list): List of input sequences for fine-tuning and evaluation.
        base_length (int): Original context window length of the model.
        target_length (int): Target context window length (up to 2048k in the paper).
        population_size (int): Size of the population for evolutionary search.
        num_mutations (int): Number of mutations per iteration in the search.
        num_crossovers (int): Number of crossovers per iteration in the search.
        max_iterations (int): Maximum number of iterations for evolutionary search.

    Returns:
        tuple: (Extended model, final lambda factors, final n_hat, 256k lambda factors, 256k n_hat)
    """
    # Stage 1: Extend to 256k
    curr_model = model

    # First extend to 128k
    lambda_factors_128k, n_hat_128k = search_lambda_factors(
        curr_model,
        data,
        128000 / base_length,
        population_size,
        num_mutations,
        num_crossovers,
        max_iterations,
    )

    # Fine-tune for 400 steps as specified in the paper
    curr_model = fine_tune(
        curr_model, data, 128000, lambda_factors_128k, n_hat_128k, steps=400
    )

    # Update model attributes
    curr_model.lambda_factors["128k"] = lambda_factors_128k
    curr_model.n_hat["128k"] = n_hat_128k

    # Then extend to 256k
    lambda_factors_256k, n_hat_256k = search_lambda_factors(
        curr_model,
        data,
        256000 / 128000,
        population_size,
        num_mutations,
        num_crossovers,
        max_iterations,
    )

    # Fine-tune for 600 steps as specified in the paper
    curr_model = fine_tune(
        curr_model, data, 256000, lambda_factors_256k, n_hat_256k, steps=600
    )

    # Update model attributes
    curr_model.lambda_factors["256k"] = lambda_factors_256k
    curr_model.n_hat["256k"] = n_hat_256k

    # Stage 2: Extend to target length without further fine-tuning
    if target_length > 256000:
        final_lambda_factors, final_n_hat = search_lambda_factors(
            curr_model,
            data,
            target_length / 256000,
            population_size
            // 2,  # Reduce population size for efficiency in final search
            num_mutations // 2,
            num_crossovers // 2,
            max_iterations // 2,
        )
        # Update model attributes
        curr_model.lambda_factors["2048k"] = final_lambda_factors
        curr_model.n_hat["2048k"] = final_n_hat
    else:
        final_lambda_factors, final_n_hat = lambda_factors_256k, n_hat_256k

    return (
        curr_model,
        final_lambda_factors,
        final_n_hat,
        lambda_factors_256k,
        n_hat_256k,
    )


def short_context_recovery(model, data, base_length, lambda_factors_base, n_hat_base):
    """
    This function ensures that the model maintains good performance on shorter contexts (4k and 8k)
    even after being extended to very long contexts.

    Args:
        model (nn.Module): Extended LongRoPE model.
        data (list): List of input sequences for fine-tuning and evaluation.
        base_length (int): Original context window length of the model.
        lambda_factors_base (list): Base lambda factors for the extended model.
        n_hat_base (int): Base n_hat for the extended model.

    Returns:
        nn.Module: LongRoPE model with recovered short context performance.
    """
    short_lengths = [4096, 8192]  # Specific lengths mentioned in the paper

    for length in short_lengths:
        extension_ratio = length / base_length
        lambda_factors, n_hat = search_lambda_factors(
            model,
            data,
            extension_ratio,
            population_size=64,
            num_mutations=16,
            num_crossovers=16,
            max_iterations=40,
        )
        # Fine-tune for short context recovery
        model = fine_tune(model, data, length, lambda_factors, n_hat, steps=100)

    # Store base factors for use during inference
    model.lambda_factors_base = lambda_factors_base
    model.n_hat_base = n_hat_base

    return model
