import random
import torch


def generate_passkey_prompt(passkey, context_length):
    """
    Generate a prompt with a hidden passkey for the retrieval task.
    """
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    filler_tokens = len(filler.split())

    passkey_position = random.randint(filler_tokens, context_length - filler_tokens)

    pre_passkey = filler * (passkey_position // filler_tokens)
    post_passkey = filler * (
        (context_length - passkey_position - len(passkey) - filler_tokens)
        // filler_tokens
    )

    prompt = (
        f"{pre_passkey}The pass key is {passkey}. Remember it. {passkey} is the pass key. "
        f"{post_passkey}What is the pass key? The pass key is"
    )

    return prompt


def passkey_retrieval_test(model, tokenizer, max_length, num_trials=10):
    """
    Perform the passkey retrieval test on the model.

    Args:
        model: The LongRoPE model to evaluate.
        tokenizer: Tokenizer for encoding/decoding text.
        max_length: Maximum sequence length to test.
        num_trials: Number of trials to run for each context length.

    Returns:
        dict: A dictionary of accuracies for each tested context length.
    """
    model.eval()
    accuracies = {}

    for length in [
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
    ]:
        if length > max_length:
            break

        correct_retrievals = 0

        for _ in range(num_trials):
            passkey = "".join([str(random.randint(0, 9)) for _ in range(5)])
            prompt = generate_passkey_prompt(passkey, length)

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model(input_ids)
                generated_ids = output.argmax(dim=-1)

            generated_text = tokenizer.decode(generated_ids[0])
            if passkey in generated_text:
                correct_retrievals += 1

        accuracies[length] = correct_retrievals / num_trials

    return accuracies


def evaluate_passkey_retrieval(model, tokenizer, max_length):
    accuracies = passkey_retrieval_test(model, tokenizer, max_length)
    for length, accuracy in accuracies.items():
        print(f"Passkey retrieval accuracy at {length} tokens: {accuracy:.2f}")
        wandb.log({f"passkey_retrieval_{length}": accuracy})
