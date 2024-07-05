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
