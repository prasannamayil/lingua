import time
import torch
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf

# We reuse the same generator from llama example with minimal changes
from apps.main.transformer import causal_mask  # or define your own
from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
# Import GPT2 classes
from apps.gpt2.transformer import GPT2Transformer, GPT2TransformerArgs
from lingua.tokenizer import Tokenizer, build_tokenizer


def main():
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, cfg, strict=False
    )
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(
        cfg.ckpt,
        model_cls=GPT2Transformer,
        model_args_cls=GPT2TransformerArgs,
    )

    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main() 