from dataclasses import dataclass, field
from typing import Optional, List
import torch
from torch import nn
import torch.nn.functional as F


from apps.main.generate import (
    pack_prompts,
    batch_prompts,
    sample_tokens,
)
from apps.gpt2.transformer import GPT2BaseTransformer
from lingua.tokenizer import Tokenizer


@dataclass 
class GPT2GeneratorArgs:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 512
    max_tokens: int = 1024
    max_prompt_len: Optional[int] = None
    until: List[str] = field(default_factory=list)
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False


class GPT2Generator:
    def __init__(
        self,
        cfg: GPT2GeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k
        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.until = cfg.until
        self.device = cfg.device
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        # Make sure the entire model is moved to the correct device & dtype
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def generate(self, prompts):
        # Tokenize
        prompts = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
        ]
        
        # Get model's max sequence length
        model_max_length = (
            2048  # Default GPT2 context length
            if not hasattr(self.model, "args") 
            else self.model.args.n_positions
        )
        
        # Truncate to the minimum of:
        # 1. model's max length
        # 2. configured max_tokens
        # 3. configured max_prompt_len
        max_allowed_len = min(
            model_max_length,
            self.max_tokens,
            self.max_prompt_len or model_max_length
        )
        
        # Ensure we leave room for generated tokens
        max_prompt_len = max_allowed_len - self.max_gen_len
        prompts = [p[-max_prompt_len:] for p in prompts]
        
        generation = []
        loglikelihood = []
        greedy = []
        
        for batch in batch_prompts(prompts, max_allowed_len):
            generated_tokens = [[] for _ in range(len(batch))]
            is_done = [False for _ in range(len(batch))]
            
            packed_batch, lengths = pack_prompts(batch)
            # Cast to model’s device & dtype:
            packed_batch = packed_batch.to(self.device)  # keep as long
            lengths = lengths.to(self.device)

            # Forward pass => now everything is bf16, matching the model
            logits = self.model(packed_batch.unsqueeze(0))
            
            # Get initial token
            all_tokens = sample_tokens(
                logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            
            # Generate remaining tokens
            for i in range(1, self.max_gen_len):
                next_logits = self.model(current_token)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        if self.until:
                            current_end_str = self.tokenizer.decode(
                                generated_tokens[seq_id][-len(max(self.until, key=len)) :]
                            )
                            contains_end_string = any(
                                [e in current_end_str for e in self.until]
                            )
                            is_done[seq_id] = (
                                contains_end_string or tok == self.tokenizer.eos_id
                            )
                    if all(is_done):
                        break

                    current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            # Calculate log likelihoods for the batch
            for p, logit in zip(batch, logits.squeeze(0).split(lengths.tolist())):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction='none').cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

        return generation, loglikelihood, greedy 