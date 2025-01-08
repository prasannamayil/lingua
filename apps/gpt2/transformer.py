import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# For minimal changes, we reuse the same base classes and cross_entropy
# from the original scripts.
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    cross_entropy,
)
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import create_block_mask, BlockMask


# We replace RMSNorm usage with nn.LayerNorm for GPT2
# We also rename the model from LMTransformer -> GPT2Transformer

def create_causal_mask(seqlen, attn_impl, sliding_window):
    """
    Minimal copy from the llama code. GPT2 also uses a causal (lower-triangular)
    attention mask. We keep references to local/sliding window if you'd like,
    or remove them. We'll keep them for minimal changes.
    """
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        # Note: causal_mask() is not explicitly in this file, so you may repurpose or define it here
        # or do something simpler that yields a causal block mask. We'll keep the original.
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


@dataclass
class GPT2TransformerArgs(BaseTransformerArgs):
    """
    Minimal changes from 'LMTransformerArgs'.
    GPT2 typically uses LayerNorm, so we remove references to RMSNorm.
    """
    seed: int = 42
    vocab_size: int = -1
    weight_tying: bool = True  # GPT2 typically ties input/output embeddings
    sliding_window: Optional[int] = None


class GPT2Transformer(BaseTransformer):
    """
    A minimal GPT2-like Transformer that replaces RMSNorm with nn.LayerNorm
    and reuses the rest. 
    """

    def __init__(self, args: GPT2TransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0, "GPT2 requires a vocab size > 0"

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # GPT2 uses standard LayerNorm with bias
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        # If we want weight tying
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        # The super().forward() calls into the stacked Transformer layers from BaseTransformer
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        logits = self.output(self.norm(h))
        if target is not None:
            return cross_entropy(logits, target)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        """
        Minimal re-implementation, optionally using GPT2 initialization, 
        or leaving it basically the same from LLaMA code. We'll keep minimal changes.
        """
        super().reset_parameters()
        init_std = init_std or (self.dim ** -0.5)
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )