import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional

# For convenience, you can keep or reuse some elements like cross_entropy in the same style
# as in lingua/transformer.py, but ensure references to LLaMA-specific features (e.g. rotary
# embeddings, RMSNorm, "multiple_of" feed-forward norms) are removed or replaced.

def cross_entropy(pred, target, **kwargs):
    """
    Optional: same cross_entropy from llama code (lingua/transformer.py).
    Typically used for next-token prediction.
    """
    from torch.nn import functional as F
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


@dataclass
class GPT2TransformerArgs:
    """Minimal GPT-2 style arguments, separate from any LLaMA specifics."""
    dim: int = 768                # e.g. hidden size
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 1024       # GPT2 typically 1024 context
    norm_eps: float = 1e-5
    # Additional GPT2-specific fields if needed (e.g. feed-forward factor, LR scheduling, etc.)


class GPT2Attention(nn.Module):
    """
    GPT-2 style attention block:
    - no rope embeddings
    - typically uses learned positional embeddings
    - uses nn.LayerNorm, etc.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.register_buffer("mask", None, persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, S, C = x.shape
        qkv = self.c_attn(x)  # -> [B, S, 3 * C]
        q, k, v = qkv.split(C, dim=2)

        # Reshape into [B, S, n_heads, head_dim] then transpose for attn
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nH, S, head_dim]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal mask: normally GPT-2 uses a lower-triangular approach
        # If an explicit attention_mask is provided, we incorporate it
        att_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        if attention_mask is not None:
            att_weights = att_weights + attention_mask
        att_weights = nn.functional.softmax(att_weights, dim=-1)

        out = torch.matmul(att_weights, v)  # [B, nH, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    """
    GPT-2 style Feed Forward block:
    - Typically uses a single hidden layer of 4 * hidden_size
    - with a GELU activation and then projecting back to hidden_size
    """
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.c_fc = nn.Linear(hidden_size, inner_dim, bias=True)
        self.c_proj = nn.Linear(inner_dim, hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    """
    GPT-2 Transformer Block:
    - uses Attention + MLP + residual + layer norm
    """
    def __init__(self, args: GPT2TransformerArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.attn = GPT2Attention(hidden_size=args.dim, num_heads=args.n_heads)
        self.ln_2 = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.mlp = GPT2MLP(hidden_size=args.dim, mlp_ratio=4.0)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Attention sub-layer
        h = self.ln_1(x)
        x = x + self.attn(h, attention_mask=attention_mask)

        # Feedforward sub-layer
        h2 = self.ln_2(x)
        x = x + self.mlp(h2)
        return x


class GPT2BaseTransformer(nn.Module):
    """
    A GPT-2 style base module, with learned position embeddings
    plus a stack of GPT2Block modules.
    """
    def __init__(self, args: GPT2TransformerArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.n_positions, args.dim)     # position embeddings
        self.wpe = nn.Embedding(args.n_positions, args.dim)     # optional: GPT uses wte for tokens, wpe for positions
        self.blocks = nn.ModuleList([GPT2Block(args) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [batch_size, seq_length] => token indices
        Or if embedding is done externally, x might be [batch_size, seq_length, dim]
        For minimal changes, assume x is token indices and we do the embedding here.
        """
        # If x is tokens, we embed them
        bsz, seqlen = x.shape
        positions = torch.arange(0, seqlen, device=x.device).unsqueeze(0)
        # token embeddings
        token_emb = self.wte(x)  # [B, S, dim]
        # position embeddings
        pos_emb = self.wpe(positions)  # [1, S, dim]
        h = token_emb + pos_emb

        # Potentially create a causal mask
        # GPT2's default mask = lower-triangular
        # shape: (1, 1, seq_len, seq_len), with 0 or -inf
        if attention_mask is None:
            causal_mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attention_mask = causal_mask.unsqueeze(0)  # broadcast across batch and heads

        # pass through blocks
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask)

        return self.ln_f(h)


#
# Optionally provide an init_weights routine or any additional GPT-2–style settings.
# Then you can import GPT2BaseTransformer in apps/gpt2/transformer.py and build
# your GPT2 model on top of it (e.g. add output layer tied to embeddings, etc.).
# 