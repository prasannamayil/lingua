import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math
from lingua.transformer import cross_entropy, RMSNorm
import torch.nn.functional as F

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimal copy of apply_rotary_emb from lingua/transformer.py.
    We assume the input shapes are [B, n_heads, S, head_dim].
    """
    # shape handling
    bsz, nh, s, hd = q.shape
    # freq_cis: [S, hd, 2, 2]
    # Reshape q, k to [B, n_heads, S, hd/2, 1, 2] for the 2D rotation
    q_ = q.reshape(bsz, nh, s, hd // 2, 1, 2)
    k_ = k.reshape(bsz, nh, s, hd // 2, 1, 2)

    # We'll broadcast freq_cis along B, n_heads
    # freq_cis is shaped for [S, hd/2, 2, 2], we re-index with S dimension
    # The minimal approach is to slice freq_cis if needed:
    freq_cis_sliced = freqs_cis[:s, : (hd // 2)]

    # Expand them to match [1, 1, S, hd/2, 2, 2]
    freq_cis_sliced = freq_cis_sliced.unsqueeze(0).unsqueeze(0)
    # Now multiply
    # (q_ * freq_cis): shape broadcast => [B, n_heads, S, hd/2, 1, 2]
    # Summation over last dimension
    q_out = (q_ * freq_cis_sliced).sum(-1)  # now [B, n_heads, S, hd/2]
    k_out = (k_ * freq_cis_sliced).sum(-1)
    # Flatten back
    q_out = q_out.reshape(bsz, nh, s, hd)
    k_out = k_out.reshape(bsz, nh, s, hd)
    return q_out.type_as(q), k_out.type_as(k)


class RotaryEmbedding(nn.Module):
    """
    Minimal RotaryEmbedding structure, closely mimicking lingua/transformer.py.
    Generates precomputed freqs_cis for up to n_positions.
    """

    def __init__(self, head_dim: int, max_seqlen: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.theta = theta
        self.register_buffer(
            "freqs_cis", self.precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta), persistent=False
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        """
        Minimal replication of precompute_freqs_cis from lingua/transformer.py.
        Returns shape: [end, dim, 2, 2]
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        cos, sin = freqs.cos(), freqs.sin()
        # stack
        return torch.stack((cos, -sin, sin, cos), dim=-1).view(end, dim // 2, 2, 2)

    def forward(self, seqlen: int):
        """
        Return the precomputed freqs_cis for up to seqlen.
        """
        return self.freqs_cis[:seqlen]


@dataclass
class GPT2TransformerArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 4096
    norm_eps: float = 1e-5
    norm_type: str = "layer_norm"  # can be "layer_norm" or "rms_norm"
    vocab_size: int = 50257
    dropout: float = 0.1
    seed: int = 42
    # New flags for minimal changes
    use_pre_norm: bool = True
    use_rope: bool = False  # set True to replace position embeddings with rope embeddings


class GPT2Attention(nn.Module):
    """
    GPT-2 style attention block:
    - no rope embeddings by default (but can be used if 'use_rope' is True)
    - uses nn.LayerNorm (or RMSNorm if requested)
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Combined QKV projection
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        # Output projection
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freq_cis: Optional[torch.Tensor] = None,  # freq_cis for rope
    ):
        B, S, C = x.shape  # batch, sequence, channels

        # Project to Q, K, V
        qkv = self.c_attn(x)  # [B, S, 3*C]
        q, k, v = qkv.split(self.hidden_size, dim=2)

        # Reshape to [B, n_heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, S, hd]
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # -------------------------------------------------------------------
        # Rope embedding application if freq_cis is not None
        # -------------------------------------------------------------------
        if freq_cis is not None:
            q, k = apply_rotary_emb(q, k, freq_cis)

        # Compute attention scores
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask

        # Softmax and dropout
        att = self.attn_dropout(torch.softmax(att, dim=-1))

        # Apply attention to values
        out = torch.matmul(att, v)  # [B, n_heads, S, head_dim]

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        out = self.resid_dropout(self.c_proj(out))

        return out


class GPT2MLP(nn.Module):
    """
    GPT-2 style Feed Forward block:
    - Uses a single hidden layer of 4 * hidden_size
    - GELU activation
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c_fc(x))
        x = self.dropout(self.c_proj(x))
        return x


class GPT2Block(nn.Module):
    """
    GPT-2 Transformer Block:
    - Uses Attention + MLP
    - By default, pre-norm is applied (like GPT2). 
      But we can switch to post-norm if args.use_pre_norm=False
    """
    def __init__(self, args: GPT2TransformerArgs):
        super().__init__()
        # Choose normalization type based on args
        if args.norm_type == "layer_norm":
            norm_class = nn.LayerNorm
        else:
            from lingua.transformer import RMSNorm
            norm_class = RMSNorm

        self.use_pre_norm = args.use_pre_norm

        self.ln_1 = norm_class(args.dim, eps=args.norm_eps)
        self.attn = GPT2Attention(
            hidden_size=args.dim,
            num_heads=args.n_heads,
            dropout=args.dropout
        )
        self.ln_2 = norm_class(args.dim, eps=args.norm_eps)
        self.mlp = GPT2MLP(hidden_size=args.dim, dropout=args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freq_cis: Optional[torch.Tensor] = None,
    ):
        # -------------------------------------------------------------------
        # If use_pre_norm, do it the default GPT2 "pre-norm" way. 
        # If not, switch to a "post-norm" style for demonstration.
        # -------------------------------------------------------------------
        if self.use_pre_norm:
            # Pre-norm approach (GPT2 default)
            h = self.ln_1(x)
            h = self.attn(h, attention_mask=attention_mask, freq_cis=freq_cis)
            x = x + h

            h = self.ln_2(x)
            h = self.mlp(h)
            x = x + h
        else:
            # Post-norm approach
            h = self.attn(x, attention_mask=attention_mask, freq_cis=freq_cis)
            x = self.ln_1(x + h)

            h = self.mlp(x)
            x = self.ln_2(x + h)

        return x


class GPT2BaseTransformer(nn.Module):
    def __init__(self, args: GPT2TransformerArgs):
        super().__init__()
        self.args = args

        # Choose the norm class
        if self.args.norm_type == "layer_norm":
            norm_class = nn.LayerNorm
        else:
            from lingua.transformer import RMSNorm
            norm_class = RMSNorm

        # Token embedding
        self.wte = nn.Embedding(args.vocab_size, args.dim)

        # -------------------------------------------------------------------
        # If not using rope, keep the original positional embedding
        # else we define rope embeddings
        # -------------------------------------------------------------------
        if not args.use_rope:
            self.wpe = nn.Embedding(args.n_positions, args.dim)
        else:
            self.wpe = None
            # Create rope embedding object
            self.rope_embeddings = RotaryEmbedding(
                head_dim=args.dim // args.n_heads,
                max_seqlen=args.n_positions,
                theta=10000.0,  # can adjust if needed
            )

        self.drop = nn.Dropout(args.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([GPT2Block(args) for _ in range(args.n_layers)])
        self.ln_f = norm_class(args.dim, eps=args.norm_eps)

        # Language modeling head
        # Tie weights with token embedding
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: [batch_size, seq_length]
        labels: [batch_size, seq_length] or None
        returns: (loss) if labels provided, else (logits)
        """
        bsz, seqlen = input_ids.shape

        # Input validation
        if seqlen > self.args.n_positions:
            raise ValueError(
                f"Input sequence length ({seqlen}) exceeds model's maximum "
                f"position embedding size ({self.args.n_positions})"
            )

        if torch.any(input_ids >= self.args.vocab_size) or torch.any(input_ids < 0):
            raise ValueError(
                f"Input contains token ids outside valid range [0, {self.args.vocab_size-1}]. "
                f"Min: {input_ids.min().item()}, Max: {input_ids.max().item()}"
            )

        token_emb = self.wte(input_ids)  # [B, S, dim]

        # -------------------------------------------------------------------
        # Position embeddings if not rope. If rope, we'll apply it in the attention.
        # -------------------------------------------------------------------
        if not self.args.use_rope:
            positions = torch.arange(0, seqlen, device=input_ids.device).unsqueeze(0)
            pos_emb = self.wpe(positions)  # [1, S, dim]
            hidden_states = self.drop(token_emb + pos_emb)
            freq_cis = None
        else:
            # Skip adding any wpe
            hidden_states = self.drop(token_emb)
            # Build freq_cis for rope
            freq_cis = self.rope_embeddings(seqlen)

        # Create causal mask if none provided
        if attention_mask is None:
            attention_mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=input_ids.device
            )
            attention_mask = torch.triu(attention_mask, diagonal=1)
            attention_mask = attention_mask.unsqueeze(0)  # [1, S, S]
            attention_mask = attention_mask.to(hidden_states.dtype)

        # Forward through transformer layers
        for block in self.layers:
            hidden_states = block(hidden_states, attention_mask=attention_mask, freq_cis=freq_cis)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        if labels is not None:
            from lingua.transformer import cross_entropy
            loss = cross_entropy(logits, labels)
            return loss
        else:
            return logits

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        self.apply(_init_weights)