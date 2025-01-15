import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math
from lingua.transformer import cross_entropy, RMSNorm

@dataclass
class GPT2TransformerArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 4096  # Increased from default 1024
    norm_eps: float = 1e-5
    norm_type: str = "layer_norm"  # Add this line - can be "layer_norm" or "rms_norm"
    vocab_size: int = 50257  # GPT2 vocabulary size
    dropout: float = 0.1     # Added dropout
    seed: int = 42

class GPT2Attention(nn.Module):
    """
    GPT-2 style attention block:
    - no rope embeddings
    - typically uses learned positional embeddings
    - uses nn.LayerNorm
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, S, C = x.shape  # batch, sequence, channels
        
        # Project to Q, K, V
        qkv = self.c_attn(x)  # [B, S, 3*C]
        q, k, v = qkv.split(self.hidden_size, dim=2)
        
        # Reshape to [B, n_heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

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
    - Layer norm is applied before each sub-block (pre-norm)
    """
    def __init__(self, args: GPT2TransformerArgs):
        super().__init__()
        # Choose normalization type based on args
        norm_class = nn.LayerNorm if args.norm_type == "layer_norm" else RMSNorm
        self.ln_1 = norm_class(args.dim, eps=args.norm_eps)
        self.attn = GPT2Attention(
            hidden_size=args.dim,
            num_heads=args.n_heads,
            dropout=args.dropout
        )
        self.ln_2 = norm_class(args.dim, eps=args.norm_eps)
        self.mlp = GPT2MLP(hidden_size=args.dim, dropout=args.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Pre-norm for attention
        h = self.ln_1(x)
        h = self.attn(h, attention_mask=attention_mask)
        x = x + h

        # Pre-norm for MLP
        h = self.ln_2(x)
        h = self.mlp(h)
        x = x + h
        
        return x


class GPT2BaseTransformer(nn.Module):
    def __init__(self, args: GPT2TransformerArgs):
        super().__init__()
        self.args = args
        
        # Token and position embeddings
        self.wte = nn.Embedding(args.vocab_size, args.dim)
        self.wpe = nn.Embedding(args.n_positions, args.dim)
        self.drop = nn.Dropout(args.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([GPT2Block(args) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.dim, eps=args.norm_eps)
        
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
        returns: (loss, logits) if labels provided, else (None, logits)
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

        # Get positions
        positions = torch.arange(0, seqlen, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_emb = self.wte(input_ids)  # [B, S, dim]
        pos_emb = self.wpe(positions)     # [1, S, dim]
        hidden_states = self.drop(token_emb + pos_emb)

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
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        if labels is not None:
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