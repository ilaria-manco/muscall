"""Code based on https://github.com/lucidrains/x-clip/blob/main/x_clip/"""

import torch
from torch import nn, einsum
from einops import rearrange


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim, num_attention_heads, dropout, output_dim=None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_projection = nn.Linear(embed_dim, output_dim or embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # b = batch size, n = sequence length, h = num of heads, d = head_dim, (h x d) = embed_dim
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_attention_heads),
            (q, k, v),
        )

        scaled_dot_product = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if mask is not None:
            scaled_dot_product = scaled_dot_product + mask

        attn_weights = scaled_dot_product.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_projection(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        inner_dim = int(embed_dim * 4)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([])

        num_hidden_layers = config.num_hidden_layers
        embed_dim = config.hidden_size
        num_attention_heads = config.num_attention_heads
        attention_dropout_p = config.attention_dropout
        dropout_p = config.dropout

        for _ in range(num_hidden_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            embed_dim,
                            MultiHeadAttention(
                                embed_dim, num_attention_heads, attention_dropout_p
                            ),
                        ),
                        PreNorm(embed_dim, FeedForward(embed_dim, dropout_p)),
                    ]
                )
            )

        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return self.norm_out(x)
