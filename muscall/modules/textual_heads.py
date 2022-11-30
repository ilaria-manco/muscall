import torch
from torch import nn
from einops import rearrange

from muscall.modules.transformer import Transformer, LayerNorm


class TextualHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class TextTransformer(TextualHead):
    def __init__(self, config):
        super().__init__(config)

        vocab_size = config.vocab_size
        embed_dim = config.hidden_size
        context_length = config.max_position_embeddings

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        self.transformer = Transformer(config)

        self.ln_final = LayerNorm(embed_dim)

    def _build_causal_attention_mask(self, batch_size, seq_len):
        """Create causal attention mask.

        This is a triangular matrix with upper diagonal values set to -inf.
        This is because we're using an additive mask in the attention module.
        """
        mask = torch.empty(batch_size, seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def forward(self, x: torch.Tensor, mask=None):
        b, n, device = *x.shape, x.device

        input_embeddings = self.token_embedding(x)  # [batch_size, tokens, hidden_dim]
        position_embeddings = self.position_embedding(torch.arange(n, device=device))
        embeddings = input_embeddings + rearrange(position_embeddings, "n d -> 1 n d")

        batch_size, seq_len, _ = embeddings.size()
        causal_attention_mask = self._build_causal_attention_mask(
            batch_size, seq_len
        ).to(embeddings.device)

        out = self.transformer(embeddings, mask=causal_attention_mask)
        return out
