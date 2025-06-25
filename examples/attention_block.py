# attention_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleAttention, self).__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5  # for scaling dot product

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(attn_weights, V)  # (B, T, D)
        return out

# Example usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    embed_dim = 8

    x = torch.randn(batch_size, seq_len, embed_dim)
    attention = SimpleAttention(embed_dim)
    out = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

