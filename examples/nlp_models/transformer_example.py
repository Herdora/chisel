#!/usr/bin/env python3
"""
Transformer model example using capture_model decorator.
This demonstrates NLP model profiling with attention mechanisms.
"""

import torch
import torch.nn as nn
import math
from chisel import capture_model_class


@capture_model_class(model_name="SimpleTransformer")
class SimpleTransformer(nn.Module):
    """Simplified transformer model for text processing."""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_mask=None):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)

        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        # Generate causal mask for autoregressive generation
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer forward pass
        x = self.transformer(x, src_mask)  # (batch_size, seq_len, d_model)

        # Output projection
        output = self.output_projection(x)  # (batch_size, seq_len, vocab_size)

        return output

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


@capture_model_class(model_name="BERTLikeModel")
class BERTLikeModel(nn.Module):
    """BERT-like model for masked language modeling."""

    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=3072, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # Create padding mask if provided
        padding_mask = None
        if attention_mask is not None:
            # Convert attention mask to padding mask for transformer
            # attention_mask: 1 for real tokens, 0 for padding
            # padding_mask: True for padding tokens, False for real tokens
            padding_mask = attention_mask == 0

        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # MLM head
        predictions = self.mlm_head(x)

        return predictions


def create_sample_data(batch_size=4, seq_len=128, vocab_size=10000, device="cpu"):
    """Create sample text data for testing."""
    # Random token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Random attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    # Randomly mask some positions as padding
    for i in range(batch_size):
        mask_len = torch.randint(seq_len // 2, seq_len, (1,)).item()
        attention_mask[i, mask_len:] = 0

    return input_ids, attention_mask


def main():
    """Test transformer models with profiling."""
    print("🚀 Testing Transformer Models")
    print("=" * 40)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    vocab_size = 10000
    batch_size = 4
    seq_len = 128

    # Create sample data
    input_ids, attention_mask = create_sample_data(batch_size, seq_len, vocab_size, device)

    print(f"📊 Input shape: {input_ids.shape}")
    print(f"📊 Attention mask shape: {attention_mask.shape}")

    # Test Simple Transformer
    print("\n🔍 Testing Simple Transformer (GPT-like)")
    print("-" * 40)

    gpt_model = SimpleTransformer(vocab_size=vocab_size, d_model=512, nhead=8, num_layers=6).to(
        device
    )

    print(f"📊 GPT-like parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")

    gpt_model.eval()
    with torch.no_grad():
        for i in range(3):
            print(f"🔄 GPT Forward pass {i + 1}...")
            output = gpt_model(input_ids)
            print(f"  Output shape: {output.shape}")
            print(f"  Logits range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test BERT-like Model
    print("\n🔍 Testing BERT-like Model")
    print("-" * 40)

    bert_model = BERTLikeModel(
        vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_layers=6,  # Smaller than full BERT for demo
    ).to(device)

    print(f"📊 BERT-like parameters: {sum(p.numel() for p in bert_model.parameters()):,}")

    bert_model.eval()
    with torch.no_grad():
        for i in range(3):
            print(f"🔄 BERT Forward pass {i + 1}...")
            output = bert_model(input_ids, attention_mask)
            print(f"  Output shape: {output.shape}")
            print(f"  Predictions range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\n✅ Transformer models testing completed!")


if __name__ == "__main__":
    main()
