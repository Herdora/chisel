#!/usr/bin/env python
"""
toy_gpt.py
-----------
A *tiny* GPT‑style model that is easy to read and tweak.
It does just enough to verify that:
* The tensors land on your GPU / ROCm device
* A dummy forward pass completes and returns logits `(batch, seq, vocab)`.

It purposefully avoids most engineering details:
* **No custom attention mask logic** – we rely on PyTorch’s causal mask.
* **No dropout / weight‑tying / gelu** – keeps it minimal.
* **< 50 lines of code** including the CLI.
"""

import argparse
import torch
import torch.nn as nn


class ToyGPT(nn.Module):
    """Super‑lightweight causal Transformer."""

    def __init__(
        self, vocab_size=50_257, n_embd=256, n_head=4, n_layer=2, block_size=128
    ):
        super().__init__()
        self.block_size = block_size

        # Embeddings
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Parameter(torch.zeros(1, block_size, n_embd))

        # A stack of vanilla Transformer encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        # Projection back to vocab
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.LongTensor):
        """idx – `(B, T)` ints in `[0, vocab)`"""
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.tok(idx) + self.pos[:, :T]

        # PyTorch 2.2+ supports built‑in causal mask generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, x.device
        )
        x = self.transformer(x, mask=causal_mask)
        return self.head(x)  # (B, T, vocab)


def main():
    p = argparse.ArgumentParser(
        description="Run a dummy forward pass on the simplest GPT."
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    args = p.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("hip")
        if torch.version.hip
        else torch.device("cpu")
    )

    model = ToyGPT(block_size=args.seq).to(device)
    idx = torch.randint(0, 50_257, (args.batch, args.seq), device=device)

    with torch.inference_mode():
        logits = model(idx)
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()
