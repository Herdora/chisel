#!/usr/bin/env python3
"""
Model example using custom requirements file.
Run with: chisel python model_with_custom_requirements.py --requirements requirements_examples/nlp_requirements.txt
"""

import torch
import torch.nn as nn
from chisel import capture_model_class

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@capture_model_class(model_name="CustomRequirementsModel")
class CustomRequirementsModel(nn.Module):
    """Model that demonstrates custom requirements usage."""

    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)

        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        logits = self.classifier(last_hidden)  # (batch_size, 2)

        return logits


def test_optional_dependencies():
    """Test which optional dependencies are available."""
    print("📦 Testing Optional Dependencies:")

    deps = [
        ("matplotlib", HAS_MATPLOTLIB),
        ("transformers", HAS_TRANSFORMERS),
        ("pandas", HAS_PANDAS),
    ]

    for name, available in deps:
        status = "✅" if available else "❌"
        print(f"   {status} {name}")

    return deps


def create_sample_text_data(batch_size=4, seq_len=50, vocab_size=1000):
    """Create sample text data."""
    # Random token IDs
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

    # Random labels for binary classification
    labels = torch.randint(0, 2, (batch_size,))

    return input_ids, labels


def demonstrate_transformers_integration():
    """Demonstrate HuggingFace transformers integration if available."""
    if not HAS_TRANSFORMERS:
        print("⚠️  Transformers not available, skipping integration demo")
        return None

    print("🤗 Testing HuggingFace Transformers Integration:")

    try:
        # Use a small, fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        sample_texts = [
            "This is a test sentence.",
            "Another example with different words.",
            "Machine learning is fascinating.",
            "PyTorch makes deep learning accessible.",
        ]

        print(f"   Tokenizing {len(sample_texts)} texts...")

        encoded = tokenizer(
            sample_texts, padding=True, truncation=True, max_length=32, return_tensors="pt"
        )

        print(f"   Encoded input_ids shape: {encoded['input_ids'].shape}")
        print(f"   Encoded attention_mask shape: {encoded['attention_mask'].shape}")

        return encoded["input_ids"]

    except Exception as e:
        print(f"   ❌ Error with transformers: {e}")
        return None


def create_visualization_if_possible(losses):
    """Create a simple visualization if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available, skipping visualization")
        return

    print("📊 Creating loss visualization...")

    try:
        plt.figure(figsize=(8, 4))
        plt.plot(losses, "b-", linewidth=2)
        plt.title("Training Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        # Save plot instead of showing (since we're in headless environment)
        plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("   ✅ Saved training_loss.png")

    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")


def main():
    """Test model with custom requirements."""
    print("🚀 Custom Requirements Model Test")
    print("=" * 40)

    # Test dependencies
    deps = test_optional_dependencies()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎯 Using device: {device}")

    # Model configuration
    vocab_size = 1000
    batch_size = 8
    seq_len = 32

    # Create model
    model = CustomRequirementsModel(vocab_size, embed_dim=128, hidden_dim=256).to(device)
    print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with random data
    print(f"\n🧪 Testing with random data:")
    input_ids, labels = create_sample_text_data(batch_size, seq_len, vocab_size)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Labels shape: {labels.shape}")

    model.eval()
    with torch.no_grad():
        for i in range(3):
            logits = model(input_ids)

            # Calculate loss
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()

            print(f"   Pass {i + 1}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")

    # Test transformers integration if available
    if HAS_TRANSFORMERS:
        print(f"\n🤗 Testing with HuggingFace tokenized data:")
        tokenized_data = demonstrate_transformers_integration()

        if tokenized_data is not None:
            tokenized_data = tokenized_data.to(device)
            with torch.no_grad():
                logits = model(tokenized_data)
                print(f"   Tokenized data shape: {tokenized_data.shape}")
                print(f"   Model output shape: {logits.shape}")

    # Create visualization if possible
    fake_losses = [0.8, 0.6, 0.4, 0.3, 0.25, 0.22, 0.20, 0.18]
    create_visualization_if_possible(fake_losses)

    # Summary
    print(f"\n✅ Custom requirements test completed!")
    print(f"💡 This example demonstrates:")
    print(f"   - Using custom requirements files")
    print(f"   - Optional dependency handling")
    print(f"   - Integration with popular ML libraries")

    available_count = sum(1 for _, available in deps if available)
    total_count = len(deps)
    print(f"📊 Dependencies available: {available_count}/{total_count}")


if __name__ == "__main__":
    main()
