#!/usr/bin/env python3
"""
HuggingFace model download and profiling example.
This downloads actual model weights from HuggingFace and loads them into a capture_model decorated class.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from kandc import capture_model_class
import os


@capture_model_class(model_name="DownloadedDistilBERT")
class DownloadedDistilBERTModel(nn.Module):
    """
    Model that downloads and loads actual DistilBERT weights from HuggingFace.
    This demonstrates profiling a real pretrained model.
    """

    def __init__(self, model_name="distilbert-base-uncased", cache_dir=None):
        super().__init__()
        self.model_name = model_name

        print(f"📦 Downloading model: {model_name}")
        print("   This may take a few minutes on first run...")

        # Download and load the actual model weights
        self.distilbert = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # Ensure consistent dtype
        )

        # Add a simple classification head for demonstration
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.distilbert.config.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # Binary classification
        )

        print(f"✅ Model downloaded and loaded successfully!")
        print(f"   Model config: {self.distilbert.config}")

    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from DistilBERT
        outputs = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Use [CLS] token representation (first token)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Apply classification head
        logits = self.classifier(cls_hidden_state)  # (batch_size, num_classes)

        return {
            "logits": logits,
            "hidden_states": outputs.last_hidden_state,
            "cls_representation": cls_hidden_state,
        }


@capture_model_class(model_name="DownloadedGPT2")
class DownloadedGPT2Model(nn.Module):
    """
    Model that downloads and loads actual GPT-2 weights from HuggingFace.
    """

    def __init__(self, model_name="gpt2", cache_dir=None, max_length=512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        print(f"📦 Downloading GPT-2 model: {model_name}")

        # Load model and config
        from transformers import GPT2LMHeadModel

        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
        )

        print(f"✅ GPT-2 model downloaded and loaded!")
        print(f"   Vocab size: {self.gpt2.config.vocab_size}")
        print(f"   Model size: {sum(p.numel() for p in self.gpt2.parameters()):,} parameters")

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        }


def download_and_test_distilbert(cache_dir="./model_cache"):
    """Download and test DistilBERT model."""
    print("🔍 Testing Downloaded DistilBERT Model")
    print("-" * 45)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Create model (this will download weights)
        model = DownloadedDistilBERTModel(
            model_name="distilbert-base-uncased", cache_dir=cache_dir
        ).to(device)

        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)

        # Sample texts
        texts = [
            "This is an example of using downloaded HuggingFace models with Keys & Caches profiling.",
            "The capture_model decorator will profile every forward pass automatically.",
            "GPU acceleration makes transformer inference much faster.",
            "Pretrained models from HuggingFace work seamlessly with Keys & Caches.",
        ]

        print(f"📝 Testing with {len(texts)} sample texts")

        # Tokenize
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        print(f"📊 Input shape: {input_ids.shape}")

        # Run inference (this will be profiled!)
        model.eval()
        with torch.no_grad():
            for i in range(3):
                print(f"\n🔄 DistilBERT forward pass {i + 1}...")

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs["logits"]
                cls_repr = outputs["cls_representation"]

                print(f"   Logits shape: {logits.shape}")
                print(f"   CLS representation shape: {cls_repr.shape}")
                print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

                # Get predictions
                predictions = torch.softmax(logits, dim=-1)
                print(f"   Sample predictions: {predictions[0].tolist()}")

        return True

    except Exception as e:
        print(f"❌ Error with DistilBERT: {e}")
        print("💡 This might be due to network issues or missing transformers library")
        return False


def download_and_test_gpt2(cache_dir="./model_cache"):
    """Download and test GPT-2 model."""
    print("\n🔍 Testing Downloaded GPT-2 Model")
    print("-" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Create model (this will download weights)
        model = DownloadedGPT2Model(
            model_name="gpt2",  # Smallest GPT-2 model
            cache_dir=cache_dir,
        ).to(device)

        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Load tokenizer
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

        # Sample prompts
        prompts = [
            "The future of artificial intelligence is",
            "Machine learning models can help us",
            "In the world of deep learning,",
            "Transformers have revolutionized",
        ]

        print(f"📝 Testing with {len(prompts)} sample prompts")

        # Tokenize
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        print(f"📊 Input shape: {input_ids.shape}")

        # Run inference (this will be profiled!)
        model.eval()
        with torch.no_grad():
            for i in range(3):
                print(f"\n🔄 GPT-2 forward pass {i + 1}...")

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs["logits"]

                print(f"   Logits shape: {logits.shape}")
                print(f"   Vocab predictions shape: {logits.shape}")
                print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

                # Get top predictions for first prompt's last token
                last_token_logits = logits[0, -1, :]  # Last token of first sequence
                top_k_tokens = torch.topk(last_token_logits, k=5)

                print("   Top 5 next token predictions:")
                for j, (score, token_id) in enumerate(
                    zip(top_k_tokens.values, top_k_tokens.indices)
                ):
                    token = tokenizer.decode(token_id.item())
                    print(f"     {j + 1}. '{token}' (score: {score.item():.4f})")

        return True

    except Exception as e:
        print(f"❌ Error with GPT-2: {e}")
        print("💡 This might be due to network issues or missing transformers library")
        return False


def show_cache_info(cache_dir="./model_cache"):
    """Show information about downloaded model cache."""
    print(f"\n📁 Model Cache Information")
    print("-" * 30)

    if os.path.exists(cache_dir):
        print(f"📂 Cache directory: {os.path.abspath(cache_dir)}")

        # Calculate total cache size
        total_size = 0
        file_count = 0

        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    total_size += size
                    file_count += 1
                except OSError:
                    pass

        size_mb = total_size / (1024 * 1024)
        print(f"📊 Total cache size: {size_mb:.1f} MB")
        print(f"📊 Files cached: {file_count}")

        # List model directories
        if os.path.exists(cache_dir):
            subdirs = [
                d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))
            ]
            if subdirs:
                print(f"📦 Cached models:")
                for subdir in subdirs[:5]:  # Show first 5
                    print(f"   - {subdir}")
                if len(subdirs) > 5:
                    print(f"   ... and {len(subdirs) - 5} more")
    else:
        print(f"📂 Cache directory doesn't exist yet: {cache_dir}")
        print("💡 Models will be downloaded on first run")


def main():
    """Test downloading and profiling HuggingFace models."""
    print("🚀 HuggingFace Model Download and Profiling")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Set up cache directory
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"📂 Model cache directory: {os.path.abspath(cache_dir)}")
    print("💡 Models will be downloaded to this directory for caching")

    # Show current cache state
    show_cache_info(cache_dir)

    # Test DistilBERT
    distilbert_success = download_and_test_distilbert(cache_dir)

    # Test GPT-2
    gpt2_success = download_and_test_gpt2(cache_dir)

    # Final cache info
    show_cache_info(cache_dir)

    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 30)

    print(f"✅ DistilBERT: {'Success' if distilbert_success else 'Failed'}")
    print(f"✅ GPT-2: {'Success' if gpt2_success else 'Failed'}")

    if distilbert_success or gpt2_success:
        print(f"\n🎉 Successfully downloaded and profiled HuggingFace models!")
        print(f"💡 Key features demonstrated:")
        print(f"   - Automatic model weight downloading")
        print(f"   - Model caching for faster subsequent runs")
        print(f"   - capture_model decorator profiling of real models")
        print(f"   - Integration with HuggingFace tokenizers")
        print(f"   - Both encoder (DistilBERT) and decoder (GPT-2) models")
    else:
        print(f"\n⚠️  Model download/testing failed")
        print(f"💡 Possible solutions:")
        print(f"   - Check internet connection")
        print(f"   - Install transformers: pip install transformers")
        print(f"   - Try running locally first: python {__file__}")

    print(f"\n🚀 Example Usage:")
    print(f"   # Run locally first")
    print(f"   python nlp_models/huggingface_download_example.py")
    print(f"   ")
    print(f"   # Run with profiling on cloud GPU")
    print(f"   kandc python nlp_models/huggingface_download_example.py")
    print(f"   ")
    print(f"   # With custom requirements")
    print(
        f"   kandc python nlp_models/huggingface_download_example.py --requirements requirements_examples/nlp_requirements.txt"
    )


if __name__ == "__main__":
    main()
