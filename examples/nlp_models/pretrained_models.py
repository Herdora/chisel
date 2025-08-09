#!/usr/bin/env python3
"""
Pretrained model example using capture_model decorator.
This demonstrates profiling of HuggingFace models.
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from kandc import capture_model_instance


# Use capture_model to wrap HuggingFace models directly
def create_wrapped_distilbert():
    """Create a wrapped DistilBERT model for profiling."""
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    return capture_model_instance(model, model_name="DistilBERT")


def create_wrapped_distilbert_classifier(num_labels=2):
    """Create a wrapped DistilBERT classifier for profiling."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    return capture_model_instance(model, model_name="DistilBERTClassifier")


def create_sample_text_data(tokenizer, texts, max_length=128, device="cpu"):
    """Create tokenized text data."""
    encoding = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


def main():
    """Test pretrained models with profiling."""
    print("🚀 Testing Pretrained Models")
    print("=" * 40)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Sample texts for testing
    sample_texts = [
        "This is a great example of using Keys & Caches for model profiling.",
        "The capture_model decorator automatically profiles PyTorch models.",
        "GPU acceleration makes deep learning much faster.",
        "Transformers have revolutionized natural language processing.",
    ]

    print(f"📊 Sample texts: {len(sample_texts)} examples")

    # Tokenize input
    inputs = create_sample_text_data(tokenizer, sample_texts, device=device)
    print(f"📊 Input IDs shape: {inputs['input_ids'].shape}")
    print(f"📊 Attention mask shape: {inputs['attention_mask'].shape}")

    # Test DistilBERT base model
    print("\n🔍 Testing DistilBERT Base Model")
    print("-" * 40)

    try:
        base_model = create_wrapped_distilbert().to(device)
        print(f"📊 Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

        base_model.eval()
        with torch.no_grad():
            for i in range(3):
                print(f"🔄 Base model forward pass {i + 1}...")
                outputs = base_model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                hidden_states = outputs.last_hidden_state
                print(f"  Hidden states shape: {hidden_states.shape}")
                print(
                    f"  Hidden states range: [{hidden_states.min().item():.4f}, {hidden_states.max().item():.4f}]"
                )

    except Exception as e:
        print(f"❌ Error with base model: {e}")
        print("💡 This might be due to missing transformers library or network issues")

    # Test DistilBERT classifier
    print("\n🔍 Testing DistilBERT Classifier")
    print("-" * 40)

    try:
        classifier_model = create_wrapped_distilbert_classifier(num_labels=2).to(device)
        print(
            f"📊 Classifier parameters: {sum(p.numel() for p in classifier_model.parameters()):,}"
        )

        classifier_model.eval()
        with torch.no_grad():
            for i in range(3):
                print(f"🔄 Classifier forward pass {i + 1}...")
                outputs = classifier_model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                logits = outputs.logits
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                print(f"  Probabilities for first example: {probs[0].tolist()}")

    except Exception as e:
        print(f"❌ Error with classifier: {e}")
        print("💡 This might be due to missing transformers library or network issues")

    print("\n✅ Pretrained models testing completed!")
    print("💡 Note: This example requires 'transformers' library")
    print("   Install with: pip install transformers")


if __name__ == "__main__":
    main()
