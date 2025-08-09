#!/usr/bin/env python3
"""
CLIP Vision-Language Model example using capture_model decorator.
This downloads and profiles OpenAI's CLIP model from HuggingFace.

CLIP (Contrastive Language-Image Pre-training) can understand both images and text,
making it perfect for vision-language tasks like image classification with text descriptions.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
from io import BytesIO
import os
from kandc import capture_model_instance


def create_clip_model(model_name="openai/clip-vit-base-patch32", cache_dir=None):
    """
    Create and wrap a CLIP model with capture_model for profiling.
    Downloads actual pretrained weights from HuggingFace.
    """
    print(f"📦 Downloading CLIP model: {model_name}")
    print("   This may take a few minutes on first run...")

    # Download and load the actual CLIP model
    clip_model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )

    # Load processor for text and image preprocessing
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    print(f"✅ CLIP model downloaded and loaded successfully!")
    print(f"   Vision encoder: {clip_model.vision_model.config.hidden_size}D")
    print(f"   Text encoder: {clip_model.text_model.config.hidden_size}D")

    # Wrap the model with capture_model_instance for profiling
    wrapped_model = capture_model_instance(clip_model, model_name="CLIP_VisionLanguage")

    # Attach processor to the wrapped model for convenience
    wrapped_model.processor = processor

    return wrapped_model


def download_sample_images():
    """Download sample images for testing."""
    print("📸 Downloading sample images...")

    # Sample images from the web
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/1200px-Tour_Eiffel_Wikimedia_Commons.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/ThreeTimeAKCGoldWinnerPembrookeWelshCorgi.jpg/1200px-ThreeTimeAKCGoldWinnerPembrookeWelshCorgi.jpg",
    ]

    images = []
    descriptions = [
        "a nature boardwalk through a green field",
        "a simple geometric logo design",
        "the famous Eiffel Tower in Paris",
        "a cute corgi dog sitting on grass",
    ]

    try:
        for i, url in enumerate(image_urls):
            print(f"   Downloading image {i + 1}/4...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(image)

        print(f"✅ Successfully downloaded {len(images)} sample images")
        return images, descriptions

    except Exception as e:
        print(f"❌ Error downloading images: {e}")
        print("💡 Creating synthetic images for testing...")

        # Fallback: create synthetic images
        import numpy as np

        synthetic_images = []
        for i in range(4):
            # Create different colored synthetic images
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i]
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            synthetic_images.append(Image.fromarray(img_array))

        synthetic_descriptions = [
            "a red colored image",
            "a green colored image",
            "a blue colored image",
            "a yellow colored image",
        ]

        return synthetic_images, synthetic_descriptions


def test_image_classification(model, processor, images, text_descriptions, device):
    """Test CLIP for zero-shot image classification."""
    print("\n🔍 Testing Zero-Shot Image Classification")
    print("-" * 45)

    # Classification labels
    class_labels = [
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a car",
        "a photo of a building",
        "a photo of nature",
        "a photo of food",
        "a logo or symbol",
        "a person",
    ]

    print(f"📋 Classification labels: {len(class_labels)} classes")
    print(f"🖼️  Testing images: {len(images)} images")

    # Process images and text
    inputs = processor(text=class_labels, images=images, return_tensors="pt", padding=True)

    # Move to device
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"📊 Image tensor shape: {pixel_values.shape}")
    print(f"📊 Text tensor shape: {input_ids.shape}")

    model.eval()
    with torch.no_grad():
        # Run inference (this will be profiled!)
        print(f"\n🔄 CLIP forward pass for classification...")

        outputs = model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

        logits_per_image = outputs["logits_per_image"]  # Shape: (num_images, num_classes)

        print(f"📊 Logits shape: {logits_per_image.shape}")

        # Get predictions for each image
        probs = torch.softmax(logits_per_image, dim=-1)

        print(f"\n🎯 Classification Results:")
        for i, (image_desc, image_probs) in enumerate(zip(text_descriptions, probs)):
            top_prob, top_idx = torch.max(image_probs, dim=0)
            predicted_label = class_labels[top_idx.item()]

            print(f"   Image {i + 1} ({image_desc}):")
            print(f"     Predicted: '{predicted_label}' (confidence: {top_prob.item():.3f})")

            # Show top 3 predictions
            top3_probs, top3_indices = torch.topk(image_probs, 3)
            print(f"     Top 3:")
            for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                print(f"       {j + 1}. {class_labels[idx.item()]}: {prob.item():.3f}")


def test_image_text_similarity(model, processor, images, text_descriptions, device):
    """Test CLIP for image-text similarity."""
    print("\n🔍 Testing Image-Text Similarity")
    print("-" * 35)

    # Custom text descriptions to test against images
    custom_texts = [
        "a beautiful landscape photo",
        "a corporate logo design",
        "an iconic landmark building",
        "a cute pet animal",
        "abstract art",
        "a city street scene",
    ]

    print(f"📝 Custom text descriptions: {len(custom_texts)}")

    for img_idx, (image, original_desc) in enumerate(zip(images, text_descriptions)):
        print(f"\n🖼️  Image {img_idx + 1}: {original_desc}")

        # Process single image with all text descriptions
        inputs = processor(text=custom_texts, images=[image], return_tensors="pt", padding=True)

        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
            )

            # Get similarity scores
            logits_per_image = outputs["logits_per_image"]  # Shape: (1, num_texts)
            similarities = torch.softmax(logits_per_image, dim=-1)[0]  # Shape: (num_texts,)

            # Sort by similarity
            sorted_sims, sorted_indices = torch.sort(similarities, descending=True)

            print(f"     Text similarity scores:")
            for i, (sim, idx) in enumerate(zip(sorted_sims[:3], sorted_indices[:3])):
                print(f"       {i + 1}. '{custom_texts[idx.item()]}': {sim.item():.3f}")


def test_text_embeddings(model, processor, device):
    """Test CLIP text embeddings."""
    print("\n🔍 Testing Text Embeddings")
    print("-" * 30)

    sample_texts = [
        "a photo of a dog",
        "a photo of a cat",
        "a dog playing in the park",
        "a cat sleeping on a sofa",
        "a red sports car",
        "a blue sedan car",
    ]

    print(f"📝 Sample texts: {len(sample_texts)}")

    # Process texts
    inputs = processor(text=sample_texts, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        print(f"\n🔄 Text embedding forward pass...")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        text_embeds = outputs["text_embeds"]  # Shape: (num_texts, embed_dim)

        print(f"📊 Text embeddings shape: {text_embeds.shape}")
        print(f"📊 Embedding dimension: {text_embeds.shape[1]}")

        # Compute similarity matrix between texts
        similarity_matrix = torch.cosine_similarity(
            text_embeds.unsqueeze(1), text_embeds.unsqueeze(0), dim=2
        )

        print(f"\n📊 Text Similarity Matrix:")
        print("     " + " ".join([f"{i:6d}" for i in range(len(sample_texts))]))
        for i, (text, similarities) in enumerate(zip(sample_texts, similarity_matrix)):
            sim_str = " ".join([f"{sim.item():6.3f}" for sim in similarities])
            print(f"{i:2d}: {sim_str}  | {text[:30]}...")


def main():
    """Test CLIP vision-language model with profiling."""
    print("🚀 CLIP Vision-Language Model Example")
    print("=" * 45)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Set up cache directory
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📂 Model cache directory: {os.path.abspath(cache_dir)}")

    try:
        # Create CLIP model (this will download weights)
        model = create_clip_model(
            model_name="openai/clip-vit-base-patch32", cache_dir=cache_dir
        ).to(device)

        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Get processor for later use
        processor = model.processor

        # Download sample images
        images, descriptions = download_sample_images()

        if not images:
            print("❌ Could not load any images, skipping visual tests")
            return

        # Test 1: Zero-shot image classification
        test_image_classification(model, processor, images, descriptions, device)

        # Test 2: Image-text similarity
        test_image_text_similarity(model, processor, images, descriptions, device)

        # Test 3: Text embeddings
        test_text_embeddings(model, processor, device)

        print("\n✅ CLIP vision-language model testing completed!")
        print("💡 Key features demonstrated:")
        print("   - Vision-language understanding with CLIP")
        print("   - Zero-shot image classification")
        print("   - Image-text similarity scoring")
        print("   - Text embedding generation")
        print("   - Real pretrained model profiling")

    except Exception as e:
        print(f"❌ Error with CLIP model: {e}")
        print("💡 Possible solutions:")
        print("   - Check internet connection for model download")
        print("   - Install required packages: pip install transformers pillow requests")
        print("   - Try running locally first to test setup")


if __name__ == "__main__":
    main()
