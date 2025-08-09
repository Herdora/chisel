#!/usr/bin/env python3
"""
Long-running demo script with frequent logging.
Duration: ~3 minutes with progress updates every 2-3 seconds.

This script demonstrates:
- Real-time stdout streaming with chisel
- Progress tracking for long-running jobs
- Model training simulation with detailed logging
- Command line arguments handling

Usage:
  python long_running_demo.py --epochs 5 --batch-size 32
  kandc python long_running_demo.py --epochs 10 --batch-size 64
  chisel --app-name "long-demo" --gpu 2 -- python long_running_demo.py --epochs 15 --verbose
"""

import argparse
import time
import random
from datetime import datetime
import torch
import torch.nn as nn
from kandc import capture_model_class


@capture_model_class(model_name="LongRunningModel", record_shapes=True, profile_memory=True)
class DemoModel(nn.Module):
    """A model that simulates realistic training time."""

    def __init__(self, input_size=512, hidden_size=256, num_layers=4, output_size=10):
        super().__init__()

        layers = []
        prev_size = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        param_count = sum(p.numel() for p in self.parameters())
        print(f"🏗️  Model initialized with {param_count:,} parameters")

    def forward(self, x):
        return self.network(x)


def print_progress_bar(iteration, total, length=40, fill="█", empty="░"):
    """Print a progress bar with percentage."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    return f"[{bar}] {percent}%"


def simulate_data_loading(batch_size, num_batches, verbose=False):
    """Simulate data loading with progress updates."""
    print("\n📊 Loading training data...")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {num_batches}")

    for i in range(num_batches):
        if verbose or i % max(1, num_batches // 10) == 0:
            progress = print_progress_bar(i + 1, num_batches)
            print(f"   Loading batch {i + 1:3d}/{num_batches}: {progress}")

        # Simulate loading time
        time.sleep(0.1 + random.uniform(0, 0.1))

    print(f"✅ Data loading complete! {num_batches} batches ready")


def run_training_epoch(model, epoch, total_epochs, batch_size, num_batches, device, verbose=False):
    """Run a single training epoch with detailed logging."""

    print(f"\n🚀 Epoch {epoch}/{total_epochs} - Starting training")
    print(f"   Device: {device}")
    print(f"   Timestamp: {datetime.now().strftime('%H:%M:%S')}")

    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx in range(num_batches):
        batch_start_time = time.time()

        # Generate synthetic batch
        x = torch.randn(batch_size, 512).to(device)
        y = torch.randint(0, 10, (batch_size,)).to(device)

        # Forward pass (this gets profiled by chisel)
        with torch.no_grad():
            outputs = model(x)

            # Simulate loss calculation
            loss = torch.nn.functional.cross_entropy(outputs, y)

            # Simulate accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).float().mean()

        batch_time = time.time() - batch_start_time
        epoch_loss += loss.item()
        epoch_acc += accuracy.item()

        # Log progress every few batches or if verbose
        if verbose or batch_idx % max(1, num_batches // 8) == 0 or batch_idx == num_batches - 1:
            progress = print_progress_bar(batch_idx + 1, num_batches)

            print(
                f"   Batch {batch_idx + 1:3d}/{num_batches}: {progress} | "
                f"Loss: {loss.item():.4f} | Acc: {accuracy.item():.3f} | "
                f"Time: {batch_time:.3f}s"
            )

        # Simulate realistic training time
        time.sleep(0.5 + random.uniform(0, 0.3))

    final_loss = epoch_loss / num_batches
    final_acc = epoch_acc / num_batches

    print(f"📈 Epoch {epoch} Results:")
    print(f"   Average Loss: {final_loss:.4f}")
    print(f"   Average Accuracy: {final_acc:.3f}")
    print(f"   Completed at: {datetime.now().strftime('%H:%M:%S')}")

    return final_loss, final_acc


def run_validation(model, device, batch_size=32):
    """Run validation with progress updates."""
    print("\n🔍 Running validation...")

    val_batches = 5
    val_loss = 0.0
    val_acc = 0.0

    for i in range(val_batches):
        # Generate validation batch
        x = torch.randn(batch_size, 512).to(device)
        y = torch.randint(0, 10, (batch_size,)).to(device)

        with torch.no_grad():
            outputs = model(x)
            loss = torch.nn.functional.cross_entropy(outputs, y)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).float().mean()

        val_loss += loss.item()
        val_acc += accuracy.item()

        progress = print_progress_bar(i + 1, val_batches)
        print(
            f"   Val batch {i + 1}/{val_batches}: {progress} | "
            f"Loss: {loss.item():.4f} | Acc: {accuracy.item():.3f}"
        )

        time.sleep(0.3)

    final_val_loss = val_loss / val_batches
    final_val_acc = val_acc / val_batches

    print("✅ Validation Results:")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    print(f"   Validation Accuracy: {final_val_acc:.3f}")

    return final_val_loss, final_val_acc


def main():
    """Main function with comprehensive logging."""
    parser = argparse.ArgumentParser(description="Long-running demo with frequent logging")

    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of training epochs (default: 8)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--num-batches", type=int, default=12, help="Batches per epoch (default: 12)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size (default: medium)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--validate", action="store_true", help="Run validation after each epoch")

    args = parser.parse_args()

    # Print startup banner
    print("=" * 70)
    print("🚀 CHISEL LONG-RUNNING DEMO")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚙️  Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Batches per epoch: {args.num_batches}")
    print(f"   Model size: {args.model_size}")
    print(f"   Verbose logging: {args.verbose}")
    print(f"   Validation: {args.validate}")

    # Estimate total time
    base_time = args.epochs * args.num_batches * 0.8
    val_time = args.epochs * 1.5 if args.validate else 0
    estimated_time = base_time + val_time
    print(
        f"⏱️  Estimated duration: {estimated_time:.1f} seconds ({estimated_time / 60:.1f} minutes)"
    )
    print("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎯 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Memory: {memory_gb:.1f} GB")

    # Model configuration based on size
    size_configs = {
        "small": {"hidden_size": 128, "num_layers": 2},
        "medium": {"hidden_size": 256, "num_layers": 4},
        "large": {"hidden_size": 512, "num_layers": 6},
    }
    config = size_configs[args.model_size]

    # Initialize model
    print(f"\n🏗️  Creating {args.model_size} model...")
    model = DemoModel(
        input_size=512,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=10,
    ).to(device)

    # Simulate data loading
    simulate_data_loading(args.batch_size, args.num_batches, args.verbose)

    # Training loop
    print("\n🎓 Starting training loop...")
    training_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_acc = run_training_epoch(
            model, epoch, args.epochs, args.batch_size, args.num_batches, device, args.verbose
        )

        training_history.append({"epoch": epoch, "loss": epoch_loss, "acc": epoch_acc})

        # Run validation if requested
        if args.validate:
            val_loss, val_acc = run_validation(model, device, args.batch_size)
            training_history[-1].update({"val_loss": val_loss, "val_acc": val_acc})

        # Progress summary
        progress_pct = (epoch / args.epochs) * 100
        print(f"\n📊 Overall Progress: {progress_pct:.1f}% complete ({epoch}/{args.epochs} epochs)")

        # Show trend if we have multiple epochs
        if len(training_history) >= 2:
            loss_trend = training_history[-1]["loss"] - training_history[-2]["loss"]
            acc_trend = training_history[-1]["acc"] - training_history[-2]["acc"]
            loss_arrow = "📈" if loss_trend > 0 else "📉"
            acc_arrow = "📈" if acc_trend > 0 else "📉"
            print(f"   Loss trend: {loss_arrow} {loss_trend:+.4f}")
            print(f"   Accuracy trend: {acc_arrow} {acc_trend:+.3f}")

        # Small pause between epochs
        if epoch < args.epochs:
            print(f"⏳ Preparing for epoch {epoch + 1}...")
            time.sleep(1.0)

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total epochs: {args.epochs}")
    print("📊 Training History:")

    for i, history in enumerate(training_history, 1):
        val_info = (
            f" | Val Loss: {history['val_loss']:.4f} | Val Acc: {history['val_acc']:.3f}"
            if "val_loss" in history
            else ""
        )
        print(f"   Epoch {i:2d}: Loss: {history['loss']:.4f} | Acc: {history['acc']:.3f}{val_info}")

    # Final model info
    final_loss = training_history[-1]["loss"]
    final_acc = training_history[-1]["acc"]
    print("\n🏆 Final Results:")
    print(f"   Best Training Loss: {min(h['loss'] for h in training_history):.4f}")
    print(f"   Best Training Accuracy: {max(h['acc'] for h in training_history):.3f}")
    print(f"   Final Training Loss: {final_loss:.4f}")
    print(f"   Final Training Accuracy: {final_acc:.3f}")

    if args.validate:
        best_val_loss = min(h["val_loss"] for h in training_history if "val_loss" in h)
        best_val_acc = max(h["val_acc"] for h in training_history if "val_acc" in h)
        print(f"   Best Validation Loss: {best_val_loss:.4f}")
        print(f"   Best Validation Accuracy: {best_val_acc:.3f}")

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n🔧 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {args.model_size}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Number of layers: {config['num_layers']}")

    print("\n✅ Demo completed successfully!")
    print("🚀 Thank you for using Keys & Caches!")
    print("=" * 70)


if __name__ == "__main__":
    main()
