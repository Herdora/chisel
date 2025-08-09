#!/usr/bin/env python3
"""
ResNet model example using capture_model decorator.
This demonstrates computer vision model profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kandc import capture_model_class


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@capture_model_class(model_name="ResNet18")
class ResNet18(nn.Module):
    """ResNet-18 implementation with profiling."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv layer
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global average pooling and classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


@capture_model_class(model_name="EfficientNet")
class SimpleEfficientNet(nn.Module):
    """Simplified EfficientNet-like model."""

    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # MBConv blocks (simplified)
        self.blocks = nn.Sequential(
            self._make_mbconv(32, 64, kernel_size=3, stride=1, expand_ratio=1),
            self._make_mbconv(64, 128, kernel_size=3, stride=2, expand_ratio=6),
            self._make_mbconv(128, 128, kernel_size=3, stride=1, expand_ratio=6),
            self._make_mbconv(128, 256, kernel_size=5, stride=2, expand_ratio=6),
            self._make_mbconv(256, 256, kernel_size=5, stride=1, expand_ratio=6),
            self._make_mbconv(256, 512, kernel_size=3, stride=2, expand_ratio=6),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def _make_mbconv(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """Create a MobileNetV2-style inverted residual block."""
        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )

        # Depthwise
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ]
        )

        # Pointwise
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def create_sample_images(batch_size=8, channels=3, height=224, width=224, device="cpu"):
    """Create sample image data for testing."""
    return torch.randn(batch_size, channels, height, width, device=device)


def main():
    """Test vision models with profiling."""
    print("🚀 Testing Vision Models")
    print("=" * 40)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Create sample data (ImageNet-like)
    batch_size = 4
    sample_images = create_sample_images(batch_size, 3, 224, 224, device)

    print(f"📊 Sample images shape: {sample_images.shape}")
    print(
        f"📊 Image data range: [{sample_images.min().item():.4f}, {sample_images.max().item():.4f}]"
    )

    # Test ResNet-18
    print("\n🔍 Testing ResNet-18")
    print("-" * 30)

    resnet_model = ResNet18(num_classes=1000).to(device)
    print(f"📊 ResNet-18 parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")

    resnet_model.eval()
    with torch.no_grad():
        for i in range(3):
            print(f"🔄 ResNet forward pass {i + 1}...")
            output = resnet_model(sample_images)
            print(f"  Output shape: {output.shape}")
            print(f"  Logits range: [{output.min().item():.4f}, {output.max().item():.4f}]")

            # Get top-5 predictions
            _, top5_indices = torch.topk(output[0], 5)
            print(f"  Top-5 class indices for first image: {top5_indices.tolist()}")

    # Test Simplified EfficientNet
    print("\n🔍 Testing Simplified EfficientNet")
    print("-" * 35)

    efficientnet_model = SimpleEfficientNet(num_classes=1000).to(device)
    print(
        f"📊 EfficientNet parameters: {sum(p.numel() for p in efficientnet_model.parameters()):,}"
    )

    efficientnet_model.eval()
    with torch.no_grad():
        for i in range(3):
            print(f"🔄 EfficientNet forward pass {i + 1}...")
            output = efficientnet_model(sample_images)
            print(f"  Output shape: {output.shape}")
            print(f"  Logits range: [{output.min().item():.4f}, {output.max().item():.4f}]")

            # Get top-5 predictions
            _, top5_indices = torch.topk(output[0], 5)
            print(f"  Top-5 class indices for first image: {top5_indices.tolist()}")

    # Test different input sizes
    print("\n🔍 Testing Different Input Sizes")
    print("-" * 35)

    test_sizes = [(1, 3, 32, 32), (2, 3, 128, 128), (1, 3, 512, 512)]

    for batch_size, channels, height, width in test_sizes:
        print(f"📊 Testing size: {(batch_size, channels, height, width)}")
        test_input = torch.randn(batch_size, channels, height, width, device=device)

        try:
            with torch.no_grad():
                output = resnet_model(test_input)
                print(f"  ✅ ResNet output: {output.shape}")
        except Exception as e:
            print(f"  ❌ ResNet failed: {e}")

        try:
            with torch.no_grad():
                output = efficientnet_model(test_input)
                print(f"  ✅ EfficientNet output: {output.shape}")
        except Exception as e:
            print(f"  ❌ EfficientNet failed: {e}")

    print("\n✅ Vision models testing completed!")


if __name__ == "__main__":
    main()
