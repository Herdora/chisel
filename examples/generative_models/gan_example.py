#!/usr/bin/env python3
"""
GAN model example using capture_model decorator.
This demonstrates generative model profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chisel import capture_model_class


@capture_model_class(model_name="Generator")
class Generator(nn.Module):
    """Simple generator for DCGAN-style architecture."""

    def __init__(self, latent_dim=100, img_channels=3, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Calculate the initial size after first deconv
        self.init_size = img_size // 8  # 8x8 for 64x64 images

        # Linear layer to start
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)

        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            # First deconv: 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Second deconv: 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Third deconv: 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final layer: output image
            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        batch_size = z.size(0)

        # Linear transformation
        out = self.fc(z)  # (batch_size, 512 * init_size * init_size)

        # Reshape to feature map
        out = out.view(batch_size, 512, self.init_size, self.init_size)

        # Deconvolutional layers
        img = self.deconv_layers(out)  # (batch_size, img_channels, img_size, img_size)

        return img


@capture_model_class(model_name="Discriminator")
class Discriminator(nn.Module):
    """Simple discriminator for DCGAN-style architecture."""

    def __init__(self, img_channels=3, img_size=64):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv: 64x64 -> 32x32
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Second conv: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Third conv: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Fourth conv: 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Final conv: 4x4 -> 1x1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # img shape: (batch_size, img_channels, img_size, img_size)
        out = self.conv_layers(img)  # (batch_size, 1, 1, 1)
        out = out.view(out.size(0), -1)  # (batch_size, 1)
        return out


@capture_model_class(model_name="VAE_Encoder")
class VAEEncoder(nn.Module):
    """Variational Autoencoder Encoder."""

    def __init__(self, img_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, img_channels, 64, 64)
        out = self.conv_layers(x)  # (batch_size, 512, 4, 4)
        out = out.view(out.size(0), -1)  # (batch_size, 512 * 4 * 4)

        mu = self.fc_mu(out)  # (batch_size, latent_dim)
        logvar = self.fc_logvar(out)  # (batch_size, latent_dim)

        return mu, logvar


@capture_model_class(model_name="VAE_Decoder")
class VAEDecoder(nn.Module):
    """Variational Autoencoder Decoder."""

    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Linear layer
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        out = self.fc(z)  # (batch_size, 512 * 4 * 4)
        out = out.view(out.size(0), 512, 4, 4)  # (batch_size, 512, 4, 4)
        img = self.deconv_layers(out)  # (batch_size, img_channels, 64, 64)
        return img


def sample_noise(batch_size, latent_dim, device="cpu"):
    """Sample random noise for generator input."""
    return torch.randn(batch_size, latent_dim, device=device)


def sample_images(batch_size, channels=3, height=64, width=64, device="cpu"):
    """Sample random images for testing."""
    return torch.randn(batch_size, channels, height, width, device=device)


def reparameterize(mu, logvar):
    """Reparameterization trick for VAE."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def main():
    """Test generative models with profiling."""
    print("🚀 Testing Generative Models")
    print("=" * 40)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    batch_size = 4
    latent_dim = 100
    img_channels = 3
    img_size = 64

    print(f"📊 Batch size: {batch_size}")
    print(f"📊 Latent dimension: {latent_dim}")
    print(f"📊 Image size: {img_channels}x{img_size}x{img_size}")

    # Test GAN models
    print("\n🔍 Testing GAN Models")
    print("-" * 25)

    # Generator
    generator = Generator(latent_dim, img_channels, img_size).to(device)
    print(f"📊 Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    # Discriminator
    discriminator = Discriminator(img_channels, img_size).to(device)
    print(f"📊 Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # Test generator
        for i in range(3):
            print(f"\n🔄 Generator forward pass {i + 1}...")
            noise = sample_noise(batch_size, latent_dim, device)
            fake_images = generator(noise)
            print(f"  Input noise shape: {noise.shape}")
            print(f"  Generated images shape: {fake_images.shape}")
            print(
                f"  Generated images range: [{fake_images.min().item():.4f}, {fake_images.max().item():.4f}]"
            )

            # Test discriminator with generated images
            print(f"🔄 Discriminator forward pass {i + 1}...")
            d_output = discriminator(fake_images)
            print(f"  Discriminator output shape: {d_output.shape}")
            print(f"  Discriminator predictions: {d_output.squeeze().tolist()}")

    # Test VAE models
    print("\n🔍 Testing VAE Models")
    print("-" * 25)

    # VAE Encoder and Decoder
    vae_encoder = VAEEncoder(img_channels, latent_dim=128).to(device)
    vae_decoder = VAEDecoder(latent_dim=128, img_channels=img_channels).to(device)

    print(f"📊 VAE Encoder parameters: {sum(p.numel() for p in vae_encoder.parameters()):,}")
    print(f"📊 VAE Decoder parameters: {sum(p.numel() for p in vae_decoder.parameters()):,}")

    vae_encoder.eval()
    vae_decoder.eval()

    with torch.no_grad():
        # Test VAE
        sample_imgs = sample_images(batch_size, img_channels, img_size, img_size, device)
        print(f"📊 Sample images shape: {sample_imgs.shape}")

        for i in range(3):
            print(f"\n🔄 VAE Encoder forward pass {i + 1}...")
            mu, logvar = vae_encoder(sample_imgs)
            print(f"  Mu shape: {mu.shape}")
            print(f"  Logvar shape: {logvar.shape}")
            print(f"  Mu range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
            print(f"  Logvar range: [{logvar.min().item():.4f}, {logvar.max().item():.4f}]")

            # Reparameterization
            z = reparameterize(mu, logvar)
            print(f"  Latent z shape: {z.shape}")

            print(f"🔄 VAE Decoder forward pass {i + 1}...")
            reconstructed = vae_decoder(z)
            print(f"  Reconstructed shape: {reconstructed.shape}")
            print(
                f"  Reconstructed range: [{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]"
            )

    print("\n✅ Generative models testing completed!")


if __name__ == "__main__":
    main()
