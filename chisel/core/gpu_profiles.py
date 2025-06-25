from dataclasses import dataclass


@dataclass
class GPUProfile:
    """Configuration for different GPU types and cloud providers."""

    size: str
    image: str
    region: str


AMD_MI300X = GPUProfile(size="gpu-mi300x1-192gb", image="gpu-amd-base", region="atl1")
NVIDIA_H100 = GPUProfile(size="gpu-h100x1-80gb", image="gpu-h100x1-base", region="nyc2")
NVIDIA_L40S = GPUProfile(size="gpu-l40sx1-48gb", image="gpu-h100x1-base", region="tor1")

GPU_PROFILES = {
    "amd-mi300x": AMD_MI300X,
    "nvidia-h100": NVIDIA_H100,
    "nvidia-l40s": NVIDIA_L40S,
}
