from dataclasses import dataclass
from enum import Enum
from typing import Dict


class GPUType(Enum):
    """Type of droplet."""

    AMD_MI300X = "amd-mi300x"
    NVIDIA_H100 = "nvidia-h100"
    NVIDIA_L40S = "nvidia-l40s"

    def __str__(self) -> str:
        return self.value


class GPURegion(Enum):
    """Region of the droplet."""

    NY1 = "nyc1"
    NY2 = "nyc2"
    ATL1 = "atl1"
    TOR1 = "tor1"

    def __str__(self) -> str:
        return self.value


@dataclass
class GPUProfile:
    """Configuration for different GPU types and cloud providers."""

    size: str
    image: str
    region: GPURegion
    gpu_type: GPUType


AMD_MI300X = GPUProfile(
    size="gpu-mi300x1-192gb",
    image="gpu-amd-base",
    region=GPURegion.ATL1,
    gpu_type=GPUType.AMD_MI300X,
)
NVIDIA_H100 = GPUProfile(
    size="gpu-h100x1-80gb",
    image="gpu-h100x1-base",
    region=GPURegion.NY2,
    gpu_type=GPUType.NVIDIA_H100,
)
NVIDIA_L40S = GPUProfile(
    size="gpu-l40sx1-48gb",
    image="gpu-h100x1-base",
    region=GPURegion.TOR1,
    gpu_type=GPUType.NVIDIA_L40S,
)

GPU_PROFILES: Dict[GPUType, GPUProfile] = {
    GPUType.AMD_MI300X: AMD_MI300X,
    GPUType.NVIDIA_H100: NVIDIA_H100,
    GPUType.NVIDIA_L40S: NVIDIA_L40S,
}


GPU_TYPE_TO_STRING = {
    GPUType.AMD_MI300X: "mi300x",
    GPUType.NVIDIA_H100: "h100",
    GPUType.NVIDIA_L40S: "l40s",
}


def get_gpu_type_from_command(command: str) -> GPUType:
    """Get the GPU type from the command."""
    if "mi300x" in command:
        return GPUType.AMD_MI300X
    elif "h100" in command:
        return GPUType.NVIDIA_H100
    elif "l40s" in command:
        return GPUType.NVIDIA_L40S
    else:
        raise ValueError(
            f"Could not determine GPU type from command: {command}"
        )


# TODO: figure out best way to name these.
