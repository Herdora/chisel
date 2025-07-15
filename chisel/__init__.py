"""
Chisel - Profile AMD HIP kernels and PyTorch code on RunPod instances.

A library to enable seamless remote profiling of AMD GPU workloads with a local feel.
"""

__version__ = "0.1.0"

# Import core classes
from .pod import Pod, SshDetails, get_pods
from .pod_manager import PodManager, PodManagerError

__all__ = [
    "Pod",
    "SshDetails",
    "get_pods",
    "PodManager",
    "PodManagerError",
    "__version__",
]
