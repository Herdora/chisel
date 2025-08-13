__version__ = "0.1.0"

from .constants import GPUType
from .timing import timed, timed_call

# Import capture_trace, capture_model_instance, and capture_model_class from trace module
try:
    from .trace import capture_trace, capture_model_instance, capture_model_class, parse_model_trace
except ImportError:
    # Fallback for backward compatibility - create a pass-through decorator
    def capture_trace(trace_name=None, record_shapes=False, profile_memory=False, **kwargs):
        """Fallback capture_trace decorator for backward compatibility."""

        def decorator(fn):
            return fn

        return decorator

    def capture_model_instance(
        model_instance, model_name=None, record_shapes=True, profile_memory=True, **kwargs
    ):
        """Fallback capture_model_instance function for backward compatibility."""
        return model_instance

    def capture_model_class(model_name=None, record_shapes=True, profile_memory=True, **kwargs):
        """Fallback capture_model_class decorator for backward compatibility."""

        def decorator(model):
            return model

        return decorator

    def parse_model_trace(trace_file, model_name="Unknown"):
        """Fallback parse_model_trace function for backward compatibility."""
        return None


# Also provide it as a function for backward compatibility
def capture_trace_fallback(trace_name=None, record_shapes=False, profile_memory=False, **kwargs):
    """Fallback capture_trace function for backward compatibility."""

    def decorator(fn):
        return fn

    return decorator


__all__ = [
    "capture_trace",
    "capture_model_instance",
    "capture_model_class",
    "parse_model_trace",
    "capture_trace_fallback",
    "timed",
    "timed_call",
    "GPUType",
    "__version__",
]


def main():
    """Main CLI entry point."""
    from .cli import main as cli_main

    return cli_main()
