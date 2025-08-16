"""
Sweep functionality for kandc - run experiments across multiple configurations.
"""

import json
import os
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .init import init, finish, log, get_current_run, is_initialized
from .trace import capture_model_instance
from .timing import timed, timed_call


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""

    name: str
    model_size: str = "small"
    batch_size: int = 8
    warmup: int = 3
    repeats: int = 10
    tasks: List[str] = None
    model_params: Dict[str, Any] = None
    dataset_params: Dict[str, Any] = None
    optimization_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["mlp", "cnn", "matmul"]
        if self.model_params is None:
            self.model_params = {}
        if self.dataset_params is None:
            self.dataset_params = {}
        if self.optimization_params is None:
            self.optimization_params = {}


@dataclass
class SweepResult:
    """Results from a single sweep run."""

    config_name: str
    task: str
    batch_size: int
    latencies_ms: List[float]
    throughput_samples_per_sec: float
    memory_usage_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if len(self.latencies_ms) >= 20:
            return statistics.quantiles(self.latencies_ms, n=20)[18]
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        if len(self.latencies_ms) >= 100:
            return statistics.quantiles(self.latencies_ms, n=100)[98]
        return max(self.latencies_ms) if self.latencies_ms else 0.0


class SweepManager:
    """Manages running sweeps across multiple configurations."""

    def __init__(self, project_name: str = "sweep_experiment"):
        self.project_name = project_name
        self.results: List[SweepResult] = []
        self.logger = logging.getLogger(__name__)

        # Initialize kandc if not already done
        if not is_initialized():
            init(project=project_name)
            self._auto_finish = True
        else:
            self._auto_finish = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_finish:
            finish()

    def load_configs_from_folder(self, folder_path: Union[str, Path]) -> List[SweepConfig]:
        """Load all JSON config files from a folder."""
        folder = Path(folder_path)
        configs = []

        if not folder.exists():
            raise FileNotFoundError(f"Config folder not found: {folder}")

        for config_file in folder.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                # Extract name from filename or config
                name = config_data.get("name", config_file.stem)
                # Remove name from config_data to avoid duplicate argument
                config_data_copy = config_data.copy()
                config_data_copy.pop("name", None)
                config = SweepConfig(name=name, **config_data_copy)
                configs.append(config)
                self.logger.info(f"Loaded config: {name} from {config_file}")

            except Exception as e:
                self.logger.error(f"Failed to load config {config_file}: {e}")
                continue

        return configs

    def load_configs_from_files(self, config_files: List[Union[str, Path]]) -> List[SweepConfig]:
        """Load configs from specific files."""
        configs = []

        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                name = config_data.get("name", Path(config_file).stem)
                # Remove name from config_data to avoid duplicate argument
                config_data_copy = config_data.copy()
                config_data_copy.pop("name", None)
                config = SweepConfig(name=name, **config_data_copy)
                configs.append(config)
                self.logger.info(f"Loaded config: {name} from {config_file}")

            except Exception as e:
                self.logger.error(f"Failed to load config {config_file}: {e}")
                continue

        return configs

    def run_sweep(
        self,
        configs: List[SweepConfig],
        model_factory: Callable[[SweepConfig], Any],
        input_factory: Callable[[SweepConfig], Any],
        device: str = "auto",
    ) -> List[SweepResult]:
        """Run a sweep across multiple configurations."""

        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Starting sweep with {len(configs)} configs on {device}")

        for i, config in enumerate(configs):
            self.logger.info(f"Running config {i + 1}/{len(configs)}: {config.name}")

            try:
                # Create model and input for this config
                model = model_factory(config)
                inputs = input_factory(config)

                # Run tasks for this config
                for task in config.tasks:
                    result = self._run_single_task(config, task, model, inputs, device)
                    if result:
                        self.results.append(result)

                        # Log to kandc
                        log(
                            {
                                "config_name": config.name,
                                "task": task,
                                "batch_size": config.batch_size,
                                "avg_latency_ms": result.avg_latency_ms,
                                "p50_latency_ms": result.p50_latency_ms,
                                "p95_latency_ms": result.p95_latency_ms,
                                "p99_latency_ms": result.p99_latency_ms,
                                "throughput_samples_per_sec": result.throughput_samples_per_sec,
                                "memory_usage_mb": result.memory_usage_mb,
                                "gpu_utilization": result.gpu_utilization,
                                "repeats": len(result.latencies_ms),
                                "warmup": config.warmup,
                            }
                        )

            except Exception as e:
                self.logger.error(f"Failed to run config {config.name}: {e}")
                continue

        return self.results

    def _run_single_task(
        self, config: SweepConfig, task: str, model: Any, inputs: Any, device: str
    ) -> Optional[SweepResult]:
        """Run a single task for a configuration."""

        try:
            # Warmup runs
            for _ in range(config.warmup):
                if hasattr(model, "forward"):
                    model(*inputs) if isinstance(inputs, tuple) else model(inputs)
                else:
                    model(inputs)

            # Synchronize if using GPU
            if device == "cuda":
                import torch

                torch.cuda.synchronize()

            # Timed runs
            latencies = []
            for _ in range(config.repeats):
                start_time = time.perf_counter_ns()

                if hasattr(model, "forward"):
                    output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
                else:
                    output = model(inputs)

                if device == "cuda":
                    import torch

                    torch.cuda.synchronize()

                end_time = time.perf_counter_ns()
                latency_ms = (end_time - start_time) / 1e6
                latencies.append(latency_ms)

            # Calculate throughput
            avg_latency_s = statistics.mean(latencies) / 1000
            throughput = config.batch_size / avg_latency_s if avg_latency_s > 0 else 0

            # Get memory usage if available
            memory_usage = None
            if device == "cuda":
                import torch

                memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            return SweepResult(
                config_name=config.name,
                task=task,
                batch_size=config.batch_size,
                latencies_ms=latencies,
                throughput_samples_per_sec=throughput,
                memory_usage_mb=memory_usage,
            )

        except Exception as e:
            self.logger.error(f"Failed to run task {task} for config {config.name}: {e}")
            return None

    def save_results(self, output_path: Union[str, Path]):
        """Save sweep results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_data = [asdict(result) for result in self.results]

        with open(output_path, "w") as f:
            json.dump(
                {
                    "sweep_info": {
                        "project_name": self.project_name,
                        "total_configs": len(set(r.config_name for r in self.results)),
                        "total_tasks": len(self.results),
                        "timestamp": datetime.now().isoformat(),
                    },
                    "results": results_data,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved results to {output_path}")

    def print_summary(self):
        """Print a summary of sweep results."""
        if not self.results:
            print("No results to summarize")
            return

        print(f"\n{'=' * 60}")
        print(f"SWEEP SUMMARY - {self.project_name}")
        print(f"{'=' * 60}")
        print(f"Total configs: {len(set(r.config_name for r in self.results))}")
        print(f"Total tasks: {len(self.results)}")
        print(f"Results saved: {len(self.results)}")

        # Group by config
        configs = {}
        for result in self.results:
            if result.config_name not in configs:
                configs[result.config_name] = []
            configs[result.config_name].append(result)

        for config_name, config_results in configs.items():
            print(f"\n📊 Config: {config_name}")
            print(f"{'─' * 40}")

            for result in config_results:
                print(f"  {result.task.upper()}:")
                print(f"    Batch: {result.batch_size}")
                print(
                    f"    Latency: {result.avg_latency_ms:.2f}ms avg, {result.p95_latency_ms:.2f}ms p95"
                )
                print(f"    Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
                if result.memory_usage_mb:
                    print(f"    Memory: {result.memory_usage_mb:.1f} MB")


# Convenience functions for common use cases
def sweep_folder(
    folder_path: Union[str, Path],
    project_name: str = "sweep_experiment",
    model_factory: Callable[[SweepConfig], Any] = None,
    input_factory: Callable[[SweepConfig], Any] = None,
    device: str = "auto",
    output_path: Optional[Union[str, Path]] = None,
) -> List[SweepResult]:
    """Run a sweep on all config files in a folder."""

    with SweepManager(project_name) as manager:
        configs = manager.load_configs_from_folder(folder_path)

        if not configs:
            raise ValueError(f"No valid configs found in {folder_path}")

        if model_factory is None or input_factory is None:
            raise ValueError("model_factory and input_factory must be provided")

        results = manager.run_sweep(configs, model_factory, input_factory, device)

        if output_path:
            manager.save_results(output_path)

        manager.print_summary()
        return results


def sweep_files(
    config_files: List[Union[str, Path]],
    project_name: str = "sweep_experiment",
    model_factory: Callable[[SweepConfig], Any] = None,
    input_factory: Callable[[SweepConfig], Any] = None,
    device: str = "auto",
    output_path: Optional[Union[str, Path]] = None,
) -> List[SweepResult]:
    """Run a sweep on specific config files."""

    with SweepManager(project_name) as manager:
        configs = manager.load_configs_from_files(config_files)

        if not configs:
            raise ValueError("No valid configs found in provided files")

        if model_factory is None or input_factory is None:
            raise ValueError("model_factory and input_factory must be provided")

        results = manager.run_sweep(configs, model_factory, input_factory, device)

        if output_path:
            manager.save_results(output_path)

        manager.print_summary()
        return results
