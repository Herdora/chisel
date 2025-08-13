#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from kandc import capture_model_instance, timed, timed_call


def _time_it(fn, warmup: int, repeats: int, event_name: str | None = None) -> List[float]:
    # Warmup
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    latencies_ms: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter_ns()
        if event_name:
            timed_call(event_name, fn)
        else:
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)
    return latencies_ms


def _print_stats(name: str, latencies_ms: List[float], batch_size: int) -> None:
    avg = sum(latencies_ms) / len(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p95 = (
        statistics.quantiles(latencies_ms, n=20)[18]
        if len(latencies_ms) >= 20
        else max(latencies_ms)
    )
    throughput = (batch_size / (avg / 1e3)) if avg > 0 else 0.0
    print(f"\n[name] {name}")
    print(f"  repeats={len(latencies_ms)}, batch={batch_size}")
    print(
        f"  avg={avg:.3f} ms  p50={p50:.3f} ms  p95={p95:.3f} ms  throughput~{throughput:.2f} samples/s"
    )


def _run_mlp(
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> None:
    model = (
        nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        .to(device)
        .eval()
    )
    # Capture only this singular nn.Module in the sweep
    model = capture_model_instance(
        model, model_name="SweepMLP", record_shapes=True, profile_memory=True
    )
    x = torch.randn(batch_size, input_dim, device=device)
    with torch.no_grad():
        lat = _time_it(lambda: model(x), warmup, repeats, event_name="MLP_infer")
    _print_stats("MLP", lat, batch_size)


def _run_cnn(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> None:
    model = (
        nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 100),
        )
        .to(device)
        .eval()
    )
    x = torch.randn(batch_size, channels, height, width, device=device)
    with torch.no_grad():
        lat = _time_it(lambda: model(x), warmup, repeats, event_name="CNN_infer")
    _print_stats("CNN", lat, batch_size)


def _run_matmul(
    batch_size: int, m: int, k: int, n: int, device: torch.device, warmup: int, repeats: int
) -> None:
    a = torch.randn(batch_size, m, k, device=device)
    b = torch.randn(batch_size, k, n, device=device)
    with torch.no_grad():
        lat = _time_it(lambda: torch.matmul(a, b), warmup, repeats, event_name="BatchedMatmul")
    _print_stats("BatchedMatmul", lat, batch_size)


def sizes_for_model(model_size: str) -> Tuple[int, int, int, int]:
    # Returns (mlp_hidden, image_size, matmul_m, matmul_k_n)
    size = (model_size or "small").lower()
    if size == "large":
        return 4096, 384, 2048, 2048
    if size == "medium":
        return 2048, 256, 1024, 1024
    return 1024, 224, 512, 512


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep inference timing")
    parser.add_argument("--config", required=True)
    args, extra = parser.parse_known_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_size = cfg.get("model_size", "small")
    batch_size = int(cfg.get("batch_size", 8))
    warmup = int(cfg.get("warmup", 3))
    repeats = int(cfg.get("repeats", 10))
    tasks: List[str] = cfg.get("tasks", ["mlp", "cnn", "matmul"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clear, prominent banner with script and config names
    try:
        script_name = __file__
    except NameError:
        script_name = "<unknown>"
    print("\n" + "=" * 70)
    print("SWEEP INFERENCE START")
    print("=" * 70)
    print(f"script: {script_name}")
    print(f"config: {args.config}")
    print(
        f"device={device} model_size={model_size} batch={batch_size} warmup={warmup} repeats={repeats}"
    )
    print(f"[sweep_inference] tasks={tasks}")

    mlp_hidden, image_size, mm_dim, mm_kn = sizes_for_model(model_size)

    for task in tasks:
        t = task.lower()
        if t == "mlp":
            _run_mlp(
                batch_size=batch_size,
                input_dim=mlp_hidden,
                hidden_dim=mlp_hidden,
                output_dim=100,
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
        elif t == "cnn":
            _run_cnn(
                batch_size=batch_size,
                channels=3,
                height=image_size,
                width=image_size,
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
        elif t == "matmul":
            _run_matmul(
                batch_size=batch_size,
                m=mm_dim,
                k=mm_kn,
                n=mm_kn,
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
        else:
            print(f"[sweep_inference] unknown task '{task}', skipping")

    print("\n[sweep_inference] complete")


if __name__ == "__main__":
    main()
