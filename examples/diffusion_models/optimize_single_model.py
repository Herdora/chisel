#!/usr/bin/env python3
"""
Optimize a single diffusion model for latency by sweeping inference params.

This simulates an engineer workflow focused on speed improvements for one model.

It loads Stable Diffusion v1.5 via diffusers, then sweeps across samplers and
step counts while keeping prompt, negative prompt, guidance, size, and seed
fixed so results are comparable. It records latency and writes a CSV summary,
plus saves generated images per setting.

Usage:
  python kandc/examples/diffusion_models/optimize_single_model.py \
    --prompt "A serene alpine lake at sunrise, ultra realistic, 35mm, high detail, golden hour" \
    --output-dir outputs/optimize_sd15

Optional flags:
  --model-id runwayml/stable-diffusion-v1-5
  --width 512 --height 512
  --guidance 7.5 --seed 12345

Outputs:
  - CSV: <output-dir>/summary.csv
  - Images under <output-dir>/<sampler>_steps<steps>/sample_000.png
"""

import argparse
import csv
import os
import time
from typing import List, Tuple

import torch
from PIL import Image


def build_scheduler(name: str, pipeline):
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        HeunDiscreteScheduler,
        DDPMScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
    )

    n = name.strip().lower()
    if n == "ddim":
        return DDIMScheduler.from_config(pipeline.scheduler.config)
    if n in {"dpm-solver", "dpmsolver", "dpm-solver++"}:
        return DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    if n == "euler":
        return EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    if n in {"euler-a", "euler_ancestral"}:
        return EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if n == "heun":
        return HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
    if n == "lms":
        return LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    if n == "pndm":
        return PNDMScheduler.from_config(pipeline.scheduler.config)
    if n in {"k-dpm2", "kdpm2"}:
        return KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config)
    if n in {"k-dpm2-a", "kdpm2-a", "karras"}:
        return KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    # default
    return DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


def load_pipe(model_id: str, device: torch.device, dtype: torch.dtype):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    return pipe.to(device)


def save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a single diffusion model for latency")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--prompt",
        default="A serene alpine lake at sunrise, ultra realistic, 35mm, high detail, golden hour",
    )
    parser.add_argument("--negative-prompt", default="blurry, low quality, deformed, watermark")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="outputs/optimize_sd15")
    parser.add_argument(
        "--samplers",
        nargs="*",
        default=["dpm-solver++", "euler-a", "euler", "heun", "lms", "pndm", "ddim"],
    )
    parser.add_argument("--steps", nargs="*", type=int, default=[10, 15, 20, 25, 30])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = load_pipe(args.model_id, device, dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    records: List[Tuple[str, int, float]] = []  # (sampler, steps, latency_ms)

    print("\n=== Optimize single model (latency) ===")
    print(f"device={device} dtype={dtype}")
    print(f"output: {args.output_dir}")

    for sampler in args.samplers:
        pipe.scheduler = build_scheduler(sampler, pipe)
        for steps in args.steps:
            t0 = time.perf_counter()
            with torch.no_grad():
                out = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance,
                    num_inference_steps=steps,
                    generator=generator,
                    width=args.width,
                    height=args.height,
                )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            img = out.images[0]
            out_dir = os.path.join(args.output_dir, f"{sampler}_steps{steps}")
            save_image(img, os.path.join(out_dir, "sample_000.png"))
            records.append((sampler, steps, dt_ms))
            print(f"{sampler:12s} steps={steps:2d} -> {dt_ms:8.1f} ms")

    # Write summary CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sampler", "steps", "latency_ms"])
        for sampler, steps, dt in records:
            w.writerow([sampler, steps, f"{dt:.3f}"])

    # Print best few
    best = sorted(records, key=lambda x: x[2])[:5]
    print("\nTop (fastest) configurations:")
    for sampler, steps, dt in best:
        print(f"- {sampler} steps={steps}: {dt:.1f} ms")
    print(f"\nSummary: {csv_path}")


if __name__ == "__main__":
    main()


