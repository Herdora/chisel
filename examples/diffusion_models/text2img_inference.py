#!/usr/bin/env python3
"""
Text-to-Image diffusion inference example using diffusers Stable Diffusion.

Usage:
  python diffusion_models/text2img_inference.py --config examples/diffusion_configs/config_01.json

Outputs are written to the specified output directory in the config.
Each config is designed to run on a single A100-80GB GPU.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # avoid importing tensorflow in transformers
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF logs if present

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from kandc import timed_call


def _build_scheduler(scheduler_name: str, pipeline):
    # Import schedulers lazily to avoid heavy imports when not needed
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

    name = (scheduler_name or "").strip().lower()
    if name in {"ddim"}:
        return DDIMScheduler.from_config(pipeline.scheduler.config)
    if name in {"ddpm"}:
        return DDPMScheduler.from_config(pipeline.scheduler.config)
    if name in {"dpm-solver", "dpmsolver", "dpmsolver++", "dpm-solver++"}:
        return DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    if name in {"euler"}:
        return EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    if name in {"euler-a", "euler_ancestral", "euler-ancestral"}:
        return EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    if name in {"heun"}:
        return HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
    if name in {"lms"}:
        return LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    if name in {"pndm"}:
        return PNDMScheduler.from_config(pipeline.scheduler.config)
    if name in {"k-dpm2", "kdpm2"}:
        return KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config)
    if name in {"k-dpm2-a", "kdpm2-a", "karras"}:
        return KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    # default to DPMSolver multistep
    return DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


def load_pipeline(model_id: str, device: torch.device, torch_dtype: torch.dtype):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    return pipe


def save_image(img: Image.Image, out_dir: str, stem: str, idx: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}_{idx:03d}.png")
    img.save(path)
    return path


def run_from_config(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    prompt = cfg.get("prompt", "A photo of a corgi in a field, ultra realistic, 35mm")
    negative_prompt = cfg.get("negative_prompt")
    guidance_scale = float(cfg.get("guidance_scale", 7.5))
    num_inference_steps = int(cfg.get("num_inference_steps", 30))
    scheduler_name = cfg.get("scheduler", "dpmsolver++")
    width = int(cfg.get("width", 512))
    height = int(cfg.get("height", 512))
    seed = int(cfg.get("seed", 42))
    num_images = int(cfg.get("num_images", 4))
    output_dir = cfg.get("output_dir", "outputs/diffusion")
    run_name = cfg.get("run_name", time.strftime("%Y%m%d-%H%M%S"))

    out_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("DIFFUSION INFERENCE START")
    print("=" * 70)
    print(f"device={device} dtype={dtype}")
    print(f"model_id={model_id}")
    print(f"prompt={prompt}")
    print(f"scheduler={scheduler_name} steps={num_inference_steps} guidance={guidance_scale}")
    print(f"size={width}x{height} seed={seed} num_images={num_images}")
    print(f"output_dir={out_dir}")

    pipe = load_pipeline(model_id, device, dtype)
    pipe.scheduler = _build_scheduler(scheduler_name, pipe)

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(seed)

    images = []

    for i in range(num_images):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = timed_call(
                "sd_infer",
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                width=width,
                height=height,
            )
        dt = time.perf_counter() - t0
        img = out.images[0]
        images.append(img)
        path = save_image(img, out_dir, stem="sample", idx=i)
        print(f"saved: {path} ({dt * 1000:.1f} ms)")

    # Save the resolved config used (locally)
    config_local_path = os.path.join(out_dir, "config_used.json")
    with open(config_local_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Save a simple manifest locally
    manifest = {
        "run_name": run_name,
        "model_id": model_id,
        "num_images": len(images),
        "images": sorted([fn for fn in os.listdir(out_dir) if fn.lower().endswith(".png")]),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Done.")
    print(f"🖼️  Wrote {len(images)} images to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Diffusion text2img inference")
    parser.add_argument("--config", required=True)
    args = parser.parse_known_args()[0]

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Ensure distinct output dirs per config by default
    if not cfg.get("run_name"):
        cfg["run_name"] = Path(args.config).stem

    run_from_config(cfg)


if __name__ == "__main__":
    main()
