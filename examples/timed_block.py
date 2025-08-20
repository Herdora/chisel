# pip install diffusers transformers accelerate torch kandc
import time
from contextlib import contextmanager

from kandc.annotators.timing import timed_call
import torch
from diffusers import StableDiffusionPipeline
import kandc



run = kandc.init(project="inference-examples", name="diffusers")

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

prompt = "A cozy cabin in the woods at dusk"
W, H, steps = 768, 768, 30
        
result = timed_call("pipe", pipe, prompt, num_inference_steps=steps, height=H, width=W)
img = result.images[0]

kandc.log("image_done", size=f"{W}x{H}")
kandc.finish()
