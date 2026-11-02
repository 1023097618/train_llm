"""The open-source model that draws in the conversation. diffusers lives outside the training environment:
PYTHONPATH=~/train_llm_pylibs/image, installed with pip install --no-deps --target=... diffusers accelerate
"""
import re

import numpy as np
import torch

from common.data_paths import IMAGE_BACKEND, RAW_T2I


class ZImageBackend:
    def __init__(self, model_dir=str(IMAGE_BACKEND), device="cuda:0"):
        from diffusers import ZImageImg2ImgPipeline, ZImagePipeline
        self.device = device
        self.pipe = ZImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, local_files_only=True)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.img2img = ZImageImg2ImgPipeline(**self.pipe.components)
        self.img2img.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def edit(self, image, prompt, previous_prompt, seed=0, size=768, steps=9):
        """the picture (uint8 array) + its complete new description -> uint8 array. A small change keeps closer to the picture, a large one
        needs more freedom to happen at all (0.6 changed sunglasses but not a cat's face or the time of day, 0.8 did)."""
        from PIL import Image
        words = lambda text: set(re.findall(r"[a-z]+", text.lower()))
        kept = len(words(previous_prompt) & words(prompt)) / max(1, len(words(previous_prompt)))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.img2img(prompt=prompt, image=Image.fromarray(image), strength=0.6 if kept >= 0.9 else 0.8, height=size, width=size,
                           num_inference_steps=steps, guidance_scale=0.0, generator=generator).images[0]
        return np.asarray(out.convert("RGB"))

    @torch.inference_mode()
    def draw(self, prompt, seed=0, size=768, steps=9):
        """uint8 array (H, W, 3); the Turbo model is distilled to run without guidance"""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=0.0, generator=generator).images[0]
        return np.asarray(image.convert("RGB"))


class FluxBackend:
    """FLUX.1-schnell: 33 GB of weights, more than one 32 GB GPU holds, so the parts take turns on the GPU"""

    def __init__(self, model_dir=str(RAW_T2I / "flux-schnell"), device="cuda:0"):
        from diffusers import FluxPipeline
        self.device = device
        self.pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
        self.pipe.enable_model_cpu_offload(device=device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def draw(self, prompt, seed=0, size=768, steps=4):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=0.0,
                          max_sequence_length=256, generator=generator).images[0]
        return np.asarray(image.convert("RGB"))
