"""The open-source model that draws in the conversation. diffusers lives outside the training environment:
PYTHONPATH=~/train_llm_pylibs/image, installed with pip install --no-deps --target=... diffusers accelerate
"""
import re

import numpy as np
import torch

from common.data_paths import IMAGE_BACKEND, VLM_ENCODER

EDIT_STRENGTHS = [0.65, 0.75, 0.85, 0.95]


class ZImageBackend:
    def __init__(self, model_dir=str(IMAGE_BACKEND), device="cuda:0"):
        from diffusers import ZImageImg2ImgPipeline, ZImagePipeline
        self.device = device
        self.pipe = ZImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, local_files_only=True)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.img2img = ZImageImg2ImgPipeline(**self.pipe.components)
        self.img2img.set_progress_bar_config(disable=True)
        self.judge = None

    @torch.inference_mode()
    def looks_like(self, image, prompt, previous_prompt):
        """does the picture match the new description better than the old one? SigLIP2 compares them"""
        if self.judge is None:
            from transformers import AutoModel, AutoProcessor
            self.judge = AutoModel.from_pretrained(str(VLM_ENCODER), dtype=torch.bfloat16).to(self.device).eval()
            self.judge_inputs = AutoProcessor.from_pretrained(str(VLM_ENCODER))
        inputs = self.judge_inputs(text=[prompt, previous_prompt], images=[image], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        scores = self.judge(**inputs).logits_per_image[0].float()
        return scores[0] - scores[1] > 1.0  # a whole car still red except a green streak scores +0.25, changes that worked from +1.6 up

    @torch.inference_mode()
    def edit(self, image, prompt, previous_prompt, seed=0, size=768, steps=9, check=True):
        """the picture (uint8 array) + its complete new description -> uint8 array. No single strength fits: a small object changes color at
        0.65 while a whole car needs 0.85, where the composition starts to move. So it starts low and goes up until the result looks like the
        new description more than the old one; a change of several words starts higher, because at 0.65 a cat stays as cute as it was."""
        from PIL import Image
        words = lambda text: set(re.findall(r"[a-z]+", text.lower()))
        ladder = EDIT_STRENGTHS if len(words(prompt) - words(previous_prompt)) <= 1 else EDIT_STRENGTHS[1:]
        if not check:
            ladder = ladder[-2:-1]
        for strength in ladder:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            out = self.img2img(prompt=prompt, image=Image.fromarray(image), strength=strength, height=size, width=size,
                               num_inference_steps=steps, guidance_scale=0.0, generator=generator).images[0].convert("RGB")
            if check and self.looks_like(out, prompt, previous_prompt):
                break
        self.last_strength = strength
        return np.asarray(out)

    @torch.inference_mode()
    def draw(self, prompt, seed=0, size=768, steps=9):
        """uint8 array (H, W, 3); the Turbo model is distilled to run without guidance"""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=0.0, generator=generator).images[0]
        return np.asarray(image.convert("RGB"))

