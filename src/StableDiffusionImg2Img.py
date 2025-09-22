import os
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
import torchvision.transforms as T
import torchvision


class DummySafetyChecker:
    @staticmethod
    def __call__(images, clip_input):
        return images, False


class StableDiffusionImg2Img():
    def __init__(self):
        model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"

        pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
        )
        pipe_img2img = pipe_img2img.to("cuda")
        pipe_img2img.safety_checker = DummySafetyChecker()


        self.pipe_img2img = pipe_img2img
        self.SEED = 42
        self.STRENGTH = 0.5
        self.GUIDANCE = 7.5
        self.NUM_STEPS = 50

    def manipulate(self, images, prompt):
        b = images.shape[0]
        man_imgs = self.pipe_img2img(prompt=prompt, image=images, strength=self.STRENGTH, guidance_scale=self.GUIDANCE, num_inference_steps=self.NUM_STEPS, num_images_per_prompt=b).images

        img_tensor = []
        for img in man_imgs: img_tensor.append(T.functional.to_tensor(img))
        return torch.stack(img_tensor, dim=0).to(images.device)
    



        
