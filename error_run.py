import oneflow as torch
from PIL import Image
import numpy as np
import os
import sys
from oneflowrealesrgan import RealESRGAN


import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline

RealESRGAN_path = "weights/RealESRGANx2"

model_id = "CompVis/stable-diffusion-v1-4"
pipe = OneFlowStableDiffusionPipeline.from_pretrained(model_id)#OneFlow
pipe = pipe.to("cuda")
prompt = "a boy"
image=pipe(prompt, compile_unet = False).images[0]


device = torch.device('cuda')

model = RealESRGAN(device, scale=2)
model.load_weights(RealESRGAN_path)

#path_to_image = 'inputs/lr_image.png'
#image = Image.open(path_to_image).convert('RGB')
import datetime
t1=datetime.datetime.now()
sr_image = model.predict(image)
t2=datetime.datetime.now()
print((t2-t1).total_seconds(),sr_image.size)
