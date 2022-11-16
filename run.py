import oneflow as torch
from PIL import Image
import numpy as np
from oneflowrealesrgan import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=2)
model.load_weights('/home/gzyaoxin/model/RealESRGAN/x2')

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')
import datetime
t1=datetime.datetime.now()
sr_image = model.predict(image)
t2=datetime.datetime.now()
print((t2-t1).total_seconds(),sr_image.size)

path_to_image = 'inputs/lr_face.png'
image = Image.open(path_to_image).convert('RGB')
t1=datetime.datetime.now()
sr_image = model.predict(image)
t2=datetime.datetime.now()
print((t2-t1).total_seconds(),sr_image.size)

path_to_image = 'inputs/lr_lion.png'
image = Image.open(path_to_image).convert('RGB')
t1=datetime.datetime.now()
sr_image = model.predict(image)
t2=datetime.datetime.now()
print((t2-t1).total_seconds(),sr_image.size)

sr_image.save('results/sr_image.png')