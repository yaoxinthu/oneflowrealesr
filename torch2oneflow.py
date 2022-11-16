import oneflow as flow
import oneflow.nn as nn
import torch
from oneflowrrdbnet_arch import RRDBNet
from realesrgan import RealESRGAN
scale = 2
model_flow = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
model = RealESRGAN("cuda:1", scale=scale)
model.load_weights('weights/RealESRGAN_x%d.pth'%scale)
parameters = model.model.state_dict()

for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val

model_flow.load_state_dict(parameters)


flow.save(model_flow,"oneflowweights/RealESRGAN_x%d"%scale)