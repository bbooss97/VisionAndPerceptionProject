import torch
from PIL import Image
import numpy as np
# Model
model = torch.hub.load('./yolov5/', 'custom', path='best.pt', source='local') 
# print(model)
# model.load_state_dict(torch.load("./best.pt"))
model.eval()
# print(model)
# Images
# img=Image.open('./trainAnnotated/1B1B1N2-1r6-n2R2k1-7b-1B6-8-8-Kn6.jpeg')
img=Image.open('./testAnnotated/1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg')

# Inference
results = model(img)

# Results
results.show()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)