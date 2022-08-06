import torch
from PIL import Image
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img=Image.open('./../trainAnnotated/1B1B1N2-1r6-n2R2k1-7b-1B6-8-8-Kn6.jpeg')

# Inference
results = model(img)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)