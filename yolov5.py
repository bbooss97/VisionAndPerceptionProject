import torch
from PIL import Image
import numpy as npù
import torchvision.transforms as transforms
import os
import random
# Model
model = torch.hub.load('./yolov5/', 'custom', path='yoloaugmentedbest.pt', source='local') 
# print(model)
# model.load_state_dict(torch.load("./best.pt"))
model.eval()
# print(model)
# Images
# img=Image.open('./trainAnnotated/1B1B1N2-1r6-n2R2k1-7b-1B6-8-8-Kn6.jpeg')
path=os.listdir("./testAnnotated")
path=path[random.randint(0,len(path))].replace(".txt",".jpeg")
print(path)
img=Image.open('./testAnnotated/'+path)

fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])
randomPerspective = transforms.RandomPerspective(distortion_scale=.0, p=1, interpolation=2)
# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = randomPerspective(img)


# Inference
results = model(img_tensor)

# Results
results.show()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)