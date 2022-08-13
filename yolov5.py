import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import random
import sys
import torch.optim as optim
# sys.path.append("./yolov5/utils")
# sys.path.append("./yolov5")
# from MyLoss import ComputeLoss
# Model
trainedModels=["normalTask.pt","normalTaskAugmented.pt","modified0.5.pt","modified0.75.pt",'modified1']
# model = torch.hub.load('./yolov5/', 'custom', path='yoloaugmentedbest.pt', source='local') 
model = torch.hub.load('./yolov5/', 'custom', path=trainedModels[4], source='local') 
for params in model.parameters():
    params.requires_grad = True
# model.train()
# print(model)
# model.load_state_dict(torch.load("./best.pt"))
# model.test()
# print(model)
# Images
# img=Image.open('./trainAnnotated/1B1B1N2-1r6-n2R2k1-7b-1B6-8-8-Kn6.jpeg')
path=os.listdir("./testAnnotated")
path=path[random.randint(0,len(path))].replace(".txt",".jpeg")
print(path)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
img=Image.open('./testAnnotated/'+path)

fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])
randomPerspective = transforms.RandomPerspective(distortion_scale=0.75, p=1, interpolation=2)
# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = randomPerspective(img)


# print(model)
# Inference
# tensore=fromPilToTensor(img_tensor).unsqueeze(dim=0)
# print(tensore.shape)

results = model(img_tensor)
results.show()
# optimizer.zero_grad()
# r=results.pred[0].float()
# loss=(torch.ones_like(r)-r).sum()
# print(loss)
# loss.requires_grad=True
# loss.backward()
# optimizer.step()
# print(r.shape)
# print(type(model.model[-1]))
# print(model[-1])
# model.train()
# loss=ComputeLoss(model)

# Results

# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)