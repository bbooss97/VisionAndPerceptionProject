import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import random
import sys
import torch.optim as optim
#select type of models for the modified task
trainedModels=["normalTask.pt","normalTaskAugmented.pt","modified0.5.pt","modified0.75.pt",'modified1.pt']
howManyImages=10
distorsion=0.75
# model = torch.hub.load('./yolov5/', 'custom', path='yoloaugmentedbest.pt', source='local') 
path="./weights/"+trainedModels[3]
model = torch.hub.load('./yolov5/', 'custom', path=path, source='local',force_reload=True) 
fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])
for i in range(howManyImages):
    path=os.listdir("./testAnnotated")
    path=path[random.randint(0,len(path))].replace(".txt",".jpeg")
    print(path)
    img=Image.open('./testAnnotated/'+path)
    randomPerspective = transforms.RandomPerspective(distortion_scale=distorsion, p=1, interpolation=2)
    #modify image get results and show the image
    img_tensor = randomPerspective(img)
    results = model(img_tensor)
    print(results.xywh)
    results.show()
