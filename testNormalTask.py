from dataset import ChessDataset
import torchvision.transforms as transforms
import torch
import numpy as np
from nn import BasicMlp
import torchvision
import random 
def getPatches(obs):
    unfold=torch.nn.Unfold(kernel_size=(50,50),stride=50)
    obs=unfold(obs)
    obs=obs.transpose(1,2)
    return obs

type="mlp"

model=torch.load("./models/"+type+".pt",map_location=torch.device('cpu'))

fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])
dataset = ChessDataset(percentage=1)
id=random.randint(0,len(dataset)-1)
image_tensor,label=dataset[id]
image=fromTensorToPil(image_tensor)

patches=getPatches(image_tensor.unsqueeze(0)).squeeze()
results=model(patches).argmax(1).reshape(8,8).tolist()
print("predicted")
for i in results:
    print(i)
print("true labels")
label=label.argmax(1).reshape(8,8).tolist()
for i in label:
    print(i)
errors=0
for i in range(8):
    for j in range(8):
        if results[i][j]!=label[i][j]:
            errors+=1
print("errors")
print(errors)
image.show()
