from dataset import ChessDataset
import torchvision.transforms as transforms
import torch
import numpy as np
from nn import BasicMlp
import torchvision
import random 
from encoding import Encoding
#script to use a model to get the pieces from the image
def getPatches(obs):
    unfold=torch.nn.Unfold(kernel_size=(50,50),stride=50)
    obs=unfold(obs)
    obs=obs.transpose(1,2)
    return obs

type="resnetFrom0"
model=torch.load("./weights/"+type+".pt",map_location=torch.device('cpu'))

fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])
dataset = ChessDataset(percentage=1)
#get a random image form the dataset
id=random.randint(0,len(dataset)-1)
image_tensor,label=dataset[id]
image=fromTensorToPil(image_tensor)
patches=getPatches(image_tensor.unsqueeze(0)).squeeze()
#apply the model to the image
typesToChange=["resnetFrom0","resnetPretrainedFineTuneFc","resnetPretrainedFineTuneAll","mobilenetPretrainedFineTuneAll"]
#transpose
if type in typesToChange:
    patches =patches.reshape(64,50,50,3)
    patches=torch.einsum("abcd->adbc",patches)
#get predictions
results=model(patches).argmax(1).reshape(8,8).tolist()
print("predicted")
encoding=Encoding()
for i in results:
    i=[encoding.reverseEncoding[j]for j in i]
    print(i)
print("true labels")
#get labels
label=label.argmax(1).reshape(8,8).tolist()
for i in label:
    i=[encoding.reverseEncoding[j]for j in i]
    print(i)
errors=0
#count the errors
for i in range(8):
    for j in range(8):
        if results[i][j]!=label[i][j]:
            errors+=1
print("errors")
print(errors)
image.show()
