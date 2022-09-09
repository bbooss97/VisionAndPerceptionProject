from itertools import count
from dataset import ChessDataset
import torchvision.transforms as transforms
import torch
from PIL import ImageGrab
import numpy as np
from nn import BasicMlp
import torchvision
import random 
from pynput import mouse
from pynput import keyboard
count=0
points=[]
accept=False

def on_press(key):
    global accept
    if count==4:
        return False
    if key ==keyboard.Key.ctrl_l:
        accept=True
        print("select point")
listener = keyboard.Listener(on_press=on_press)
listener.start()

def on_click(x, y, button, pressed):
    global count,points,accept
    if pressed and button==mouse.Button.left and accept:
        points.append((x,y))
        count+=1
        accept=False
        print(count)
    if count==4:
        return False
listener = mouse.Listener(on_click=on_click)
listener.start()

x=input()
if count!=4:
    print("not enough points retry")
    quit()
print(points)



#script to use a model to get the pieces from the image
def getPatches(obs):
    unfold=torch.nn.Unfold(kernel_size=(50,50),stride=50)
    obs=unfold(obs)
    obs=obs.transpose(1,2)
    return obs
def readImage(points):
    screenshot = ImageGrab.grab()
    fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
    ])
    return fromPilToTensor(screenshot)


type="resnetFrom0"
model=torch.load("./weights/"+type+".pt",map_location=torch.device('cpu'))

fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])

# image_tensor=torchvision.io.read_image("./lichess.png")
# image_tensor=image_tensor[:-1,:,:]
image_tensor=readImage(points)

image_tensor=image_tensor/255
image=fromTensorToPil(image_tensor).resize((400,400))
image_tensor=fromPilToTensor(image).float()

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
for i in results:
    print(', '.join('{:2d}'.format(f) for f in i))
image.show()

