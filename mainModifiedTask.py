import torchvision.transforms as transforms
import torch
import os
from PIL import ImageGrab
from PIL import ImageTransform
import numpy as np
from nn import BasicMlp
import torchvision
from pynput import mouse
from pynput import keyboard
import time
#script to use a model to get the pieces from the image given points
count=0
points=[]
accept=False

def on_press(key):
    global accept
    if count==4:
        return False
    if key==keyboard.Key.shift_l:
        accept=False
        print("deselected")
    if key ==keyboard.Key.ctrl_l:
        accept=True
        print("select point")

def on_click(x, y, button, pressed):
    global count,points,accept
    if pressed and button==mouse.Button.left and accept:
        points.append((x,y))
        count+=1
        accept=False
        print(count)
    if count==4:
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()
listener = mouse.Listener(on_click=on_click)
listener.start()

x=input()
if count!=4:
    print("not enough points retry")
    quit()
print(points)




def getPatches(obs):
    unfold=torch.nn.Unfold(kernel_size=(50,50),stride=50)
    obs=unfold(obs)
    obs=obs.transpose(1,2)
    return obs
def readImage(points):
    screenshot = ImageGrab.grab().convert("RGB")
    fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
    ])
    #from top left antiorary
    transform=[*points[0],*points[1],*points[2],*points[3]]
    # width=points[2][0]-points[0][0]
    # height=points[2][1]-points[0][1]
    screenshot = screenshot.transform((400,400), ImageTransform.QuadTransform(transform))
    # screenshot.show()
    return screenshot


trainedModels=["normalTask.pt","normalTaskAugmented.pt","modified0.5.pt","modified0.75.pt",'modified1.pt']
path="./weights/"+trainedModels[1]
model = torch.hub.load('./yolov5/', 'custom', path=path, source='local',force_reload=True) 

fromPilToTensor = transforms.Compose([
    transforms.PILToTensor()
])
fromTensorToPil=transforms.Compose([
    transforms.ToPILImage()
])


while True:
    os.system("cls")
    image=readImage(points)
    board=[[0 for i in range(8)]for j in range(8)]
    results=model(image)
    # results.show()
    xywh=results.xyxy[0]
    result=results.pred
    for i in range(xywh.size()[0]):
        centerx=int((xywh[i][0]+25)/50)
        centery=int((xywh[i][1]+25)/50)
        board[centery][centerx]=int(result[0][i][5])
    print("predicted")
    for i in board:
        print(', '.join('{:2d}'.format(f) for f in i))
    time.sleep(2)
# image.show()

