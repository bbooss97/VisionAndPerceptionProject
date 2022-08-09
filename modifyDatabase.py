from dataset import ChessDataset
import os
from encoding import Encoding
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from labeler import Labeler
class Modifier:
    def __init__(self,path,label=False,distorsion_scale=.5):
        self.path=path
        self.label=label
        self.distorsion_scale=distorsion_scale
        self.labeler=Labeler(self.path)
        self.titles=self.labeler.trainTitles
        self.fromPilToTensor = transforms.Compose([
            transforms.PILToTensor()
        ])
        self.fromTensorToPil=transforms.Compose([
            transforms.ToPILImage()
        ])
        self.randomPerspective = transforms.RandomPerspective(distortion_scale=self.distorsion_scale, p=1, interpolation=2)
        # img_tensor = randomPerspective(img)
        
    def modify(self):
        titles=self.titles
        for title in titles:
            image=self.getTensorImage(title)
            positions=self.labeler.positionsFromTitle(title)
            for i in range(8):
                for j in range(8):
                    if positions[i][j]!=0:
                        element=[positions[i][j],(j*50+25)/400,(i*50+25)/400,50/400,50/400]
                        patch=image[:,50*i:50*i+50,50*j:50*j+50]
                        patch=self.randomPerspective(patch)
                        # self.fromTensorToPil(patch).show()
                        image[:,50*i:50*i+50,50*j:50*j+50]=patch
            self.fromTensorToPil(image).save(self.path+"/"+title)

        if self.label:
            self.labeler.label()

    def getTensorImage(self,title):
        img=Image.open(self.path+"/"+title)
        return self.fromPilToTensor(img)
    
modifier=Modifier("./trainModified0.5",label=False,distorsion_scale=0.5)
modifier.modify()
# model = torch.hub.load('./yolov5/', 'custom', path='best.pt', source='local') 
# model.eval()
# img=Image.open('./testAnnotated/1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg')

# fromPilToTensor = transforms.Compose([
#     transforms.PILToTensor()
# ])
# fromTensorToPil=transforms.Compose([
#     transforms.ToPILImage()
# ])
# randomPerspective = transforms.RandomPerspective(distortion_scale=.5, p=1, interpolation=3)
# img_tensor = randomPerspective(img)
# results = model(img_tensor)
# results.show()