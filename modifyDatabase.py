from dataset import ChessDataset
import os
from encoding import Encoding
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from labeler import Labeler
#in this class i apply at the database the perspective transformation with 
#a distortion scale for every pieces but not for the entire image

class Modifier:
    def __init__(self,path,label=False,distorsion_scale=.5):
        self.path=path
        self.label=label
        self.distorsion_scale=distorsion_scale
        self.labeler=Labeler(self.path)
        #get the titles
        self.titles=self.labeler.trainTitles
        #craete the transformations used to apply the perspective transformation from the pytorch library
        self.fromPilToTensor = transforms.Compose([
            transforms.PILToTensor()
        ])
        self.fromTensorToPil=transforms.Compose([
            transforms.ToPILImage()
        ])
        self.randomPerspective = transforms.RandomPerspective(distortion_scale=self.distorsion_scale, p=1, interpolation=2)
        # img_tensor = randomPerspective(img)
    
    #this function modify the image pieces with the transformationapplied and return it
    #the labels in this way are the same
    def modify(self):
        titles=self.titles
        for title in titles:
            image=self.getTensorImage(title)
            positions=self.labeler.positionsFromTitle(title)
            for i in range(8):
                for j in range(8):
                    if positions[i][j]!=0:
                        #get the elements in the patch if there is a piece and modify the patch with the new piece
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
