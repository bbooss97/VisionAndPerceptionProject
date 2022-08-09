from dataset import ChessDataset
import os
from encoding import Encoding
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from labeler import Labeler
from torch import Tensor

class PerspectiveTransorm(transforms.RandomPerspective):
    def __init__(self,distortion_scale=.5,p=1,interpolation=2,fill=0):
        super().__init__(distortion_scale=distortion_scale,p=p,interpolation=transforms.functional.InterpolationMode.BILINEAR,fill=fill)
        self.channel=3
        self.width=400
        self.height=400
        self.initializeParameters()
        
    def initializeParameters(self):
        self.starpoint,self.endpoint=self.get_params(self.width,self.height,self.distortion_scale)
    def forward(self,img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * self.channel
            else:
                fill = [float(f) for f in fill]

        if torch.rand(1) < self.p:
            return transforms.functional.perspective(img, self.starpoint, self.endpoint, self.interpolation, fill)
        return img
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
        self.randomPerspective = PerspectiveTransorm(distortion_scale=self.distorsion_scale, p=1, interpolation=2)
    
    def yoloModification(self):
        titles=self.titles
        for title in titles:
            image=self.getTensorImage(title)
            positions=self.labeler.positionsFromTitle(title)
            titolo=title.replace(".jpeg",".txt")
            f=open(self.path+"/"+titolo,"w")
            perspectiveImage=self.randomPerspective(image)
            self.fromTensorToPil(perspectiveImage).save(self.path+"/"+title)
            # perspectiveImage=self.fromTensorToPil(perspectiveImage).show()
            count=0
            pieces={}
            imageForPositions=torch.zeros(3,400,400)
            for i in range(8):
                for j in range(8):
                    if positions[i][j]!=0:
                        count+=1
                        pieces[count] = positions[i][j]
                        imageForPositions[:,i*50:(i+1)*50,j*50:(j+1)*50]=torch.ones(3,50,50)*count
            imageForPositions=self.randomPerspective(imageForPositions) 
            # self.fromTensorToPil(imageForPositions).show()
            for i in range(1,count+1):
                b=imageForPositions==i
                b=b.nonzero(as_tuple=True)
                rows=b[1].float()
                cols=b[2].float()
                if rows.shape[0]==0 or cols.shape[0]==0:
                    continue
                iCenterRow=int(rows.mean().item())/400
                iCenterCol=int(cols.mean().item())/400
                iwidth=int(cols.max().item()-cols.min().item())/400
                iheight=int(rows.max().item()-rows.min().item())/400
                s=""+str(pieces[i])+" "+str(iCenterCol)+" "+str(iCenterRow)+" "+str(iwidth)+" "+str(iheight)+"\n"
                f.write(s)
            self.randomPerspective.initializeParameters()
            f.close()


    def getTensorImage(self,title):
        img=Image.open(self.path+"/"+title)
        return self.fromPilToTensor(img)
    
modifier=Modifier("./trainModifiedYolo1",label=False,distorsion_scale=1)
modifier.yoloModification()