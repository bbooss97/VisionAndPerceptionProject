from dataset import ChessDataset
import os
from encoding import Encoding
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from labeler import Labeler
from torch import Tensor
#this function modifies the perspective transform of the pytorchvision package
#it saves the transformation applied so that it can be reused more than one time one for the image and one 
#to get the temporary image used to label
class PerspectiveTransorm(transforms.RandomPerspective):
    def __init__(self,distortion_scale=.5,p=1,interpolation=2,fill=0):
        super().__init__(distortion_scale=distortion_scale,p=p,interpolation=transforms.functional.InterpolationMode.BILINEAR,fill=fill)
        #initialize the parameters of the class
        self.channel=3
        self.width=400
        self.height=400
        self.initializeParameters()
    #get the parameters from the distorsion
    def initializeParameters(self):
        self.starpoint,self.endpoint=self.get_params(self.width,self.height,self.distortion_scale)
    #this is the fucntion to apply the perspective transformation
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
#this class is used to create the new database for the yolo network
#applying the perspective transformation and get the right corresponding output format
class Modifier:
    def __init__(self,path,label=False,distorsion_scale=.5):
        self.path=path
        self.label=label
        self.distorsion_scale=distorsion_scale
        self.labeler=Labeler(self.path)
        self.titles=self.labeler.trainTitles
        #transformations
        self.fromPilToTensor = transforms.Compose([
            transforms.PILToTensor()
        ])
        self.fromTensorToPil=transforms.Compose([
            transforms.ToPILImage()
        ])
        self.randomPerspective = PerspectiveTransorm(distortion_scale=self.distorsion_scale, p=1, interpolation=2)
    #function that modifies image and get the yolo labels accordingly
    def yoloModification(self):
        titles=self.titles
        for title in titles:
            image=self.getTensorImage(title)
            positions=self.labeler.positionsFromTitle(title)
            titolo=title.replace(".jpeg",".txt")
            f=open(self.path+"/"+titolo,"w")
            #define the perspective transformation
            perspectiveImage=self.randomPerspective(image)
            #save the transformed image 
            self.fromTensorToPil(perspectiveImage).save(self.path+"/"+title)
            # perspectiveImage=self.fromTensorToPil(perspectiveImage).show()
            count=0
            pieces={}
            #create an image with same numbers for every piece box 
            imageForPositions=torch.zeros(3,400,400)
            for i in range(8):
                for j in range(8):
                    if positions[i][j]!=0:
                        count+=1
                        pieces[count] = positions[i][j]
                        imageForPositions[:,i*50:(i+1)*50,j*50:(j+1)*50]=torch.ones(3,50,50)*count
            #apply the perspective transformation to the image with the numbers
            imageForPositions=self.randomPerspective(imageForPositions) 
            # self.fromTensorToPil(imageForPositions).show()
            #from the image with numbers get the positions of elements equal to i in range di count
            #from these get center with the average of x and y and the width and height with the difference between max and min
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
                #write the label in the right format for yolo
                s=""+str(pieces[i])+" "+str(iCenterCol)+" "+str(iCenterRow)+" "+str(iwidth)+" "+str(iheight)+"\n"
                f.write(s)
            #reset parameters
            self.randomPerspective.initializeParameters()
            f.close()
    def getTensorImage(self,title):
        img=Image.open(self.path+"/"+title)
        return self.fromPilToTensor(img)


# modifier=Modifier("./testModifiedYolo0.5",label=False,distorsion_scale=1)
# modifier.yoloModification()
