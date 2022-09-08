import torch
from encoding import Encoding
import os
import torchvision
#this class is used to load the dataset given as input
class ChessDataset(torch.utils.data.Dataset):
    def __init__(self,type="train",percentage=1):
        #i load a percentage of the initial folder
        self.percentage=percentage
        #i set the path based on the type
        if type == "train":
            self.path="./train"
        else:
            self.path="./test"
        #i list all the elements in the folder
        self.trainTitles=os.listdir(self.path)
        #i get the first elements based on the percentage passed in input
        self.trainTitles=self.trainTitles[:int(len(self.trainTitles)*self.percentage)]
        #i create a dictionary that maps the index to the title used by get elements
        self.titleToId={i : title for i,title in enumerate(self.trainTitles) }
        
    #return len of database
    def __len__(self):
        return len(self.trainTitles)
    #override to get element by id
    def __getitem__(self, idx):
        itempath=self.path+"/"+self.titleToId[idx]
        #read the image
        image = torchvision.io.read_image(itempath)
        #i get the label from the title of elemnet in index idx
        label = self.positionsFromTitle(self.titleToId[idx])
        #i normalize the input image 
        image=image/255
        #return the couple image and its label
        return image, label
    #return the positions label from the title
    def positionsFromTitle(self,title):
        positions=[]
        #remove the extension
        title=title[:-5]
        #split the rows
        rows=title.split("-")
        #for every row i scan the pieces and add the pieces to the positions list
        for row in rows:
            for i in range(len(row)):
                if row[i].isnumeric():
                    for _ in range(int(row[i])):
                        positions.append("0")
                else:
                    positions.append(row[i])
        encoding=Encoding()
        # res=torch.tensor(encoding.encode(positions))
        # i return the one hot encoding of the positions
        res=torch.tensor(encoding.oneHotEncoding(positions)).float()
        return res
