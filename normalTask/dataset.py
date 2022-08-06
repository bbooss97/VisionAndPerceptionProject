import torch
from encoding import Encoding
import os
import torchvision

class ChessDataset(torch.utils.data.Dataset):
    stride=50
    def __init__(self,type="train"):
        if type == "train":
            self.path="./train"
        else:
            self.path="./test"
        self.trainTitles=os.listdir(self.path)
        self.titleToId={i : title for i,title in enumerate(self.trainTitles) }
        

    def __len__(self):
        return len(self.trainTitles)

    def __getitem__(self, idx):
        itempath=self.path+"/"+self.titleToId[idx]
        image = torchvision.io.read_image(itempath)
        label = self.positionsFromTitle(self.titleToId[idx])
        image=image/255
        # if transform:
        #     transform(image)

        return image, label
    
    def positionsFromTitle(self,title):
        positions=[]
        title=title[:-5]
        rows=title.split("-")
        for row in rows:
            for i in range(len(row)):
                if row[i].isnumeric():
                    for _ in range(int(row[i])):
                        positions.append("0")
                else:
                    positions.append(row[i])
        encoding=Encoding()
        # res=torch.tensor(encoding.encode(positions))
        res=torch.tensor(encoding.oneHotEncoding(positions)).float()
        return res
