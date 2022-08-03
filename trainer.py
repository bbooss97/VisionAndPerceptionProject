import torch
from torch.utils.data import DataLoader
from dataset import ChessDataset


class Trainer:
    def __init__(self,model,optimizer,batchsize,epochs,loss):
        self.model=model
        self.optimizer=optimizer
        self.batchsize=batchsize
        self.epochs=epochs
        self.loss=loss
        self.loadDataset()
    
    def train(self):
        for i in range(self.epochs):
            for id,data in enumerate(self.trainDataloader):
                input,label=data[0],data[1]
                self.optimizer.zero_grad()

                patches=self.getPatches(input,label)

                for patch in patches:
                    output=self.model(patch)
                    loss=self.loss(output,label)
                    loss.backward()
                    self.optimizer.step()

             
                
        

    def test(self):
        pass

    def loadDataset(self):
        self.trainDataset=ChessDataset(type="train")
        self.testDataset=ChessDataset(type="test")
        self.trainDataloader = DataLoader(self.trainDataset, batch_size=self.batchsize, shuffle=True)
        self.testDataloader = DataLoader(self.testDataset, batch_size=self.batchsize, shuffle=True)

