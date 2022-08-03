import torch
from torch.utils.data import DataLoader
from dataset import ChessDataset
from nn import BasicMlp

class Trainer:
    def __init__(self,model,optimizer,batchsize,epochs,loss):
        self.model=model
        self.optimizer=optimizer
        self.batchsize=batchsize
        self.epochs=epochs
        self.loss=loss
        self.loadDataset()
    
    def train(self):
        for epoch in range(self.epochs):
            for id,data in enumerate(self.trainDataloader):
                input,label=data[0],data[1]
                self.optimizer.zero_grad()
                print("input {} label {}".format(input.shape,label.shape))
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


model=BasicMlp(40,30,13)
optmimizer=torch.optim.Adam(model.parameters(),lr=0.001)
batchsize=5
epochs=50
loss=torch.nn.CrossEntropyLoss()
trainer=Trainer(model,optmimizer,batchsize,epochs,loss)

trainer.train()
