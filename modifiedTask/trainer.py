import torch
from torch.utils.data import DataLoader
from dataset import ChessDataset
from nn import BasicMlp
from torchmetrics import F1Score
class Trainer:
    def __init__(self,model,optimizer,batchsize,epochs,lossfn):
        self.model=model
        self.optimizer=optimizer
        self.batchsize=batchsize
        self.epochs=epochs
        self.lossfn=lossfn
        self.loadDataset()
    def getPatches(self,obs):
        unfold=torch.nn.Unfold(kernel_size=(50,50),stride=50)
        obs=unfold(obs)
        obs=obs.transpose(1,2)
        return obs

    def train(self):
        it=0
        for epoch in range(self.epochs):
            for id,data in enumerate(self.trainDataloader):
                it+=1
                
                input,label=data[0].float(),data[1]
                
                
                patches=self.getPatches(input)

                patches=patches.reshape(64*self.batchsize,-1)
                label=label.reshape(64*self.batchsize,-1)
                
                patches.to(device)
                label.to(device)
                
                
                output=self.model(patches)
                loss=self.lossfn(output,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # with torch.no_grad():
                #     argmax=torch.argmax(output,dim=1)
                #     onehotoutput=torch.zeros_like(output)
                #     onehotoutput[torch.arange(64*self.batchsize),argmax]=1
                #     onehotoutput.to(device)
                    
                #     label.float()
                #     onehotoutput.float()
                #     # print(onehotoutput)
                #     onehotoutput=torch.where(onehotoutput==0,-1,onehotoutput)
                #     # print(accuracy)
                #     # accuracy=torch.sum(onehotoutput==label).float().item()
                #     accuracy=torch.sum(onehotoutput==label).float().item()/(64*self.batchsize)
                with torch.no_grad():
                    predicted=torch.argmax(output,dim=1)
                    label=torch.argmax(label,dim=1)
                    f1=F1Score(num_classes=13)
                    accuracy=f1(predicted,label)

                if it%100==0:
                    print("Epoch:",epoch,"Iteration:",it,"Loss:",loss.data.mean(),"Training accuracy:",accuracy)
             
                
        

    def test(self):
        pass

    def loadDataset(self):
        self.trainDataset=ChessDataset(type="train")
        self.testDataset=ChessDataset(type="test")
        self.trainDataloader = DataLoader(self.trainDataset, batch_size=self.batchsize, shuffle=True)
        self.testDataloader = DataLoader(self.testDataset, batch_size=self.batchsize, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=BasicMlp(7500,50,13)

# model.to(device)
batchsize=5
epochs=2
loss=torch.nn.CrossEntropyLoss()

optmimizer=torch.optim.Adam(model.parameters())
trainer=Trainer(model,optmimizer,batchsize,epochs,loss)

trainer.train()
