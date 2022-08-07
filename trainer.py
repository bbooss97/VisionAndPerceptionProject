import torch
from torch.utils.data import DataLoader
from dataset import ChessDataset
from nn import BasicMlp
from torchmetrics import F1Score
import wandb
import torchvision
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Trainer:
    def __init__(self,model,optimizer,batchsize,epochs,lossfn,percentage,type="mlp"):
        self.model=model
        self.percentage=percentage
        self.type=type
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
                mod=1
                typesToChange=["resnetFrom0","resnetPretrainedFineTuneFc","resnetPretrainedFineTuneAll","mobilenetPretrainedFineTuneAll"]
                if self.type in typesToChange:
                    # mod=1
                    patches =patches.reshape(64*self.batchsize,50,50,3)
                    patches=torch.einsum("abcd->adbc",patches)
                
                output=self.model(patches)
                
                loss=self.lossfn(output,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    predicted=torch.argmax(output,dim=1)
                    label=torch.argmax(label,dim=1)
                    f1=F1Score(num_classes=13)
                    accuracy=f1(predicted,label)

                if it%mod==0:
                    print("Epoch:",epoch,"Iteration:",it,"Loss:",loss.data.mean(),"Training accuracy:",accuracy)
                    wandb.log({"epoch_train":epoch,"iteration_train":it,"loss_train":loss.data.mean(),"accuracy_train":accuracy})
            self.test() 
    def test(self):
        print("testing phase")
        count=0
        it=0
        test_loss=0
        test_accuracy=0
        with torch.no_grad():
            for id,data in enumerate(self.testDataloader):
                    it+=1
                    input,label=data[0].float(),data[1]
                    patches=self.getPatches(input)
                    patches=patches.reshape(64*self.batchsize,-1)
                    label=label.reshape(64*self.batchsize,-1)
                    
                    patches.to(device)
                    label.to(device)
                    mod=1
                    typesToChange=["resnetFrom0","resnetPretrainedFineTuneFc","resnetPretrainedFineTuneAll","mobilenetPretrainedFineTuneAll"]
                    if self.type in typesToChange:
                        # mod=1
                        patches =patches.reshape(64*self.batchsize,50,50,3)
                        patches=torch.einsum("abcd->adbc",patches)
                    
                    output=self.model(patches)
                    loss=self.lossfn(output,label)
                    predicted=torch.argmax(output,dim=1)
                    label=torch.argmax(label,dim=1)
                    f1=F1Score(num_classes=13)
                    accuracy=f1(predicted,label)
                    test_accuracy+=accuracy
                    test_loss+=loss.data.mean()
        print("mean_loss_test",test_loss/it,"mean_accuracy_test:",test_accuracy/it)
        wandb.log({"mean_loss_test":test_loss/it,"mean_accuracy_test":test_accuracy/it})

    def loadDataset(self):
        self.trainDataset=ChessDataset(type="train",percentage=self.percentage)
        self.testDataset=ChessDataset(type="test",percentage=self.percentage)
        self.trainDataloader = DataLoader(self.trainDataset, batch_size=self.batchsize, shuffle=True)
        self.testDataloader = DataLoader(self.testDataset, batch_size=self.batchsize, shuffle=True)

type="mobilenetPretrainedFineTuneAll"
wandb.init(project='visionAndPerceptionProject', entity='bbooss97',name=type)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

if type=="mlp":
    model=BasicMlp(7500,50,13)
elif type=="resnetPretrainedFineTuneFc":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc=torch.nn.Linear(512,13)
    toFreeze=[j for i,j in model.named_parameters()][:-2]
    for i in toFreeze:
        i.requires_grad=False
elif type=="resnetPretrainedFineTuneAll":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.fc=torch.nn.Linear(512,13)
elif type=="resnetFrom0":
    model=torch.hub.load('pytorch/vision:v0.6.0', 'resnet18',pretrained=False)
    model.fc=torch.nn.Linear(512,13)

elif type=="mobilenetPretrainedFineTuneAll":
    model=torchvision.models.mobilenet_v3_small()
    model.classifier[3]=torch.nn.Linear(1024,13)



model.to(device)
wandb.watch(model)
batchsize=5
epochs=5
loss=torch.nn.CrossEntropyLoss()

optmimizer=torch.optim.Adam(model.parameters())
trainer=Trainer(model,optmimizer,batchsize,epochs,loss,type=type,percentage=0.01)

trainer.train()

