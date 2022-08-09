from dataset import ChessDataset
import os
from encoding import Encoding
import torch

class Labeler:
    def __init__(self,path):
        self.path=path
        self.trainTitles=os.listdir(self.path)
        

    def label(self):
        for id,t in enumerate(self.trainTitles):
            print(id)
            titolo=t.replace(".jpeg",".txt")
            f=open(self.path+"/"+titolo,"w")
            objects=[]
            positions=self.positionsFromTitle(t)
            
            
            for i in range(8):
                for j in range(8):
                    if positions[i][j]!=0:
                        element=[positions[i][j],(j*50+25)/400,(i*50+25)/400,50/400,50/400]
                        objects.append(element)
                        s=""+str(element[0])+" "+str(element[1])+" "+str(element[2])+" "+str(element[3])+" "+str(element[4])+"\n"
                        f.write(s)
            
            

            f.close()
    
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
        encod=encoding.encode(positions)
        
        res=torch.tensor(encod).reshape(8,8).tolist()
        return res

if __name__=='__main__':
    labeler=Labeler("./testAnnotated")
    labeler.label()
    labeler=Labeler("./trainAnnotated")
    labeler.label()