#class used to encode the pieces in numbers
class Encoding:
    def __init__(self):
        self.encoding={
            "P":1,
            "p":2,
            "R":3,
            "r":4,
            "N":5,
            "n":6,
            "B":7,
            "b":8,
            "Q":9,
            "q":10,
            "K":11,
            "k":12,
            "0":0
        }
        self.reverseEncoding={
            1:"P",
            2:"p",
            3:"R",
            4:"r",
            5:"N",
            6:"n",
            7:"B",
            8:"b",
            9:"Q",
            10:"q",
            11:"K",
            12:"k",
            0:"0"
        }
    #returns a list of encoded pieces
    def encode(self,list):
        encoded=[self.encoding[i] for i in list]
        return encoded
    #returns a list of encoded pieces in a one hot encoding way
    def oneHotEncoding(self,list):
        res=[[0 for i in range(13)] for j in range(len(list))]
        for i,n in enumerate(list):
            res[i][self.encoding[n]]=1
        return res