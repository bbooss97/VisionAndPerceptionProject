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
    def encode(self,list):
        encoded=[self.encoding[i] for i in list]
        return encoded