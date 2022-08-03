import torch

class BasicMlp(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(BasicMlp, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc3 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc4 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x= torch.nn.functional.relu(self.fc2(x))
        x= torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        x=torch.sigmoid(x)
        return x
