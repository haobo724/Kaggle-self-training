import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
        def __init__(self, x, y):
            self.data = torch.from_numpy(np.array(x, dtype=np.float32))
            self.labels = torch.from_numpy(np.array(y, dtype=np.float32))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]
        
class NeuralNetwork(torch.nn.Module):
    def __init__(self,input_size):
        super(NeuralNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size*4),
            torch.nn.BatchNorm1d(input_size*4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(input_size*4, input_size*4),
            torch.nn.BatchNorm1d(input_size*4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear( input_size*4, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        x= self.net(x)
        return x