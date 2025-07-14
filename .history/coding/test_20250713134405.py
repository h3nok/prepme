import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, Dataset


class LogisticRegression:
    def ___init__(self, input_dim: int, lr:float):
        
        self.w = nn.Parameter(torch.zeros(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))
        self.learning_rate = lr
        
        