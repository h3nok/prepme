import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, Dataset


class LogisticRegression:
    def ___init__(self, input_dim: int, lr:float):
        
        self.w = nn.Parameter(torch.zeros(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))
        self.learning_rate = lr
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for logistic regression.
        
        Args:
            X: Input tensor of shape (n_samples, n_features)
        
        Returns:
            Output tensor of shape (n_samples, 1) with probabilities
        """
        logits = X @ self.w + self.b
        return torch.sigmoid(logits)