import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CIFAR100, ImageNet


class LogisticRegression:
    def ___init__(self, input_dim: int):
        
        self.w = nn.Parameter(torch.zeros(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))
        
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

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int):
        """
        Train the logistic regression model.

        Args:
            X: Input tensor of shape (n_samples, n_features)
            y: Target tensor of shape (n_samples, 1)
            epochs: Number of training epochs
        """
        optimizer = optim.SGD([self.w, self.b], lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X)
            loss = nn.BCELoss()(output, y)
            loss.backward()
            optimizer.step()


class CNN:
    def __init__(self, input_channels: int, num_classes: int):
        """
        Initialize the CNN model.

        Args:
            input_channels: Number of input channels (e.g., 1 for grayscale images)
            num_classes: Number of output classes
        """


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.model = nn.Sequential(self.feature_extractor, self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN model.

        Args:
            x: Input tensor of shape (n_samples, input_channels, height, width)

        Returns:
            Output tensor of shape (n_samples, num_classes) with class probabilities
        """
        return self.model(x)

    def train(self, train_loader: DataLoader, epochs: int):
        """
        Train the CNN model.

        Args:
            train_loader: DataLoader for the training dataset
            epochs: Number of training epochs
        """
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


data_loader = 