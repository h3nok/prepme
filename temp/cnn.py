# ---------- 0. Imports ----------
import torch  # PyTorch main package
from torch import nn, optim  # Neural network and optimization modules
from torch.utils.data import DataLoader, TensorDataset  # Data utilities for batching and datasets


class CNN(nn.Module):  # Define a convolutional neural network class inheriting from nn.Module
    def __init__(self):  # Constructor for the CNN class
        """
        A minimal yet complete 3-layer convolutional network:

        Input  : (N, 1, 28, 28)  # MNIST-like greyscale images
        Conv1  : 1→16 kernels 3×3, stride=1, pad=1   -> (N,16,28,28)
        Pool   : 2×2 max-pool                         -> (N,16,14,14)
        Conv2  : 16→32 kernels 3×3, stride=1, pad=1  -> (N,32,14,14)
        Pool   : 2×2 max-pool                         -> (N,32,7,7)
        Flatten: 32·7·7 = 1568 features
        FC     : 1568 → 10 class logits
        """
        super().__init__()  # Call the parent class constructor
    
        #feature extraction 
        self.features_ext = nn.Sequential(  # Sequential container for feature extraction layers
            nn.Conv2d(in_channel=1, number_of_filter=16, kernel_size=3, padding=1),  # 1 input channel, 16 output channels, 3x3 kernel, padding 1
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2D(2),  # 2x2 max pooling
            nn.Conv2d(1, 32, 3, 1),  # 1 input channel, 32 output channels, 3x3 kernel, stride 1
            nn.ReLU(inplace=True),  # ReLU activation
            nn.MaxPool2D(2),  # 2x2 max pooling
        
        )
    
        self.classifier = nn.Linear(32 * 7 * 7, 10)  # Fully connected layer for classification
    
    def forward(self, x):  # Forward pass definition
        x = self.features_ext(x)  # Pass input through feature extractor
        x = x.flatten(1)  # flatten all dimensions except batch
        return self.classifier(x)  # -> (batch, 10)
    
    
    def backward(self, loss):  # Backward pass to compute gradients
        """
        Backward pass to compute gradients.
        """
        loss.backward()  # Compute gradients
    
    def train(self, dataset, batch_size=64, epochs=3, lr=3e-4):  # Training loop for the CNN
        """
        Train the CNN on the provided dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create data loader
        optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer
        learning_criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
        
        for epoch in range(epochs):  # Loop over epochs
            for xb, yb in loader:  # Loop over batches
                optimizer.zero_grad()  # Zero gradients
                loss = learning_criterion(self(xb), yb)  # Compute loss
                self.backward(loss)  # Backward pass
                optimizer.step()  # Update parameters
            print(f"epoch {epoch}  loss {loss.item():.4f}")  # Print loss for epoch
            
            
dataset = TensorDataset(torch.randn(10_000, 1, 28, 28),  # Random images (10,000 samples, 1 channel, 28x28)
                        torch.randint(0, 10, (10_000,)))  # Random labels (10 classes)
# loader  = DataLoader(dataset, batch_size=64, shuffle=True)  # DataLoader for batching

model = CNN()  # Instantiate the CNN model

model.train(dataset, batch_size=64, epochs=3, lr=3e-4)  # Train the model


