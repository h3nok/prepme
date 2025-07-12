import torch  # PyTorch for tensor operations
from torch import nn, optim  # Neural network and optimization modules

class SimpleMLP(nn.Module):
    """
    Simple feedforward neural network (MLP) for classification.
    Architecture: Input -> Hidden (ReLU) -> Output (logits)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units
            output_dim (int): Number of output classes
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input to hidden
            nn.ReLU(),                        # Non-linearity
            nn.Linear(hidden_dim, output_dim)  # Hidden to output
        )

    def forward(self, x):
        """
        Forward pass for the MLP.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
        Returns:
            Tensor: Output logits (before softmax)
        """
        return self.net(x)

# Example usage for SimpleMLP
if __name__ == "__main__":
    # Generate random classification data: 3 classes, 2 features
    torch.manual_seed(0)
    X = torch.randn(200, 2)  # 200 samples, 2 features
    y = torch.randint(0, 3, (200,))  # 3 classes (0, 1, 2)

    model = SimpleMLP(input_dim=2, hidden_dim=16, output_dim=3)
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

    # Training loop (no DataLoader, direct tensor batches)
    for epoch in range(100):
        optimizer.zero_grad()  # Zero gradients
        logits = model(X)  # Forward pass
        loss = criterion(logits, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        if (epoch+1) % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}")

    # Print final weights of first layer
    print("First layer weights:", model.net[0].weight.data)
