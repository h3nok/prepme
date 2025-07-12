import torch  # PyTorch for tensor operations
from torch import nn, optim  # Neural network and optimization modules

class LinearRegression(nn.Module):
    """
    Simple Linear Regression model using PyTorch.
    y = Xw + b
    """
    def __init__(self, in_features, out_features=1):
        """
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features (default 1)
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)  # Linear layer

    def forward(self, x):
        """
        Forward pass for linear regression.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features)
        Returns:
            Tensor: Output predictions
        """
        return self.linear(x)

# Example usage for Linear Regression
if __name__ == "__main__":
    # Generate random data: y = 2x + 3 + noise
    torch.manual_seed(0)
    X = torch.randn(100, 1)
    y = 2 * X + 3 + 0.1 * torch.randn(100, 1)

    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Training loop
    for epoch in range(200):
        optimizer.zero_grad()  # Zero gradients
        y_pred = model(X)  # Forward pass
        loss = criterion(y_pred, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        if (epoch+1) % 50 == 0:
            print(f"[Linear] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Print learned parameters
    print("Learned weight:", model.linear.weight.item())
    print("Learned bias:", model.linear.bias.item())


class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression model using PyTorch.
    Applies a linear transformation followed by a sigmoid activation.
    """
    def __init__(self, in_features):
        """
        Args:
            in_features (int): Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(in_features, 1)  # Linear layer

    def forward(self, x):
        """
        Forward pass for logistic regression.
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features)
        Returns:
            Tensor: Output probabilities (between 0 and 1)
        """
        return torch.sigmoid(self.linear(x))

# Example usage for Logistic Regression
if __name__ == "__main__":
    # Generate random binary classification data
    torch.manual_seed(0)
    X = torch.randn(100, 2)
    true_w = torch.tensor([[2.0], [-1.0]])
    true_b = 0.5
    logits = X @ true_w + true_b
    y = (torch.sigmoid(logits) > 0.5).float()  # Binary labels (0 or 1)

    model = LogisticRegression(in_features=2)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Training loop
    for epoch in range(200):
        optimizer.zero_grad()  # Zero gradients
        y_pred = model(X)  # Forward pass
        loss = criterion(y_pred, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        if (epoch+1) % 50 == 0:
            print(f"[Logistic] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Print learned parameters
    print("Learned weights:", model.linear.weight.data)
    print("Learned bias:", model.linear.bias.data)
