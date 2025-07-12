# ---------- 0. Imports ----------
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class CNN(nn.Module):
    def __init__(self):
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
        super().__init__()
    
        #feature extraction 
        self.features_ext = nn.Sequential(
            nn.Conv2d(in_channel=1, number_of_filter=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2D(2),
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2D(2),
        
        )
    
        self.classifier = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.features_ext(x)
        x = x.flatten(1)  # flatten all dimensions except batch
        return self.classifier(x)  # -> (batch, 10)
    
    
    def backward(self, loss):
        """
        Backward pass to compute gradients.
        """
        loss.backward()
    
    def train(self, dataset, batch_size=64, epochs=3, lr=3e-4):
        """
        Train the CNN on the provided dataset.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        learning_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = learning_criterion(self(xb), yb)
                self.backward(loss)
                optimizer.step()
            print(f"epoch {epoch}  loss {loss.item():.4f}")
            
            
dataset = TensorDataset(torch.randn(10_000, 1, 28, 28),
                        torch.randint(0, 10, (10_000,)))
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

model = CNN()

model.train(dataset, batch_size=64, epochs=3, lr=3e-4)


from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 1. Create two linearly-separable blobs
X, y = make_blobs(n_samples=40, centers=2, random_state=42, cluster_std=1.5)
y = (y*2 - 1)          # convert {0,1} → {-1,1}

# 2. Hard-margin SVM (very large C)
clf = SVC(kernel="linear", C=1e5)
clf.fit(X, y)

# 3. Parameters: w·x + b = 0
w, b = clf.coef_[0], clf.intercept_[0]
margin = 1/np.linalg.norm(w)
