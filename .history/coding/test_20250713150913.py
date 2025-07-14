import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from PIL import Image
import os
from typing import List, Tuple, Optional, Callable
import numpy as np


class ImageDatasetLoader(Dataset):
    """
    Custom image dataset loader for interview preparation.
    Supports various image formats and transformations.
    """
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image dataset loader.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            transform: Optional transform to be applied to images
            target_size: Target size for resizing images (height, width)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', self.target_size, (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(dataset_name: str = 'cifar10', 
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       augment_data: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders for common datasets.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist')
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        augment_data: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Define transforms
    if augment_data:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    if dataset_name.lower() == 'cifar10':
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    elif dataset_name.lower() == 'mnist':
        # Adjust normalization for MNIST (grayscale)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform)
        val_dataset = MNIST(root='./data', train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def create_custom_dataset(image_dir: str, 
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
                         batch_size: int = 32) -> DataLoader:
    """
    Create a custom dataset loader from a directory of images.
    Assumes directory structure: image_dir/class_name/image_files
    
    Args:
        image_dir: Root directory containing class subdirectories
        extensions: List of valid image file extensions
        batch_size: Batch size for the data loader
        
    Returns:
        DataLoader for the custom dataset
    """
    image_paths = []
    labels = []
    class_names = []
    
    # Get class names from subdirectories
    for class_idx, class_name in enumerate(sorted(os.listdir(image_dir))):
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            
            # Get all image files in this class directory
            for filename in os.listdir(class_path):
                if any(filename.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(class_path, filename))
                    labels.append(class_idx)
    
    print(f"Found {len(image_paths)} images across {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Create dataset and data loader
    dataset = ImageDatasetLoader(image_paths, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return data_loader


# Example usage function
def demo_image_loading():
    """
    Demonstrate how to use the image dataset loaders.
    """
    print("=== Image Dataset Loader Demo ===")
    
    try:
        # Load CIFAR-10 dataset
        print("Loading CIFAR-10 dataset...")
        train_loader, val_loader = create_data_loaders('cifar10', batch_size=16)
        
        # Get a batch of data
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Label values: {labels[:5]}")
        
        # Load MNIST dataset  
        print("\nLoading MNIST dataset...")
        mnist_train, mnist_val = create_data_loaders('mnist', batch_size=16)
        mnist_images, mnist_labels = next(iter(mnist_train))
        print(f"MNIST batch shape: {mnist_images.shape}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("This is normal if torchvision datasets are not available")


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
        self.input_channels = input_channels
        self.num_classes = num_classes

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
            nn.Linear(64 * 7 * 7, 128),  # Assuming 28x28 input -> 7x7 after pooling
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

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10):
        """
        Train the CNN model with validation.

        Args:
            train_loader: DataLoader for the training dataset
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
        """
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            # Training loop
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], '
                          f'Loss: {loss.item():.4f}')
            
            # Calculate training metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct_predictions / total_samples
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
            
            # Validation loop
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    def evaluate(self, test_loader: DataLoader, criterion=None) -> Tuple[float, float]:
        """
        Evaluate the model on test/validation data.
        
        Args:
            test_loader: DataLoader for test/validation data
            criterion: Loss function (optional)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return avg_loss, accuracy

    # Keep the old train method for backward compatibility
    def train(self, train_loader: DataLoader, epochs: int):
        """
        Simple training method (for backward compatibility).
        """
        self.train_model(train_loader, epochs=epochs)


# Example usage for CNN with image data
def demo_cnn_training():
    """
    Demonstrate CNN training with image datasets.
    """
    print("=== CNN Training Demo ===")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader = create_data_loaders('mnist', batch_size=32)
        
        # Create CNN model (MNIST has 1 channel, 10 classes)
        print("Creating CNN model...")
        cnn = CNN(input_channels=1, num_classes=10)
        
        # Train the model
        print("Training CNN...")
        cnn.train_model(train_loader, val_loader, epochs=2)  # Small number for demo
        
        # Evaluate on validation set
        print("Final evaluation...")
        val_loss, val_acc = cnn.evaluate(val_loader)
        print(f"Final Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
    except Exception as e:
        print(f"Error in CNN training demo: {e}")


if __name__ == "__main__":
    # Run demos
    demo_image_loading()
    print("\n" + "="*50 + "\n")
    demo_cnn_training()