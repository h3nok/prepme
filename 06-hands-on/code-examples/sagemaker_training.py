"""
AWS SageMaker Training Example
=============================

Production-ready example of training a model on AWS SageMaker.
Demonstrates best practices for ML training in AWS environment.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3

class ExampleDataset(Dataset):
    """Simple dataset for demonstration"""
    
    def __init__(self, size=1000, input_dim=100):
        self.size = size
        self.input_dim = input_dim
        
        # Generate synthetic data
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleClassifier(nn.Module):
    """Simple neural network for binary classification"""
    
    def __init__(self, input_dim=100, hidden_dim=64, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(args):
    """
    Training function that runs on SageMaker
    
    Args:
        args: Command line arguments containing hyperparameters
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = ExampleDataset(size=8000)
    val_dataset = ExampleDataset(size=2000)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = SimpleClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{args.model_dir}/best_model.pth')
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), f'{args.model_dir}/final_model.pth')
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

def launch_sagemaker_training():
    """
    Launch training job on SageMaker
    """
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()  # IAM role for SageMaker
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'input_dim': 100,
        'hidden_dim': 64,
        'num_classes': 2,
        'num_workers': 4
    }
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='train.py',  # This script
        source_dir='.',
        role=role,
        instance_type='ml.p3.2xlarge',  # GPU instance
        instance_count=1,
        framework_version='1.12.0',
        py_version='py38',
        hyperparameters=hyperparameters,
        
        # Training job configuration
        max_run=3600,  # Max runtime in seconds
        use_spot_instances=True,  # Cost optimization
        max_wait=7200,  # Max wait time for spot instances
        
        # Output configuration
        output_path='s3://your-bucket/models/',
        code_location='s3://your-bucket/code/',
        
        # Environment variables
        environment={
            'PYTHONPATH': '/opt/ml/code',
            'CUDA_VISIBLE_DEVICES': '0'
        }
    )
    
    # Start training
    estimator.fit({
        'training': 's3://your-bucket/data/train/',
        'validation': 's3://your-bucket/data/val/'
    })
    
    return estimator

def create_model_for_inference(estimator):
    """
    Deploy trained model for inference
    """
    
    # Create model
    model = estimator.create_model(
        model_server_timeout=60,
        model_server_workers=1
    )
    
    # Deploy to endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',  # CPU instance for inference
        endpoint_name='my-model-endpoint'
    )
    
    return predictor

def distributed_training_example():
    """
    Example of distributed training configuration
    """
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=sagemaker.get_execution_role(),
        
        # Multi-instance configuration
        instance_type='ml.p3.8xlarge',
        instance_count=4,  # 4 instances
        
        # Distributed training configuration
        distribution={
            'torch_distributed': {
                'enabled': True
            }
        },
        
        framework_version='1.12.0',
        py_version='py38',
        
        hyperparameters={
            'epochs': 20,
            'batch_size': 32,  # Per-device batch size
            'learning_rate': 0.001
        }
    )
    
    return estimator

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--input_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # SageMaker specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './models'))
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data/train'))
    parser.add_argument('--val_dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data/val'))
    
    args = parser.parse_args()
    
    # Run training
    train_model(args)


"""
AWS SageMaker Best Practices & Interview Points:

1. **Cost Optimization**:
   - Use spot instances for training (up to 90% cost savings)
   - Right-size instances based on workload
   - Use S3 Intelligent Tiering for data storage
   - Monitor usage with AWS Cost Explorer

2. **Security**:
   - IAM roles with least privilege principle
   - VPC configuration for network isolation
   - Encryption at rest and in transit
   - AWS PrivateLink for secure API access

3. **Scalability**:
   - Distributed training across multiple instances
   - Model parallelism for large models
   - Auto-scaling endpoints based on traffic
   - Batch transform for large-scale inference

4. **Monitoring & Debugging**:
   - CloudWatch metrics and logs
   - SageMaker Debugger for training insights
   - Model Monitor for data drift detection
   - A/B testing with multi-variant endpoints

5. **MLOps Integration**:
   - SageMaker Pipelines for workflow orchestration
   - Model Registry for version control
   - Automatic model deployment with CI/CD
   - Feature Store for feature management

6. **Common Configurations**:
   - P3/P4 instances for GPU training
   - EFS for shared storage across instances
   - ECR for custom container images
   - CloudFormation for infrastructure as code
"""
