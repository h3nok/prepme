# Training Optimization for Large Scale Models

## 🎯 Overview
Deep dive into optimization techniques for training large-scale generative AI models, covering memory efficiency, distributed training, gradient optimization, and scaling strategies.

## 📊 Training Dynamics & Optimization

### Gradient Descent Variants

#### Adam Optimizer Family
```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import math

class AdamWWithWarmup:
    def __init__(self, model, lr=1e-4, weight_decay=0.01, warmup_steps=4000):
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        # Warmup schedule
        if self.step_count <= self.warmup_steps:
            lr_scale = min(1.0, self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (100000 - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * lr_scale
            
        self.optimizer.step()
```

#### Learning Rate Schedules
```python
class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

### Gradient Clipping & Stabilization
```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """Enhanced gradient clipping with monitoring"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.)
    
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) 
                                        for p in parameters]), norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    
    return total_norm
```

## 🔧 Memory Optimization Techniques

### Gradient Checkpointing
```python
import torch.utils.checkpoint as checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, attention_mask=None):
        if self.training:
            return checkpoint.checkpoint(self._forward, x, attention_mask)
        else:
            return self._forward(x, attention_mask)
    
    def _forward(self, x, attention_mask):
        # Self-attention
        residual = x
        x = self.ln1(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def training_step(self, batch):
        self.optimizer.zero_grad()
        
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
            
        # Scaled backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping in scaled space
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### Memory-Efficient Attention
```python
class MemoryEfficientAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x, attention_mask=None):
        B, L, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        
        # Flash attention pattern - chunk processing
        chunk_size = 1024
        output = torch.zeros_like(q[0])
        
        for i in range(0, L, chunk_size):
            end_i = min(i + chunk_size, L)
            q_chunk = q[:, :, i:end_i]
            
            for j in range(0, L, chunk_size):
                end_j = min(j + chunk_size, L)
                k_chunk = k[:, :, j:end_j]
                v_chunk = v[:, :, j:end_j]
                
                # Compute attention for chunk
                attn = torch.softmax(q_chunk @ k_chunk.transpose(-2, -1) * self.scale, dim=-1)
                output[:, :, i:end_i] += attn @ v_chunk
        
        output = output.transpose(1, 2).reshape(B, L, C)
        return self.proj(output)
```

## 🌐 Distributed Training

### Data Parallel Training
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Initialize process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Move model to GPU and wrap with DDP
        torch.cuda.set_device(rank)
        model = model.cuda(rank)
        self.model = DDP(model, device_ids=[rank])
        
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.cuda(self.rank) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Synchronize gradients across processes
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
```

### Model Parallel Training (Pipeline)
```python
class PipelineParallelModel(nn.Module):
    def __init__(self, config, device_map):
        super().__init__()
        self.device_map = device_map
        self.stages = nn.ModuleList()
        
        layers_per_stage = config.num_layers // len(device_map)
        
        for stage_id, device in enumerate(device_map):
            start_layer = stage_id * layers_per_stage
            end_layer = min((stage_id + 1) * layers_per_stage, config.num_layers)
            
            stage_layers = nn.ModuleList([
                TransformerBlock(config) for _ in range(start_layer, end_layer)
            ]).to(device)
            
            self.stages.append(stage_layers)
    
    def forward(self, x):
        for stage_id, stage in enumerate(self.stages):
            x = x.to(self.device_map[stage_id])
            for layer in stage:
                x = layer(x)
        return x
```

### Gradient Accumulation
```python
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=8):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        
    def train_step(self, batch):
        # Scale loss by accumulation steps
        outputs = self.model(**batch)
        loss = outputs.loss / self.accumulation_steps
        loss.backward()
        
        return loss.item() * self.accumulation_steps
    
    def should_step(self, step):
        return (step + 1) % self.accumulation_steps == 0
    
    def optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
```

## 🎛️ Advanced Optimization Techniques

### Layer-wise Adaptive Rate Scaling (LARS)
```python
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=1e-4, trust_coefficient=1e-3):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, 
                       trust_coefficient=trust_coefficient)
        super(LARS, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)
                
                if param_norm > 0 and grad_norm > 0:
                    # Compute adaptive learning rate
                    adaptive_lr = trust_coefficient * param_norm / grad_norm
                    adaptive_lr = min(adaptive_lr, group['lr'])
                else:
                    adaptive_lr = group['lr']
                
                # Apply momentum and weight decay
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(p.grad.data)
                
                p.data.add_(buf, alpha=-adaptive_lr)
        
        return loss
```

### Parameter Efficient Fine-tuning
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
    def forward(self, x):
        # x @ A^T @ B^T * scaling
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha)
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

## 📈 Training Monitoring & Diagnostics

### Training Metrics Dashboard
```python
class TrainingMonitor:
    def __init__(self, log_dir="./logs"):
        self.metrics = {}
        self.log_dir = log_dir
        self.step = 0
        
    def log_scalar(self, name, value, step=None):
        if step is None:
            step = self.step
            
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))
    
    def log_gradient_stats(self, model):
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log per-layer gradient norms
                self.log_scalar(f"grad_norm/{name}", param_norm.item())
        
        total_norm = total_norm ** (1. / 2)
        self.log_scalar("grad_norm/total", total_norm)
        
    def log_learning_rate(self, optimizer):
        for i, param_group in enumerate(optimizer.param_groups):
            self.log_scalar(f"lr/group_{i}", param_group['lr'])
    
    def log_loss_landscape(self, model, dataloader, num_samples=100):
        """Sample loss landscape around current parameters"""
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        losses = []
        perturbations = []
        
        for _ in range(num_samples):
            # Random perturbation
            epsilon = 0.01
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.randn_like(param) * epsilon
                    param.add_(noise)
                    perturbations.append(noise.norm().item())
            
            # Evaluate loss
            model.eval()
            with torch.no_grad():
                batch = next(iter(dataloader))
                outputs = model(**batch)
                losses.append(outputs.loss.item())
            
            # Restore parameters
            with torch.no_grad():
                for (name, param), original_param in zip(model.named_parameters(), original_state.values()):
                    param.copy_(original_param)
        
        self.log_scalar("loss_landscape/variance", torch.tensor(losses).var().item())
        self.log_scalar("loss_landscape/mean", torch.tensor(losses).mean().item())
```

### Early Stopping & Checkpointing
```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class CheckpointManager:
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, metadata=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metadata': metadata or {}
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Remove old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.max_checkpoints:
            # Sort by epoch number and remove oldest
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
```

## 🔬 Hyperparameter Optimization

### Bayesian Optimization for Hyperparameters
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

class BayesianHyperparameterOptimizer:
    def __init__(self, parameter_bounds, acquisition_function='ucb'):
        self.bounds = parameter_bounds
        self.acquisition_function = acquisition_function
        self.X_sample = []
        self.y_sample = []
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
    def suggest_parameters(self):
        if len(self.X_sample) < 2:
            # Random exploration for first few samples
            return self._random_sample()
        
        # Fit GP to existing data
        self.gp.fit(self.X_sample, self.y_sample)
        
        # Optimize acquisition function
        best_x = None
        best_acquisition = float('-inf')
        
        for _ in range(1000):  # Random search for acquisition optimization
            x_candidate = self._random_sample()
            acquisition_val = self._acquisition(x_candidate)
            
            if acquisition_val > best_acquisition:
                best_acquisition = acquisition_val
                best_x = x_candidate
                
        return best_x
    
    def update(self, parameters, performance):
        self.X_sample.append(parameters)
        self.y_sample.append(performance)
    
    def _random_sample(self):
        sample = {}
        for param, (low, high) in self.bounds.items():
            if isinstance(low, int):
                sample[param] = np.random.randint(low, high + 1)
            else:
                sample[param] = np.random.uniform(low, high)
        return sample
    
    def _acquisition(self, x):
        x_array = np.array([list(x.values())]).reshape(1, -1)
        mu, sigma = self.gp.predict(x_array, return_std=True)
        
        if self.acquisition_function == 'ucb':
            # Upper Confidence Bound
            return mu + 2.0 * sigma
        elif self.acquisition_function == 'ei':
            # Expected Improvement
            best_y = max(self.y_sample) if self.y_sample else 0
            z = (mu - best_y) / (sigma + 1e-9)
            return (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
```

## 📋 Training Best Practices

### 1. **Training Pipeline Design**
- **Data Loading**: Efficient data loading with prefetching and parallel workers
- **Batch Size**: Start with smaller batches, scale up with learning rate
- **Validation**: Regular validation with early stopping
- **Logging**: Comprehensive logging of metrics, gradients, and learning rates

### 2. **Scaling Laws**
- **Compute Budget**: C = 6ND (N=parameters, D=dataset size)
- **Optimal Ratio**: N ∝ C^0.73, D ∝ C^0.27
- **Batch Size Scaling**: Increase batch size with model size
- **Learning Rate**: Scale with √(batch_size) or linear scaling

### 3. **Memory Management**
- **Gradient Checkpointing**: Trade compute for memory (√n memory reduction)
- **Mixed Precision**: FP16 for forward/backward, FP32 for optimization
- **Optimizer States**: Consider memory footprint of optimizer states
- **Dynamic Loss Scaling**: Prevent underflow in mixed precision

### 4. **Distributed Training Strategy**
- **Data Parallel**: Multiple GPUs, same model
- **Model Parallel**: Split model across GPUs
- **Pipeline Parallel**: Sequential stages across GPUs
- **Hybrid**: Combination of above strategies

## 🎯 Interview Questions & Answers

### Q1: How would you optimize training for a 175B parameter model?
**Answer**: 
1. **Model Parallelism**: Split layers across multiple GPUs using pipeline parallelism
2. **ZeRO Optimizer**: Use ZeRO-3 to partition optimizer states, gradients, and parameters
3. **Gradient Checkpointing**: Reduce memory by recomputing activations
4. **Mixed Precision**: Use FP16 with automatic loss scaling
5. **Efficient Attention**: Implement Flash Attention for memory-efficient attention computation
6. **Gradient Accumulation**: Simulate larger batch sizes across multiple steps

### Q2: Explain the trade-offs in different attention mechanisms for long sequences.
**Answer**:
- **Full Attention**: O(n²) complexity but best quality
- **Linear Attention**: O(n) complexity but information loss
- **Sparse Attention**: Reduces computation but requires careful pattern design
- **Flash Attention**: Memory-efficient full attention using tiling
- **Ring Attention**: Distributed attention computation across devices

### Q3: How do you handle gradient explosion in large model training?
**Answer**:
1. **Gradient Clipping**: Clip gradients by norm (typically max_norm=1.0)
2. **Learning Rate Scheduling**: Use warmup and decay schedules
3. **Batch Size Management**: Smaller batches can help stability
4. **Mixed Precision**: Proper loss scaling prevents numerical issues
5. **Architecture Design**: Skip connections and normalization layers
6. **Optimizer Choice**: AdamW with weight decay for stability

### Q4: What metrics do you monitor during large-scale training?
**Answer**:
- **Loss Curves**: Training and validation loss trends
- **Gradient Norms**: Per-layer and total gradient magnitudes
- **Learning Rate**: Current learning rate values
- **Memory Usage**: GPU memory utilization
- **Throughput**: Tokens per second, samples per second
- **Hardware Metrics**: GPU utilization, temperature, communication overhead
- **Model Quality**: Perplexity, downstream task performance

### Q5: How would you debug slow training convergence?
**Answer**:
1. **Learning Rate**: Too high (unstable) or too low (slow convergence)
2. **Batch Size**: Too small (noisy gradients) or too large (poor generalization)
3. **Data Quality**: Check for data leakage, imbalance, or preprocessing issues
4. **Architecture**: Verify proper initialization, normalization, and skip connections
5. **Optimization**: Check gradient flow and potential bottlenecks
6. **Hardware**: Ensure efficient data loading and GPU utilization

## 🔗 Additional Resources

- **Papers**: "Scaling Laws for Neural Language Models", "Training Compute-Optimal Large Language Models"
- **Documentation**: PyTorch DDP, Hugging Face Accelerate, DeepSpeed
- **Tools**: Weights & Biases, TensorBoard, NVIDIA Nsight
- **Libraries**: Flash Attention, Megatron-LM, FairScale
