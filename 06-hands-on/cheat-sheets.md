# Quick Reference Cheat Sheets

## 🚀 Essential Formulas & Concepts

### Transformer Architecture
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Training Optimization
```python
# Learning Rate Scheduling
lr = base_lr * (decay_rate ** (global_step / decay_steps))

# Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Mixed Precision
with torch.cuda.amp.autocast():
    loss = model(batch)
scaler.scale(loss).backward()
```

### Memory Optimization
- **Gradient Checkpointing**: Trade compute for memory
- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **Flash Attention**: Memory-efficient attention computation
- **Model Sharding**: Split model across devices

### AWS Key Services
- **SageMaker**: Training, inference, pipelines
- **Bedrock**: Foundation models API
- **S3**: Data storage and versioning
- **EC2**: Compute instances (p4d, p3 for ML)
- **EFS**: Shared file systems for training

## 🎯 Interview Formulas

### Model Scaling Laws
```
L(N) = (N_c/N)^α + L_∞
where N = parameters, α ≈ 0.076 for transformers
```

### Diffusion Models
```python
# Forward process
q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

# Reverse process  
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
```

### Evaluation Metrics
- **BLEU**: n-gram overlap for text generation
- **FID**: Frechet Inception Distance for images
- **CLIP Score**: Vision-language alignment
- **Perplexity**: Language model quality

## 📋 Common Interview Topics

### Technical Deep Dives
1. **Attention Mechanisms**: Self vs cross-attention, computational complexity
2. **Training Stability**: Gradient issues, normalization techniques
3. **Scaling Laws**: Parameter vs data scaling, emergent abilities
4. **Multimodal Fusion**: Early vs late fusion, attention mechanisms

### Research Questions
1. **Problem Formulation**: How to define research objectives
2. **Experimental Design**: Controls, baselines, metrics
3. **Reproducibility**: Code, data, environment documentation
4. **Impact Assessment**: Academic vs business value

### AWS-Specific
1. **Cost Optimization**: Spot instances, auto-scaling, efficient architectures
2. **Security**: IAM, VPC, encryption, compliance
3. **Monitoring**: CloudWatch, custom metrics, alerting
4. **Deployment**: Blue-green, canary, A/B testing

## 🔧 Code Templates

### PyTorch Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

### Distributed Training Setup
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Wrap model
model = DDP(model.cuda(local_rank), device_ids=[local_rank])
```

### AWS SageMaker Estimator
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p3.8xlarge',
    instance_count=4,
    framework_version='1.12',
    py_version='py38',
    hyperparameters={
        'learning_rate': 1e-4,
        'batch_size': 32
    }
)
```

## 🎪 Presentation Tips

### Technical Talks
1. **Start with motivation**: Why is this problem important?
2. **Clear methodology**: What did you do exactly?
3. **Strong results**: Show compelling evidence
4. **Honest limitations**: What didn't work or needs improvement?
5. **Future directions**: What's next?

### Demo Guidelines
1. **Have backups**: Screenshots, videos if live demo fails
2. **Practice transitions**: Smooth flow between concepts
3. **Explain while showing**: Don't just click through
4. **Prepare for questions**: Anticipate deep technical queries

## 📚 Must-Know Papers (2023-2024)

### LLMs
- **GPT-4 Technical Report** (OpenAI, 2023)
- **PaLM 2 Technical Report** (Google, 2023)
- **LLaMA 2** (Meta, 2023)

### Multimodal
- **GPT-4V System Card** (OpenAI, 2023)
- **DALL-E 3** (OpenAI, 2023)
- **Flamingo** (DeepMind, 2022)

### Training
- **Flash Attention 2** (Stanford, 2023)
- **QLoRA** (Washington, 2023)
- **Constitutional AI** (Anthropic, 2022)

---

*Print this out and keep it handy during interviews!* 📄
