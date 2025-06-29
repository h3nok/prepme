# Large Language Models (LLMs)

## 🎯 Learning Objectives
- Understand LLM training phases and methodologies
- Master scaling laws and emergent abilities
- Know RLHF and alignment techniques
- Understand different LLM architectures and their trade-offs

## 🏗️ LLM Foundation

### What Makes a Model "Large"?
**Scale Dimensions**:
1. **Parameters**: 1B+ (GPT-3: 175B, GPT-4: ~1.7T)
2. **Training data**: 100B+ tokens
3. **Compute**: 1000+ GPU-days
4. **Emergent abilities**: New capabilities at scale

### Key Architectural Choices for LLMs

#### Decoder-Only Architecture (GPT-style)
```
[BOS] token₁ token₂ ... tokenₙ [EOS]
  ↓      ↓      ↓         ↓      ↓
 emb₁  emb₂   emb₃     embₙ   pred
```

**Why Decoder-Only Won**:
- **Simplicity**: Single architecture for all tasks
- **Scaling**: Easier to scale than encoder-decoder
- **Flexibility**: Can handle any text-to-text task
- **Inference**: Efficient autoregressive generation

#### Architectural Innovations

**Layer Normalization Placement**:
```python
# Pre-norm (modern, more stable)
def transformer_block_pre_norm(x):
    x = x + attention(layer_norm(x))
    x = x + ffn(layer_norm(x))
    return x

# Post-norm (original, less stable at scale)
def transformer_block_post_norm(x):
    x = layer_norm(x + attention(x))
    x = layer_norm(x + ffn(x))
    return x
```

**RMSNorm vs LayerNorm**:
```python
# LayerNorm (traditional)
def layer_norm(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)

# RMSNorm (more efficient, used in LLaMA)
def rms_norm(x):
    rms = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
    return x / (rms + eps)
```

**Activation Functions**:
```python
# SwiGLU (used in LLaMA, PaLM)
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x

# GeGLU (used in some T5 variants)
def geglu(x):
    x, gate = x.chunk(2, dim=-1)
    return F.gelu(gate) * x
```

## 🎓 Training Phases

### Phase 1: Pre-training

#### Objective: Next Token Prediction
```
P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)
```

**Loss Function**:
```python
def next_token_loss(logits, targets):
    # Cross-entropy loss for each position
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        targets.view(-1), 
        ignore_index=pad_token_id
    )
    return loss
```

#### Data Sources & Curation
**Common Sources**:
- **Web crawls**: Common Crawl (filtered)
- **Books**: Project Gutenberg, book collections
- **News**: Reuters, news articles
- **Wikipedia**: High-quality encyclopedic content
- **Code**: GitHub repositories
- **Academic**: arXiv papers, academic texts

**Data Quality Filters**:
```python
def quality_filters(text):
    filters = [
        lambda x: len(x.split()) > 10,  # Minimum length
        lambda x: detect_language(x) == 'en',  # Language filter
        lambda x: not is_spam(x),  # Spam detection
        lambda x: toxicity_score(x) < 0.5,  # Toxicity filter
        lambda x: perplexity(x) < threshold,  # Quality via perplexity
    ]
    return all(f(text) for f in filters)
```

#### Training Dynamics
**Learning Rate Schedule**:
- **Warmup**: Linear increase for first 1-5% of training
- **Cosine decay**: Gradual decrease following cosine curve
- **Learning rate**: Peak at 1e-4 to 5e-4 for large models

**Batch Size Scaling**:
- **Start small**: 256-512 examples
- **Gradually increase**: Up to 4M+ tokens per batch
- **Dynamic batching**: Pack sequences efficiently

### Phase 2: Supervised Fine-Tuning (SFT)

#### Instruction Following Format
```json
{
  "instruction": "Explain quantum computing to a 5-year-old",
  "input": "",
  "output": "Quantum computing is like having a super special computer that can think about many different answers to a problem all at the same time..."
}
```

#### Dataset Types
**Instruction Datasets**:
- **Alpaca**: 52K instruction-following examples
- **Dolly**: 15K human-generated examples
- **FLAN**: Task-specific instruction datasets
- **ShareGPT**: Conversations from ChatGPT usage

**Training Process**:
```python
def sft_loss(model, batch):
    # Only compute loss on the output tokens
    input_ids = batch['input_ids']
    labels = batch['labels']  # -100 for input tokens, real tokens for output
    
    logits = model(input_ids).logits
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100  # Don't compute loss on input tokens
    )
    return loss
```

### Phase 3: Reinforcement Learning from Human Feedback (RLHF)

#### Step 1: Reward Model Training
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model.transformer
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        # Take last non-padded token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        rewards = self.reward_head(outputs.last_hidden_state)
        return rewards[range(len(rewards)), sequence_lengths]
```

**Preference Data Collection**:
```python
# Human preference data format
preference_data = {
    "prompt": "What is the capital of France?",
    "response_a": "Paris is the capital of France.",
    "response_b": "The capital of France is Paris, a beautiful city...",
    "preference": "b"  # Human prefers response B
}

# Bradley-Terry preference model
def preference_loss(reward_a, reward_b, preference):
    if preference == "a":
        return -F.logsigmoid(reward_a - reward_b)
    else:
        return -F.logsigmoid(reward_b - reward_a)
```

#### Step 2: PPO (Proximal Policy Optimization)
```python
def ppo_loss(policy_logprobs, old_logprobs, advantages, rewards, kl_penalty):
    # Policy gradient loss
    ratio = torch.exp(policy_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # Value loss (critic)
    value_loss = F.mse_loss(values, rewards)
    
    # KL divergence penalty to prevent too much deviation
    kl_loss = kl_penalty * (policy_logprobs - old_logprobs).mean()
    
    return policy_loss + value_loss + kl_loss
```

**RLHF Training Loop**:
1. **Generate**: Sample responses from current policy
2. **Score**: Get rewards from reward model
3. **Optimize**: Update policy using PPO
4. **Repeat**: Iterate until convergence

## 📈 Scaling Laws

### Chinchilla Scaling Laws
**Optimal Compute Allocation**:
```
N* ≈ (C/6)^0.5  # Optimal parameters
D* ≈ (C/6)^0.5  # Optimal tokens

where C = total compute budget (FLOPs)
```

**Key Insights**:
- **20:1 ratio**: ~20 tokens per parameter for optimal training
- **Most models are undertrained**: GPT-3 should have seen 4.3T tokens, not 300B
- **Compute scaling**: Both model size and data should scale with compute

### Emergent Abilities

**Capability Thresholds**:
- **10M parameters**: Basic language understanding
- **100M parameters**: Simple reasoning
- **1B parameters**: Few-shot learning emerges
- **10B parameters**: Complex reasoning begins
- **100B parameters**: Chain-of-thought reasoning
- **1T+ parameters**: Advanced reasoning, planning

**Examples of Emergent Abilities**:
```python
# Few-shot in-context learning (emerges ~1B params)
prompt = """
Translate English to French:
English: Hello
French: Bonjour

English: Goodbye  
French: Au revoir

English: Thank you
French: """
# Model outputs: "Merci"

# Chain-of-thought reasoning (emerges ~100B params)
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let me think step by step.
Roger starts with 5 tennis balls.
He buys 2 cans, each with 3 balls.
So he gets 2 × 3 = 6 more balls.
Total: 5 + 6 = 11 tennis balls.
"""
```

## 🏛️ Major LLM Architectures

### GPT Family
**GPT-1** (117M parameters):
- Proof of concept for transformer language modeling
- Unsupervised pre-training + supervised fine-tuning

**GPT-2** (1.5B parameters):
- "Language models are unsupervised multitask learners"
- Zero-shot task performance
- Initially withheld due to safety concerns

**GPT-3** (175B parameters):
- Few-shot in-context learning
- API-first release model
- Demonstrated scaling law benefits

**GPT-4** (~1.7T parameters, rumored):
- Multimodal capabilities (vision + text)
- Significantly improved reasoning
- More aligned and safer outputs

### LLaMA Family
**Key Innovations**:
- **RMSNorm**: More efficient normalization
- **SwiGLU**: Better activation function
- **RoPE**: Rotary positional embeddings
- **Efficient implementation**: Optimized for inference

**Model Sizes**:
- LLaMA-7B, 13B, 30B, 65B
- LLaMA-2: 7B, 13B, 70B
- Code Llama: Code-specialized variants

### PaLM (Pathways Language Model)
**Innovations**:
- **Pathways**: Efficient training infrastructure
- **540B parameters**: Larger than GPT-3
- **High-quality data**: Heavily filtered training corpus
- **Strong reasoning**: Excellent on math and reasoning tasks

### Claude (Constitutional AI)
**Training Process**:
1. **Initial training**: Similar to standard RLHF
2. **Constitutional AI**: Self-improvement through critique
3. **Harmlessness**: Strong focus on beneficial outputs

## 🎯 Advanced Training Techniques

### Curriculum Learning
```python
def curriculum_schedule(epoch, total_epochs):
    # Start with shorter sequences, gradually increase
    if epoch < total_epochs * 0.2:
        max_length = 512
    elif epoch < total_epochs * 0.5:
        max_length = 1024
    else:
        max_length = 2048
    return max_length
```

### Gradient Checkpointing
```python
# Trade computation for memory
def gradient_checkpointing_forward(model, x):
    # Only store activations at checkpoints
    # Recompute intermediate activations during backward
    with torch.utils.checkpoint.checkpoint(model.layer1, x) as x1:
        with torch.utils.checkpoint.checkpoint(model.layer2, x1) as x2:
            return model.layer3(x2)
```

### Mixed Precision Training
```python
# Use FP16 for forward pass, FP32 for gradients
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🔧 Production Optimization

### Model Compression
**Quantization**:
```python
# INT8 quantization
def quantize_weights(weights, scale, zero_point):
    return torch.clamp(
        torch.round(weights / scale + zero_point),
        0, 255
    ).to(torch.uint8)

# Dynamic quantization (inference time)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

**Knowledge Distillation**:
```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0):
    # Soft targets from teacher
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets (original labels)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return 0.7 * soft_loss + 0.3 * hard_loss
```

### Inference Optimization
**KV-Cache**:
```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, head_dim):
        self.cache_k = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.cache_v = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
    
    def update(self, xk, xv, start_pos):
        self.cache_k[:, :, start_pos:start_pos+xk.size(2)] = xk
        self.cache_v[:, :, start_pos:start_pos+xv.size(2)] = xv
        return self.cache_k, self.cache_v
```

**Speculative Decoding**:
```python
def speculative_decoding(large_model, small_model, input_ids, num_lookahead=4):
    # Small model generates multiple tokens quickly
    candidates = small_model.generate(input_ids, max_new_tokens=num_lookahead)
    
    # Large model verifies candidates in parallel
    logits = large_model(candidates)
    
    # Accept/reject based on probability ratios
    accepted_tokens = []
    for i, candidate in enumerate(candidates[len(input_ids):]):
        if should_accept(logits[i], candidate):
            accepted_tokens.append(candidate)
        else:
            break
    
    return accepted_tokens
```

## 🎯 Interview Questions & Answers

### Q1: "What are the key differences between training GPT and BERT?"

**Answer Framework**:
1. **Training objective**: 
   - GPT: Next token prediction (autoregressive)
   - BERT: Masked language modeling (bidirectional)
2. **Architecture**: 
   - GPT: Decoder-only with causal masking
   - BERT: Encoder-only with bidirectional attention
3. **Use cases**:
   - GPT: Generation, completion, few-shot learning
   - BERT: Classification, NER, question answering
4. **Inference**:
   - GPT: Sequential generation
   - BERT: Parallel processing of entire sequence

### Q2: "Explain the intuition behind RLHF. Why is it necessary?"

**Answer Framework**:
1. **Problem with standard training**: Models optimize for likelihood, not human preferences
2. **Human preference modeling**: Learn what humans actually want from AI
3. **Alignment**: Make model outputs more helpful, harmless, honest
4. **Process**: Reward model learns preferences → PPO optimizes policy against rewards
5. **Benefits**: More controlled outputs, better instruction following, reduced harmful content

### Q3: "How would you scale LLM training to 1 trillion parameters?"

**Answer Framework**:
1. **3D Parallelism**:
   - Data parallelism: Distribute batches across GPUs
   - Model parallelism: Split layers across GPUs
   - Pipeline parallelism: Different GPUs handle different stages
2. **Memory optimization**:
   - ZeRO: Shard optimizer states and gradients
   - Gradient checkpointing: Recompute activations
   - Mixed precision: FP16/BF16 training
3. **Infrastructure**:
   - High-bandwidth interconnect (NVLink, InfiniBand)
   - Efficient data loading and preprocessing
   - Fault tolerance and checkpointing
4. **Optimization**:
   - Learning rate scaling with batch size
   - Gradient clipping for stability
   - Careful initialization

### Q4: "What are the current limitations of LLMs and how might we address them?"

**Answer Framework**:
1. **Hallucination**: 
   - Problem: Generate plausible but false information
   - Solutions: Retrieval augmentation, uncertainty quantification, verification systems
2. **Context length**: 
   - Problem: Limited context window (2K-32K tokens)
   - Solutions: Hierarchical attention, memory mechanisms, retrieval
3. **Reasoning**: 
   - Problem: Limited logical reasoning capabilities
   - Solutions: Chain-of-thought, tool use, neuro-symbolic approaches
4. **Efficiency**: 
   - Problem: High computational cost
   - Solutions: Sparse models, distillation, efficient architectures
5. **Alignment**: 
   - Problem: Not always doing what humans want
   - Solutions: Better RLHF, constitutional AI, AI safety research

## 🚀 Research Frontiers

### Emerging Architectures
1. **Mixture of Experts (MoE)**: Sparse activation for scaling
2. **Retrieval-Augmented Generation**: External knowledge integration
3. **Multimodal transformers**: Vision, audio, text unified models
4. **Efficient attention**: Linear attention, sparse patterns

### Training Innovations
1. **Constitutional AI**: Self-improvement through critique
2. **Chain-of-thought training**: Explicit reasoning in training
3. **Tool-augmented training**: Learning to use external tools
4. **Federated learning**: Distributed training across organizations

### Evaluation & Safety
1. **Truthfulness**: TruthfulQA, fact-checking capabilities
2. **Alignment**: Helpful, harmless, honest evaluation
3. **Robustness**: Adversarial examples, distribution shift
4. **Interpretability**: Understanding model decision-making

---

## 📝 Study Checklist

- [ ] Understand the three phases of LLM training (pre-training, SFT, RLHF)
- [ ] Can explain scaling laws and emergent abilities
- [ ] Know architectural choices and their trade-offs
- [ ] Understand RLHF process and why it's important
- [ ] Can discuss production optimization techniques
- [ ] Know current limitations and research directions
- [ ] Familiar with major LLM families (GPT, LLaMA, PaLM, etc.)
- [ ] Can design training pipeline for large-scale LLM

**Next**: [Diffusion Models →](../01-core-concepts/03-diffusion-models.md)
