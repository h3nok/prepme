# Transformer Architecture Deep Dive

## 🎯 Learning Objectives
By the end of this section, you'll be able to:
- Explain the transformer architecture from memory
- Understand why transformers revolutionized NLP and beyond
- Compare different transformer variants and their use cases
- Implement key components of a transformer

## 🏗️ Architecture Overview

### The Big Picture
```
Input → Embedding → Positional Encoding → Encoder/Decoder Layers → Output
```

**Key Innovation**: Self-attention mechanism allows parallel processing and captures long-range dependencies better than RNNs.

## 🔍 Self-Attention Mechanism

### Mathematical Foundation
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Components Explained**:
- **Q (Query)**: "What am I looking for?" - represents the current position/token
- **K (Key)**: "What information is available?" - all positions in the sequence
- **V (Value)**: "What is the actual information?" - the content to be retrieved
- **√d_k scaling**: Prevents softmax saturation in high dimensions

### Intuitive Understanding
Think of attention as a **soft database lookup**:
1. **Query**: Your search term
2. **Keys**: Database indices  
3. **Values**: Actual data stored
4. **Attention weights**: How relevant each database entry is to your query

### Step-by-Step Process
1. **Linear Transformations**: Input embeddings → Q, K, V matrices
2. **Attention Scores**: Compute QK^T (how much each position attends to others)
3. **Scaling**: Divide by √d_k to prevent gradient vanishing
4. **Softmax**: Convert scores to probabilities (attention weights)
5. **Weighted Sum**: Multiply attention weights by values

## 🎯 Multi-Head Attention

### Why Multiple Heads?
Single attention head might focus on one type of relationship. Multiple heads can capture:
- **Syntactic relationships** (subject-verb agreement)
- **Semantic relationships** (word meanings)
- **Positional relationships** (nearby vs distant words)
- **Different abstraction levels**

### Mathematical Formulation
```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Implementation Details
- **Typical setup**: 8-16 heads for base models, 32+ for large models
- **Dimension per head**: d_model / h (e.g., 768/12 = 64 for BERT-base)
- **Parameter sharing**: Each head has its own W^Q, W^K, W^V matrices

## 📍 Positional Encoding

### The Problem
Transformers process all positions simultaneously → **no inherent sequence understanding**

### Sinusoidal Encoding (Original Transformer)
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- **Deterministic**: Same position always gets same encoding
- **Relative distances**: PE(pos+k) can be expressed as linear combination of PE(pos)
- **Extrapolation**: Can handle sequences longer than training

### Learned Positional Embeddings
- **Trainable parameters** for each position
- **Better performance** on fixed-length sequences
- **No extrapolation** beyond training length
- Used in BERT, GPT

### Advanced Positional Encodings
**Rotary Position Embedding (RoPE)**:
- Encodes positions as rotations in complex space
- Better length extrapolation
- Used in LLaMA, GPT-NeoX

**Alibi (Attention with Linear Biases)**:
- Adds position-dependent bias to attention scores
- Very good extrapolation properties
- Used in some recent models

## 🏗️ Transformer Variants

### 1. Encoder-Only (BERT-style)
```
Input → [CLS] token₁ token₂ ... [SEP] → Bidirectional attention → Outputs
```

**Characteristics**:
- **Bidirectional context**: Can see entire sequence
- **Masked Language Modeling**: Predict masked tokens
- **Use cases**: Classification, NER, question answering

**When to Use**:
- Need understanding of entire context
- Classification tasks
- Tasks where you have complete input upfront

### 2. Decoder-Only (GPT-style)
```
Input → token₁ token₂ ... → Causal attention (triangular mask) → Next token
```

**Characteristics**:
- **Causal masking**: Can only see previous tokens
- **Autoregressive generation**: Predict next token
- **Use cases**: Text generation, completion, dialogue

**When to Use**:
- Text generation tasks
- Need streaming/online processing
- Want single model for multiple tasks

### 3. Encoder-Decoder (T5-style)
```
Encoder: Input → Bidirectional attention → Context
Decoder: Output tokens → Causal attention + Cross-attention to encoder → Generation
```

**Characteristics**:
- **Separate encoding/decoding**: Different objectives
- **Cross-attention**: Decoder attends to encoder outputs
- **Use cases**: Translation, summarization, seq2seq tasks

**When to Use**:
- Input and output are different modalities/languages
- Clear separation between understanding and generation phases
- Complex structured outputs

## 🔧 Key Architectural Components

### Layer Normalization
```
LayerNorm(x) = γ * (x - μ) / σ + β
```

**Pre-norm vs Post-norm**:
- **Pre-norm**: LayerNorm → Attention/FFN → Residual (more stable training)
- **Post-norm**: Attention/FFN → LayerNorm → Residual (original design)

**RMSNorm** (used in LLaMA):
- Removes mean centering: RMS(x) = √(Σx²/n)
- Faster computation, similar performance

### Feed-Forward Networks
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Modern Variants**:
- **SwiGLU**: Swish activation + Gated Linear Unit
- **GeGLU**: GELU activation + GLU
- **Typically 4x wider** than attention dimension

### Residual Connections
```
output = LayerNorm(x + SubLayer(x))
```

**Benefits**:
- **Gradient flow**: Enables training very deep networks
- **Identity mapping**: Model can learn to ignore layers if needed
- **Stability**: Reduces internal covariate shift

## 📊 Scaling Laws & Model Sizes

### Parameter Scaling
| Model | Parameters | Layers | Hidden Size | Attention Heads |
|-------|------------|--------|-------------|-----------------|
| BERT-base | 110M | 12 | 768 | 12 |
| BERT-large | 340M | 24 | 1024 | 16 |
| GPT-3 | 175B | 96 | 12288 | 96 |
| GPT-4 | ~1.7T | ~120 | ~18432 | ~128 |

### Compute Scaling
**Training FLOPs**: ≈ 6 × Parameters × Tokens
**Inference FLOPs**: ≈ 2 × Parameters × Generated tokens

### Memory Requirements
**Training**: 4 × Parameters (weights + gradients + optimizer states)
**Inference**: 2 × Parameters (weights + activations)

## 🎯 Interview Questions & Answers

### Q1: "Why did transformers replace RNNs for many NLP tasks?"

**Answer Framework**:
1. **Parallelization**: RNNs process sequentially, transformers process all positions simultaneously
2. **Long-range dependencies**: Attention provides direct connections between any two positions
3. **Training efficiency**: Parallel processing enables faster training on modern hardware
4. **Scalability**: Architecture scales better to larger models and datasets
5. **Transfer learning**: Pre-trained transformers transfer better across tasks

### Q2: "Explain the intuition behind attention weights"

**Answer Framework**:
1. **Relevance scoring**: How relevant is each position to the current query
2. **Soft selection**: Instead of hard selection, we take weighted average
3. **Context aggregation**: Combines information from multiple relevant positions
4. **Learned associations**: Model learns what to attend to during training

### Q3: "What are the computational bottlenecks in transformers?"

**Answer Framework**:
1. **Attention complexity**: O(n²) memory and computation with sequence length
2. **Memory bandwidth**: Moving large matrices between memory and compute
3. **Sequence length scaling**: Quadratic scaling limits very long sequences
4. **Solutions**: Sparse attention, linear attention, Flash Attention, chunking

### Q4: "How would you modify a transformer for very long sequences?"

**Answer Framework**:
1. **Sparse attention patterns**: Only attend to subset of positions
2. **Sliding window attention**: Local attention with some global connections
3. **Hierarchical attention**: Attend at multiple resolutions
4. **Memory mechanisms**: External memory for very long contexts
5. **Retrieval augmentation**: Retrieve relevant segments instead of processing all

## 🔨 Implementation Insights

### Efficient Attention Computation
```python
# Flash Attention insight: recompute attention on-the-fly to save memory
def flash_attention(Q, K, V, block_size):
    # Process in blocks to fit in SRAM
    # Recompute attention weights instead of storing them
    # Achieves O(n²) compute but O(n) memory
```

### Multi-Head Attention Tricks
```python
# Parallel computation of all heads
def multi_head_attention(x, W_qkv, W_o, num_heads):
    # Single matrix multiplication for all Q, K, V
    qkv = x @ W_qkv  # [batch, seq, 3 * hidden]
    
    # Reshape and transpose for parallel head computation
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    # ... similar for k, v
```

### Position Encoding Implementation
```python
def sinusoidal_pos_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    return pe
```

## 🚀 Advanced Topics for Senior Role

### Architectural Innovations
1. **Mixture of Experts (MoE)**: Sparse activation of expert networks
2. **Switch Transformer**: Simple and efficient MoE routing
3. **GLaM**: Sparsely activated language model
4. **PaLM**: Pathways Language Model with improved training

### Optimization Techniques
1. **Gradient checkpointing**: Trade compute for memory
2. **Mixed precision**: FP16 forward, FP32 gradients
3. **ZeRO**: Optimizer state sharding across devices
4. **3D parallelism**: Data + model + pipeline parallelism

### Research Directions
1. **Linear attention**: Reducing quadratic complexity
2. **Retrieval augmentation**: External memory mechanisms
3. **Multimodal transformers**: Vision, audio, text integration
4. **Efficient architectures**: MobileBERT, DistilBERT, TinyBERT

---

## 📝 Study Checklist

- [ ] Can draw transformer architecture from memory
- [ ] Understand attention mechanism mathematically and intuitively
- [ ] Know when to use encoder-only vs decoder-only vs encoder-decoder
- [ ] Understand positional encoding options and trade-offs
- [ ] Can discuss scaling laws and computational requirements
- [ ] Know recent architectural innovations and optimizations
- [ ] Can implement basic transformer components
- [ ] Understand production deployment considerations

**Next**: [Large Language Models →](../01-core-concepts/02-large-language-models.md)
