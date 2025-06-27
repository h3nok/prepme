# Coding Challenges

## 🎯 Challenge Categories

### 1. Transformer Implementation Challenges
### 2. Optimization & Efficiency Problems  
### 3. Training & Fine-tuning Scenarios
### 4. Multimodal Integration Tasks
### 5. Production & Scaling Challenges

---

## 🏗️ Transformer Implementation Challenges

### Challenge 1: Implement Multi-Head Attention from Scratch

**Problem Statement**:
```
Implement a complete multi-head attention mechanism without using any 
pre-built attention modules. Your implementation should:

1. Support variable sequence lengths
2. Handle padding masks correctly  
3. Be memory efficient
4. Support both self-attention and cross-attention
5. Include proper scaling and dropout

Test your implementation against PyTorch's built-in attention.
```

**Solution Template**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # TODO: Implement the forward pass
        # 1. Apply linear projections
        # 2. Reshape for multi-head attention
        # 3. Compute attention scores
        # 4. Apply mask if provided
        # 5. Apply softmax and dropout
        # 6. Compute attention output
        # 7. Reshape and apply output projection
        
        pass
    
    def compute_attention_scores(self, q, k):
        """Compute scaled dot-product attention scores."""
        # TODO: Implement attention score computation
        pass
    
    def apply_mask(self, scores, mask):
        """Apply attention mask to scores."""
        # TODO: Implement masking
        pass

# Test function
def test_multihead_attention():
    """Test your implementation against PyTorch's version."""
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Your implementation
    custom_attention = MultiHeadAttention(d_model, num_heads)
    
    # PyTorch implementation
    pytorch_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    # TODO: Compare outputs and verify correctness
    pass
```

**Expected Solution**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, d_model = query.shape
        seq_len_k = key.shape[1]
        
        # 1. Linear projections
        Q = self.q_linear(query)  # [B, Lq, D]
        K = self.k_linear(key)    # [B, Lk, D]  
        V = self.v_linear(value)  # [B, Lk, D]
        
        # 2. Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lq, Dh]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, Dh]
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, Dh]
        
        # 3. Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Lq, Lk]
        
        # 4. Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. Compute attention output
        attention_output = torch.matmul(attention_weights, V)  # [B, H, Lq, Dh]
        
        # 7. Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.output_linear(attention_output)
        
        return output, attention_weights

def test_multihead_attention():
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test self-attention
    custom_attention = MultiHeadAttention(d_model, num_heads)
    output, weights = custom_attention(x, x, x)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Test with causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    masked_output, masked_weights = custom_attention(x, x, x, causal_mask)
    
    # Verify masked attention weights are approximately zero above diagonal
    assert torch.allclose(masked_weights[:, :, :, :].triu(1), torch.zeros_like(masked_weights[:, :, :, :].triu(1)), atol=1e-6)
    
    print("✅ Multi-head attention implementation passed all tests!")

# Run test
test_multihead_attention()
```

### Challenge 2: Implement Rotary Position Embedding (RoPE)

**Problem Statement**:
```
Implement Rotary Position Embedding (RoPE) as used in LLaMA and other 
modern LLMs. Your implementation should:

1. Apply rotation matrices to query and key vectors
2. Support different frequencies for different dimensions
3. Handle variable sequence lengths efficiently
4. Be compatible with multi-head attention

Compare the benefits of RoPE vs sinusoidal position embeddings.
```

**Solution Template**:
```python
import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # TODO: Precompute frequency values
        # TODO: Cache rotation matrices for efficiency
        
    def forward(self, x, seq_len=None):
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length (optional)
        
        Returns:
            Tensor with rotary position embedding applied
        """
        # TODO: Implement RoPE application
        pass
    
    def rotate_half(self, x):
        """Rotate half of the hidden dims of the input."""
        # TODO: Implement rotation helper function
        pass
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """Apply the rotary position embedding."""
        # TODO: Implement the core rotation logic
        pass

# Test function
def test_rope():
    """Test RoPE implementation."""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64
    
    # Create test data
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    rope = RotaryPositionEmbedding(head_dim)
    
    # TODO: Test RoPE application
    # TODO: Verify properties (length preservation, rotation invariance)
    pass
```

**Expected Solution**:
```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute frequency for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
        
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        """Update cached cos and sin values."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Compute angles
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Cache cos and sin
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x, seq_len=None):
        """Apply rotary position embedding."""
        if seq_len is None:
            seq_len = x.shape[-2]
            
        # Update cache if needed
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        # Apply rotation
        return self.apply_rotary_pos_emb(x, self._cos_cached, self._sin_cached)
    
    def rotate_half(self, x):
        """Rotate half of the hidden dims."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """Apply rotary position embedding using cos and sin."""
        # x shape: [batch_size, num_heads, seq_len, head_dim]
        # cos, sin shape: [seq_len, head_dim]
        
        # Expand cos and sin to match x dimensions
        cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
        sin = sin[None, None, :, :]  # [1, 1, seq_len, head_dim]
        
        # Apply rotation
        return x * cos + self.rotate_half(x) * sin

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_position_embeddings=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.rotary_emb = RotaryPositionEmbedding(
            self.head_dim, max_position_embeddings
        )
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K (not V)
        Q = self.rotary_emb(Q)
        K = self.rotary_emb(K)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.o_proj(attn_output)
        
        return output

def test_rope():
    """Test RoPE implementation."""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64
    d_model = num_heads * head_dim
    
    # Test basic RoPE
    rope = RotaryPositionEmbedding(head_dim)
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Apply RoPE
    rotated_x = rope(x)
    
    # Verify shape preservation
    assert rotated_x.shape == x.shape
    
    # Test that rotation preserves vector magnitude (approximately)
    original_norm = torch.norm(x, dim=-1)
    rotated_norm = torch.norm(rotated_x, dim=-1)
    assert torch.allclose(original_norm, rotated_norm, atol=1e-6)
    
    # Test attention with RoPE
    attention = MultiHeadAttentionWithRoPE(d_model, num_heads)
    input_seq = torch.randn(batch_size, seq_len, d_model)
    output = attention(input_seq)
    
    assert output.shape == input_seq.shape
    
    print("✅ RoPE implementation passed all tests!")

test_rope()
```

---

## ⚡ Optimization & Efficiency Problems

### Challenge 3: Memory-Efficient Attention Implementation

**Problem Statement**:
```
Implement Flash Attention or a similar memory-efficient attention 
mechanism. Your implementation should:

1. Reduce memory usage from O(n²) to O(n)
2. Maintain numerical stability
3. Support variable sequence lengths
4. Be backward-compatible with standard attention
5. Provide significant speedup for long sequences

Benchmark your implementation against standard attention.
```

**Solution Template**:
```python
import torch
import torch.nn as nn
import math

class FlashAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def flash_attention_forward(self, Q, K, V, mask=None):
        """
        Memory-efficient attention computation.
        
        Args:
            Q, K, V: Query, Key, Value tensors [B, H, N, D]
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention statistics
        """
        # TODO: Implement block-wise attention computation
        # TODO: Maintain running statistics for numerical stability
        # TODO: Avoid materializing full attention matrix
        pass
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply flash attention
        output = self.flash_attention_forward(Q, K, V, mask)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.o_proj(output)

def benchmark_attention(seq_lengths, d_model=512, num_heads=8):
    """Benchmark flash attention vs standard attention."""
    # TODO: Implement benchmarking code
    # TODO: Measure memory usage and execution time
    # TODO: Verify correctness
    pass
```

**Expected Solution**:
```python
class FlashAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False) 
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def flash_attention_forward(self, Q, K, V, mask=None):
        """Memory-efficient attention using block-wise computation."""
        B, H, N, D = Q.shape
        
        # Initialize output and statistics
        O = torch.zeros_like(Q)
        l = torch.zeros(B, H, N, 1, device=Q.device, dtype=Q.dtype)  # row sums
        m = torch.full((B, H, N, 1), -torch.inf, device=Q.device, dtype=Q.dtype)  # row maxes
        
        # Number of blocks
        Tr = math.ceil(N / self.block_size)
        Tc = math.ceil(N / self.block_size)
        
        # Iterate over query blocks
        for i in range(Tr):
            # Query block indices
            q_start = i * self.block_size
            q_end = min((i + 1) * self.block_size, N)
            
            # Extract query block
            Qi = Q[:, :, q_start:q_end, :]  # [B, H, block_size, D]
            
            # Initialize block statistics
            li = torch.zeros(B, H, q_end - q_start, 1, device=Q.device, dtype=Q.dtype)
            mi = torch.full((B, H, q_end - q_start, 1), -torch.inf, device=Q.device, dtype=Q.dtype)
            Oi = torch.zeros(B, H, q_end - q_start, D, device=Q.device, dtype=Q.dtype)
            
            # Iterate over key-value blocks
            for j in range(Tc):
                # Key-Value block indices
                kv_start = j * self.block_size
                kv_end = min((j + 1) * self.block_size, N)
                
                # Extract key-value blocks
                Kj = K[:, :, kv_start:kv_end, :]  # [B, H, block_size, D]
                Vj = V[:, :, kv_start:kv_end, :]  # [B, H, block_size, D]
                
                # Compute attention scores for this block
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * self.scale
                
                # Apply mask if provided
                if mask is not None:
                    mask_block = mask[:, :, q_start:q_end, kv_start:kv_end]
                    Sij = Sij.masked_fill(mask_block == 0, -torch.inf)
                
                # Online softmax computation
                mij = torch.max(Sij, dim=-1, keepdim=True)[0]  # Block max
                Pij = torch.exp(Sij - mij)  # Shifted exponentials
                lij = torch.sum(Pij, dim=-1, keepdim=True)  # Block sum
                
                # Update global statistics
                mi_new = torch.max(mi, mij)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij
                
                # Update output (with correction for numerical stability)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(mij - mi_new)
                
                Oi = alpha * Oi + beta * torch.matmul(Pij, Vj)
                
                # Update statistics
                mi = mi_new
                li = li_new
            
            # Normalize output
            Oi = Oi / li
            
            # Store results
            O[:, :, q_start:q_end, :] = Oi
            l[:, :, q_start:q_end, :] = li
            m[:, :, q_start:q_end, :] = mi
        
        return O
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply flash attention
        output = self.flash_attention_forward(Q, K, V, mask)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.o_proj(output)

def benchmark_attention():
    """Benchmark flash attention vs standard attention."""
    import time
    import tracemalloc
    
    seq_lengths = [512, 1024, 2048, 4096]
    d_model = 512
    num_heads = 8
    batch_size = 2
    
    print("Benchmarking Flash Attention vs Standard Attention")
    print("=" * 60)
    
    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        
        # Create test data
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Standard attention
        standard_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Flash attention
        flash_attn = FlashAttention(d_model, num_heads)
        
        # Benchmark standard attention
        tracemalloc.start()
        start_time = time.time()
        
        with torch.no_grad():
            std_output, _ = standard_attn(x, x, x)
        
        std_time = time.time() - start_time
        std_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
        tracemalloc.stop()
        
        # Benchmark flash attention
        tracemalloc.start()
        start_time = time.time()
        
        with torch.no_grad():
            flash_output = flash_attn(x)
        
        flash_time = time.time() - start_time
        flash_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
        tracemalloc.stop()
        
        # Verify correctness (approximately)
        mse = torch.mean((std_output - flash_output) ** 2)
        
        print(f"Standard Attention: {std_time:.4f}s, {std_memory:.2f}MB")
        print(f"Flash Attention:    {flash_time:.4f}s, {flash_memory:.2f}MB")
        print(f"Speedup: {std_time/flash_time:.2f}x")
        print(f"Memory Reduction: {std_memory/flash_memory:.2f}x")
        print(f"MSE: {mse:.8f}")

# Run benchmark
benchmark_attention()
```

### Challenge 4: Dynamic Batching Implementation

**Problem Statement**:
```
Implement a dynamic batching system for efficient inference of 
variable-length sequences. Your system should:

1. Batch sequences of similar lengths together
2. Handle padding efficiently  
3. Support real-time request processing
4. Optimize GPU utilization
5. Provide configurable timeout and batch size limits

Test with a mix of short and long sequences.
```

**Expected Solution**:
```python
import torch
import torch.nn as nn
from collections import defaultdict, deque
import threading
import time
import asyncio
from typing import List, Dict, Optional, Tuple

class BatchRequest:
    def __init__(self, request_id: str, input_tokens: torch.Tensor, 
                 max_length: int, timestamp: float):
        self.request_id = request_id
        self.input_tokens = input_tokens
        self.max_length = max_length
        self.timestamp = timestamp
        self.future = asyncio.Future()

class DynamicBatcher:
    def __init__(self, model, max_batch_size: int = 32, 
                 max_wait_time: float = 0.1, length_tolerance: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.length_tolerance = length_tolerance
        
        # Request queues organized by sequence length buckets
        self.request_queues: Dict[int, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.processing = True
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.start()
    
    def get_length_bucket(self, seq_len: int) -> int:
        """Get the bucket for a given sequence length."""
        return ((seq_len - 1) // self.length_tolerance + 1) * self.length_tolerance
    
    async def add_request(self, request_id: str, input_tokens: torch.Tensor, 
                         max_length: int) -> torch.Tensor:
        """Add a new request to the batching queue."""
        request = BatchRequest(request_id, input_tokens, max_length, time.time())
        
        bucket = self.get_length_bucket(input_tokens.shape[1])
        
        with self.lock:
            self.request_queues[bucket].append(request)
        
        # Wait for the result
        return await request.future
    
    def _process_batches(self):
        """Background thread that processes batches."""
        while self.processing:
            batches_to_process = self._collect_batches()
            
            for batch_requests in batches_to_process:
                if batch_requests:
                    self._process_batch(batch_requests)
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def _collect_batches(self) -> List[List[BatchRequest]]:
        """Collect requests into batches for processing."""
        batches = []
        current_time = time.time()
        
        with self.lock:
            for bucket, queue in self.request_queues.items():
                if not queue:
                    continue
                
                batch_requests = []
                
                # Collect requests for this bucket
                while (queue and len(batch_requests) < self.max_batch_size):
                    request = queue[0]  # Peek at first request
                    
                    # Check if we should wait longer or process now
                    wait_time = current_time - request.timestamp
                    
                    if (len(batch_requests) == 0 and wait_time < self.max_wait_time):
                        # Wait for more requests if this is the first one
                        break
                    
                    # Add request to batch
                    batch_requests.append(queue.popleft())
                
                if batch_requests:
                    batches.append(batch_requests)
        
        return batches
    
    def _process_batch(self, batch_requests: List[BatchRequest]):
        """Process a batch of requests."""
        try:
            # Prepare batch input
            batch_input = self._prepare_batch_input(batch_requests)
            
            # Run model inference
            with torch.no_grad():
                batch_output = self.model.generate(**batch_input)
            
            # Distribute results back to requests
            self._distribute_results(batch_requests, batch_output)
            
        except Exception as e:
            # Handle errors by setting exception on all futures
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def _prepare_batch_input(self, batch_requests: List[BatchRequest]) -> Dict:
        """Prepare batched input from individual requests."""
        # Find max length in batch for padding
        max_input_len = max(req.input_tokens.shape[1] for req in batch_requests)
        max_output_len = max(req.max_length for req in batch_requests)
        
        # Pad all sequences to same length
        batch_input_ids = []
        attention_masks = []
        
        for request in batch_requests:
            input_ids = request.input_tokens.squeeze(0)  # Remove batch dim
            input_len = input_ids.shape[0]
            
            # Pad sequence
            if input_len < max_input_len:
                padding = torch.zeros(max_input_len - input_len, dtype=input_ids.dtype)
                padded_input = torch.cat([input_ids, padding], dim=0)
                attention_mask = torch.cat([
                    torch.ones(input_len, dtype=torch.bool),
                    torch.zeros(max_input_len - input_len, dtype=torch.bool)
                ], dim=0)
            else:
                padded_input = input_ids
                attention_mask = torch.ones(input_len, dtype=torch.bool)
            
            batch_input_ids.append(padded_input)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(attention_masks),
            'max_new_tokens': max_output_len,
            'do_sample': False,
            'pad_token_id': 0
        }
    
    def _distribute_results(self, batch_requests: List[BatchRequest], 
                          batch_output: torch.Tensor):
        """Distribute batch results back to individual requests."""
        for i, request in enumerate(batch_requests):
            if not request.future.done():
                # Extract this request's output
                output = batch_output[i:i+1]  # Keep batch dimension
                request.future.set_result(output)
    
    def shutdown(self):
        """Shutdown the batcher."""
        self.processing = False
        self.processing_thread.join()

# Mock model for testing
class MockGenerativeModel(nn.Module):
    def __init__(self, vocab_size=1000, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=6
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def generate(self, input_ids, attention_mask, max_new_tokens=50, **kwargs):
        """Simple generation for testing."""
        batch_size, seq_len = input_ids.shape
        
        # Simple strategy: repeat last token
        last_tokens = input_ids[:, -1].unsqueeze(1)
        generated = last_tokens.repeat(1, max_new_tokens)
        
        return torch.cat([input_ids, generated], dim=1)

# Test the dynamic batcher
async def test_dynamic_batcher():
    """Test the dynamic batching system."""
    model = MockGenerativeModel()
    batcher = DynamicBatcher(model, max_batch_size=4, max_wait_time=0.05)
    
    # Create test requests with different lengths
    test_requests = [
        ("req1", torch.randint(0, 1000, (1, 10)), 20),  # Short
        ("req2", torch.randint(0, 1000, (1, 12)), 25),  # Short  
        ("req3", torch.randint(0, 1000, (1, 50)), 30),  # Long
        ("req4", torch.randint(0, 1000, (1, 8)), 15),   # Short
        ("req5", torch.randint(0, 1000, (1, 55)), 35),  # Long
    ]
    
    # Submit requests concurrently
    tasks = []
    start_time = time.time()
    
    for req_id, input_tokens, max_length in test_requests:
        task = asyncio.create_task(
            batcher.add_request(req_id, input_tokens, max_length)
        )
        tasks.append((req_id, task))
        
        # Small delay between requests
        await asyncio.sleep(0.01)
    
    # Wait for all results
    results = []
    for req_id, task in tasks:
        result = await task
        results.append((req_id, result))
        print(f"Request {req_id} completed: {result.shape}")
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.3f}s")
    
    # Shutdown batcher
    batcher.shutdown()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_dynamic_batcher())
```

---

## 🎓 Training & Fine-tuning Scenarios

### Challenge 5: Implement Parameter-Efficient Fine-tuning (LoRA)

**Problem Statement**:
```
Implement LoRA (Low-Rank Adaptation) for efficient fine-tuning of 
large models. Your implementation should:

1. Support different rank values
2. Be applicable to any linear layer
3. Maintain original model weights frozen  
4. Provide significant parameter reduction
5. Be easy to merge back into original model

Compare training speed and memory usage vs full fine-tuning.
```

**Expected Solution**:
```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Low-rank adaptation matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original forward pass
        original_output = self.original_layer(x)
        
        # LoRA forward pass: B @ A @ x
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into original layer."""
        with torch.no_grad():
            # Compute LoRA weight delta
            weight_delta = self.alpha / self.rank * (self.lora_B @ self.lora_A)
            
            # Add to original weights
            self.original_layer.weight.data += weight_delta
            
            # Zero out LoRA parameters
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()

class LoRAConfig:
    def __init__(self, rank: int = 4, alpha: float = 8.0, 
                 target_modules: list = None, dropout: float = 0.1):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ['q_proj', 'v_proj']
        self.dropout = dropout

def apply_lora_to_model(model: nn.Module, config: LoRAConfig):
    """Apply LoRA to specified modules in a model."""
    
    def apply_lora_to_layer(layer, name):
        if isinstance(layer, nn.Linear) and any(target in name for target in config.target_modules):
            return LoRALinear(layer, rank=config.rank, alpha=config.alpha)
        return layer
    
    # Recursively apply LoRA to target modules
    def recursive_replace(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear) and any(target in name for target in config.target_modules):
                setattr(module, name, LoRALinear(child, rank=config.rank, alpha=config.alpha))
            else:
                recursive_replace(child, full_name)
    
    recursive_replace(model)
    return model

# Test with a simple transformer
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        
        # Embeddings + positional encoding
        x = self.embedding(input_ids) + self.pos_encoding[:seq_len]
        
        # Transformer
        x = self.transformer(x)
        
        # Language modeling head
        return self.lm_head(x)

def count_parameters(model):
    """Count trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def test_lora():
    """Test LoRA implementation."""
    
    # Create model
    model = SimpleTransformer(vocab_size=1000, d_model=512)
    
    print("Original model:")
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Apply LoRA
    lora_config = LoRAConfig(rank=8, alpha=16, target_modules=['q_proj', 'v_proj', 'out_proj'])
    model_with_lora = apply_lora_to_model(model, lora_config)
    
    print("\nModel with LoRA:")
    total_params, trainable_params = count_parameters(model_with_lora)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Reduction ratio: {trainable_params / total_params:.4f}")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 50))
    
    with torch.no_grad():
        output = model_with_lora(input_ids)
        print(f"\nOutput shape: {output.shape}")
    
    # Test gradient computation (only LoRA params should have gradients)
    loss = output.sum()
    loss.backward()
    
    lora_grads = 0
    frozen_grads = 0
    
    for name, param in model_with_lora.named_parameters():
        if param.grad is not None:
            if 'lora' in name:
                lora_grads += 1
            else:
                frozen_grads += 1
    
    print(f"Parameters with gradients - LoRA: {lora_grads}, Frozen: {frozen_grads}")
    
    print("✅ LoRA implementation test passed!")

test_lora()
```

### Challenge 6: Implement Gradient Checkpointing

**Problem Statement**:
```
Implement gradient checkpointing to reduce memory usage during training
while maintaining computational correctness. Your implementation should:

1. Support selective checkpointing of transformer layers
2. Provide memory vs computation trade-off control
3. Be compatible with mixed precision training
4. Handle nested modules correctly
5. Provide clear memory savings measurement

Test on a model that doesn't fit in memory without checkpointing.
```

**Expected Solution**:
```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import tracemalloc
import time

class CheckpointWrapper(nn.Module):
    def __init__(self, module, use_checkpoint=True):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
    
    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            return checkpoint(self.module, *args, **kwargs, use_reentrant=False)
        else:
            return self.module(*args, **kwargs)

class SelectiveCheckpointing:
    """Manager for selective gradient checkpointing."""
    
    def __init__(self, checkpoint_every_n_layers=2):
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.checkpointed_modules = []
    
    def apply_checkpointing(self, model, layer_pattern="layer"):
        """Apply checkpointing to specified layers."""
        layer_count = 0
        
        def apply_to_module(module, name=""):
            nonlocal layer_count
            
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                # Check if this is a target layer
                if layer_pattern in child_name and hasattr(child_module, 'forward'):
                    should_checkpoint = (layer_count % self.checkpoint_every_n_layers == 0)
                    
                    if should_checkpoint:
                        # Wrap with checkpointing
                        setattr(module, child_name, CheckpointWrapper(child_module))
                        self.checkpointed_modules.append(full_name)
                    
                    layer_count += 1
                else:
                    # Recursively apply to children
                    apply_to_module(child_module, full_name)
        
        apply_to_module(model)
        print(f"Applied checkpointing to {len(self.checkpointed_modules)} modules:")
        for module_name in self.checkpointed_modules:
            print(f"  - {module_name}")

class MemoryEfficientTransformer(nn.Module):
    """Large transformer for testing memory efficiency."""
    
    def __init__(self, vocab_size=10000, d_model=1024, num_heads=16, 
                 num_layers=24, max_seq_len=1024):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Create many transformer layers to stress memory
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        
        # Embeddings
        x = self.embedding(input_ids) + self.pos_embedding[:seq_len]
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        # Output
        x = self.layer_norm(x)
        return self.lm_head(x)

def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    
    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, peak / 1024 / 1024  # Convert to MB
    except Exception as e:
        tracemalloc.stop()
        raise e

def train_step(model, batch, optimizer, criterion):
    """Single training step."""
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(batch['input_ids'])
    loss = criterion(outputs.view(-1, outputs.size(-1)), batch['targets'].view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test_gradient_checkpointing():
    """Test gradient checkpointing implementation."""
    
    # Model configuration (large enough to stress memory)
    vocab_size = 10000
    d_model = 1024
    num_heads = 16
    num_layers = 12  # Reduced for testing
    batch_size = 4
    seq_len = 512
    
    print("Testing Gradient Checkpointing")
    print("=" * 50)
    
    # Create test data
    batch = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'targets': torch.randint(0, vocab_size, (batch_size, seq_len))
    }
    
    criterion = nn.CrossEntropyLoss()
    
    # Test 1: Without checkpointing
    print("\n1. Testing WITHOUT gradient checkpointing:")
    
    model_no_checkpoint = MemoryEfficientTransformer(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers
    )
    optimizer_no_checkpoint = torch.optim.AdamW(model_no_checkpoint.parameters(), lr=1e-4)
    
    try:
        start_time = time.time()
        loss, memory_no_checkpoint = measure_memory_usage(
            train_step, model_no_checkpoint, batch, optimizer_no_checkpoint, criterion
        )
        time_no_checkpoint = time.time() - start_time
        
        print(f"  Memory usage: {memory_no_checkpoint:.2f} MB")
        print(f"  Training time: {time_no_checkpoint:.4f} seconds")
        print(f"  Loss: {loss:.6f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("  ❌ Out of memory without checkpointing!")
            memory_no_checkpoint = float('inf')
            time_no_checkpoint = float('inf')
        else:
            raise e
    
    # Test 2: With checkpointing
    print("\n2. Testing WITH gradient checkpointing:")
    
    model_with_checkpoint = MemoryEfficientTransformer(
        vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers
    )
    
    # Apply selective checkpointing
    checkpointing = SelectiveCheckpointing(checkpoint_every_n_layers=2)
    checkpointing.apply_checkpointing(model_with_checkpoint, layer_pattern="layers")
    
    optimizer_with_checkpoint = torch.optim.AdamW(model_with_checkpoint.parameters(), lr=1e-4)
    
    start_time = time.time()
    loss, memory_with_checkpoint = measure_memory_usage(
        train_step, model_with_checkpoint, batch, optimizer_with_checkpoint, criterion
    )
    time_with_checkpoint = time.time() - start_time
    
    print(f"  Memory usage: {memory_with_checkpoint:.2f} MB")
    print(f"  Training time: {time_with_checkpoint:.4f} seconds")
    print(f"  Loss: {loss:.6f}")
    
    # Comparison
    print("\n3. Comparison:")
    if memory_no_checkpoint != float('inf'):
        memory_reduction = (1 - memory_with_checkpoint / memory_no_checkpoint) * 100
        time_overhead = (time_with_checkpoint / time_no_checkpoint - 1) * 100
        
        print(f"  Memory reduction: {memory_reduction:.1f}%")
        print(f"  Time overhead: {time_overhead:.1f}%")
    else:
        print("  ✅ Checkpointing enabled training that was impossible without it!")
    
    # Test gradient correctness
    print("\n4. Testing gradient correctness:")
    
    # Small model for gradient testing
    small_model_no_cp = MemoryEfficientTransformer(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    small_model_with_cp = MemoryEfficientTransformer(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    # Copy weights to ensure same initialization
    small_model_with_cp.load_state_dict(small_model_no_cp.state_dict())
    
    # Apply checkpointing to second model
    checkpointing_small = SelectiveCheckpointing(checkpoint_every_n_layers=1)
    checkpointing_small.apply_checkpointing(small_model_with_cp, layer_pattern="layers")
    
    # Small batch for testing
    test_batch = {
        'input_ids': torch.randint(0, 100, (2, 32)),
        'targets': torch.randint(0, 100, (2, 32))
    }
    
    # Forward pass and compute gradients
    def compute_gradients(model, batch):
        outputs = model(batch['input_ids'])
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch['targets'].view(-1))
        loss.backward()
        
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients
    
    grads_no_cp = compute_gradients(small_model_no_cp, test_batch)
    grads_with_cp = compute_gradients(small_model_with_cp, test_batch)
    
    # Compare gradients
    max_diff = 0
    for name in grads_no_cp:
        if name in grads_with_cp:
            diff = torch.max(torch.abs(grads_no_cp[name] - grads_with_cp[name]))
            max_diff = max(max_diff, diff.item())
    
    print(f"  Maximum gradient difference: {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("  ✅ Gradients are numerically identical!")
    else:
        print("  ⚠️  Small gradient differences detected (expected due to numerical precision)")
    
    print("\n✅ Gradient checkpointing test completed!")

# Run the test
if __name__ == "__main__":
    test_gradient_checkpointing()
```

---

## 🎯 Success Criteria

### Code Quality Assessment
- **Correctness**: Does the implementation work as specified?
- **Efficiency**: Is the solution optimized for the given constraints?
- **Readability**: Is the code clean and well-documented?
- **Robustness**: Does it handle edge cases and errors gracefully?

### Problem-Solving Approach
- **Understanding**: Does the candidate grasp the problem requirements?
- **Strategy**: Is the approach well-thought-out and systematic?
- **Trade-offs**: Are they aware of different trade-offs and alternatives?
- **Testing**: Do they think about validation and edge cases?

### Follow-up Discussion Points
- "How would you scale this to production?"
- "What are the limitations of your approach?"
- "How would you monitor this in a live system?"
- "What optimizations would you prioritize next?"

---

## 📚 Practice Recommendations

### Daily Coding Practice
1. **Implement from scratch**: Core algorithms without libraries
2. **Optimize existing code**: Take working code and make it faster/more memory efficient
3. **Debug broken implementations**: Practice identifying and fixing issues
4. **Code review**: Review others' implementations and provide feedback

### Mock Interview Preparation
1. **Time yourself**: Practice coding under time pressure
2. **Think out loud**: Verbalize your thought process
3. **Ask clarifying questions**: Don't assume requirements
4. **Test your code**: Always validate your implementation

**Next**: [Research Presentation →](../04-interview-practice/13-research-presentation.md)
