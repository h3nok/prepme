"""
Multi-Head Attention Implementation
==================================

Clean, interview-ready implementation of multi-head attention mechanism.
This is a common coding question for ML engineer/scientist roles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention is All You Need"
    
    Args:
        d_model: Dimension of the model (embedding size)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, d_k]
            k: Key tensor [batch_size, num_heads, seq_len, d_k]
            v: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional mask tensor
            
        Returns:
            attention_output: [batch_size, num_heads, seq_len, d_k]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model] 
            value: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()
        
        # 1. Linear projections
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # 2. Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 4. Concatenate heads
        # [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 5. Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights


# Example usage and testing
if __name__ == "__main__":
    # Test the implementation
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize attention layer
    attention = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, weights = attention(x, x, x)  # Self-attention
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Verify output shape
    assert output.shape == x.shape
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    print("✅ Multi-head attention implementation test passed!")


"""
Common Interview Questions & Talking Points:

1. **Computational Complexity**: 
   - Time: O(n²d) where n=seq_len, d=d_model
   - Space: O(n²) for attention matrix
   - How does this scale with sequence length?

2. **Memory Optimization**:
   - Flash Attention: Reduces memory from O(n²) to O(n)
   - Gradient checkpointing for training
   - Mixed precision training

3. **Variants**:
   - Cross-attention vs self-attention
   - Causal masking for autoregressive models
   - Relative position encoding

4. **Implementation Details**:
   - Why divide by sqrt(d_k)?
   - Bias vs no bias in linear layers
   - Dropout placement strategies

5. **Production Considerations**:
   - Batch inference optimization
   - KV caching for generation
   - Model parallelism strategies
"""
