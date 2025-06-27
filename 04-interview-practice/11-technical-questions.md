# Technical Interview Questions

## 🎯 Core Technical Concepts

### Transformer Architecture

#### Q1: "Explain the self-attention mechanism and why it's more effective than RNN for long sequences."

**Expected Answer Framework**:
```python
# Mathematical formulation
def self_attention(Q, K, V):
    """
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)

# Key advantages over RNNs:
advantages = {
    'parallelization': 'All positions computed simultaneously vs sequential',
    'long_range_dependencies': 'Direct connections between any two positions',
    'computational_efficiency': 'Better utilization of modern hardware',
    'gradient_flow': 'No vanishing gradient problem over long sequences'
}
```

**Follow-up Questions**:
- "How would you implement multi-head attention efficiently?"
- "What's the computational complexity of self-attention?"
- "How do you handle very long sequences?"

#### Q2: "Design a transformer variant for processing very long sequences (1M+ tokens)."

**Expected Solution**:
```python
class EfficientLongTransformer(nn.Module):
    def __init__(self, d_model, max_seq_len=1000000):
        super().__init__()
        
        # Sparse attention patterns
        self.local_attention = LocalAttention(window_size=512)
        self.global_attention = GlobalAttention(num_global_tokens=64)
        self.random_attention = RandomAttention(num_random=32)
        
        # Hierarchical processing
        self.segment_encoder = SegmentEncoder(segment_size=1024)
        self.global_encoder = GlobalEncoder()
        
    def forward(self, x):
        # Process in segments
        segments = self.chunk_sequence(x, chunk_size=1024)
        segment_representations = []
        
        for segment in segments:
            # Local attention within segment
            local_out = self.local_attention(segment)
            segment_repr = self.segment_encoder(local_out)
            segment_representations.append(segment_repr)
        
        # Global attention across segment representations
        global_context = torch.stack(segment_representations, dim=1)
        global_out = self.global_encoder(global_context)
        
        return global_out
```

### Large Language Models

#### Q3: "Compare different approaches to align LLMs with human preferences. What are the trade-offs?"

**Expected Answer**:
```python
alignment_approaches = {
    'supervised_fine_tuning': {
        'method': 'Train on human-written responses',
        'pros': ['Simple', 'Direct supervision', 'Fast training'],
        'cons': ['Limited by human examples', 'Expensive annotation', 'Distribution shift'],
        'when_to_use': 'Initial alignment, specific domains'
    },
    
    'rlhf': {
        'method': 'Reward model + PPO optimization',
        'pros': ['Learns from preferences', 'Can improve beyond human examples', 'Flexible'],
        'cons': ['Complex training', 'Reward hacking', 'Instability'],
        'when_to_use': 'General alignment, safety-critical applications'
    },
    
    'constitutional_ai': {
        'method': 'Self-critique and revision',
        'pros': ['Self-improving', 'Scalable', 'Principle-based'],
        'cons': ['Requires good initialization', 'Principle formulation challenging'],
        'when_to_use': 'Scalable alignment, reducing human oversight'
    },
    
    'direct_preference_optimization': {
        'method': 'Direct optimization on preference data',
        'pros': ['Simpler than RLHF', 'More stable', 'No reward model needed'],
        'cons': ['Still requires preference data', 'Less flexible than RLHF'],
        'when_to_use': 'When RLHF is too complex, preference data available'
    }
}
```

#### Q4: "How would you implement and optimize RLHF for a 70B parameter model?"

**Expected Solution**:
```python
class RLHFTrainer:
    def __init__(self, policy_model, reward_model, ref_model):
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        
        # Optimization setup
        self.setup_distributed_training()
        self.setup_memory_optimization()
        
    def setup_distributed_training(self):
        """3D parallelism for 70B model training."""
        self.config = {
            'data_parallel_size': 8,      # Batch parallelism
            'pipeline_parallel_size': 4,   # Layer parallelism  
            'tensor_parallel_size': 8,     # Within-layer parallelism
            'total_gpus': 8 * 4 * 8        # 256 GPUs total
        }
        
    def ppo_step(self, prompts, batch_size=32):
        """Optimized PPO step for large model."""
        
        # Generate responses with current policy
        with torch.no_grad():
            responses = self.policy.generate(prompts, max_length=512)
            
        # Get rewards
        rewards = self.reward_model(prompts, responses)
        
        # Get reference model logprobs (frozen)
        with torch.no_grad():
            ref_logprobs = self.ref_model.get_logprobs(prompts, responses)
            
        # Get current policy logprobs
        policy_logprobs = self.policy.get_logprobs(prompts, responses)
        
        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, values)
        
        # PPO loss with gradient checkpointing
        with torch.cuda.amp.autocast():  # Mixed precision
            ratio = torch.exp(policy_logprobs - ref_logprobs)
            clipped_ratio = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
            
            # KL penalty to prevent drift
            kl_penalty = self.kl_coeff * (policy_logprobs - ref_logprobs)
            
            total_loss = policy_loss + kl_penalty
            
        # Gradient accumulation and optimization
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

### Diffusion Models

#### Q5: "Explain the difference between DDPM and DDIM sampling. When would you use each?"

**Expected Answer**:
```python
def ddpm_sampling_step(model, x_t, t, alpha_t, alpha_bar_t, beta_t):
    """
    DDPM: Stochastic sampling following learned reverse process
    - Requires T steps (usually 1000)
    - Adds noise at each step
    - High quality but slow
    """
    with torch.no_grad():
        predicted_noise = model(x_t, t)
    
    # Compute mean of reverse distribution
    mean = (1 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
    )
    
    if t > 0:
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise
    else:
        return mean

def ddim_sampling_step(model, x_t, t_curr, t_prev, eta=0.0):
    """
    DDIM: Deterministic sampling (when eta=0)
    - Can skip timesteps (e.g., 50 steps instead of 1000)
    - Deterministic when eta=0
    - Faster but potentially lower quality
    """
    with torch.no_grad():
        predicted_noise = model(x_t, t_curr)
    
    alpha_bar_t = alpha_bar[t_curr]
    alpha_bar_t_prev = alpha_bar[t_prev] if t_prev >= 0 else 1.0
    
    # Predict x_0
    pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    
    # Direction pointing towards x_t
    dir_xt = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * variance_term) * predicted_noise
    
    # Random noise (controlled by eta)
    noise = eta * torch.sqrt(variance_term) * torch.randn_like(x_t)
    
    return torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise

# Usage guidelines
sampling_choice = {
    'ddpm': {
        'use_when': ['Highest quality needed', 'Research/analysis', 'Small batch generation'],
        'trade_offs': 'Slow but high quality'
    },
    'ddim': {
        'use_when': ['Interactive applications', 'Large batch generation', 'Real-time needs'],
        'trade_offs': 'Fast but potentially lower quality'
    }
}
```

#### Q6: "How would you implement classifier-free guidance for text-to-image generation?"

**Expected Implementation**:
```python
class ClassifierFreeGuidanceModel(nn.Module):
    def __init__(self, unet, text_encoder):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        
        # Special tokens for unconditional generation
        self.null_text_embed = nn.Parameter(torch.randn(1, 77, 768))
        
    def forward(self, x_t, t, text_prompts, guidance_scale=7.5):
        batch_size = x_t.shape[0]
        
        # Encode text prompts
        text_embeddings = self.text_encoder(text_prompts)  # [B, 77, 768]
        
        # Create unconditional embeddings
        uncond_embeddings = self.null_text_embed.expand(batch_size, -1, -1)
        
        # Concatenate for single forward pass
        combined_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        combined_x_t = torch.cat([x_t, x_t], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        
        # Single forward pass for efficiency
        combined_noise_pred = self.unet(combined_x_t, combined_t, combined_embeddings)
        
        # Split predictions
        noise_pred_uncond, noise_pred_cond = combined_noise_pred.chunk(2, dim=0)
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        return noise_pred

# Training with random conditioning dropout
def train_with_cfg_dropout(model, batch, dropout_prob=0.1):
    """
    Train model to work with and without conditioning.
    """
    x_0, text_prompts = batch['images'], batch['captions']
    
    # Randomly drop text conditioning
    mask = torch.rand(len(text_prompts)) > dropout_prob
    
    # Replace dropped prompts with empty string
    masked_prompts = [prompt if m else "" for prompt, m in zip(text_prompts, mask)]
    
    # Standard diffusion training
    t = torch.randint(0, model.num_timesteps, (x_0.shape[0],))
    noise = torch.randn_like(x_0)
    x_t = model.add_noise(x_0, noise, t)
    
    predicted_noise = model(x_t, t, masked_prompts)
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss
```

### Multimodal AI

#### Q7: "Design a multimodal model that can understand images, text, and audio simultaneously."

**Expected Architecture**:
```python
class UnifiedMultimodalModel(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        
        # Modality-specific encoders
        self.vision_encoder = VisionTransformer(d_model=d_model)
        self.text_encoder = TextTransformer(d_model=d_model)
        self.audio_encoder = AudioTransformer(d_model=d_model)
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(3, d_model)  # vision, text, audio
        
        # Cross-modal fusion
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(d_model) for _ in range(6)
        ])
        
        # Task-specific heads
        self.classification_head = nn.Linear(d_model, num_classes)
        self.generation_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, images=None, text=None, audio=None, task='classification'):
        multimodal_tokens = []
        attention_mask = []
        
        # Process each available modality
        if images is not None:
            vision_tokens = self.vision_encoder(images)
            vision_tokens += self.modality_embeddings(torch.zeros(vision_tokens.size(0), vision_tokens.size(1), dtype=torch.long))
            multimodal_tokens.append(vision_tokens)
            attention_mask.extend([1] * vision_tokens.size(1))
            
        if text is not None:
            text_tokens = self.text_encoder(text)
            text_tokens += self.modality_embeddings(torch.ones(text_tokens.size(0), text_tokens.size(1), dtype=torch.long))
            multimodal_tokens.append(text_tokens)
            attention_mask.extend([1] * text_tokens.size(1))
            
        if audio is not None:
            audio_tokens = self.audio_encoder(audio)
            audio_tokens += self.modality_embeddings(torch.full((audio_tokens.size(0), audio_tokens.size(1)), 2, dtype=torch.long))
            multimodal_tokens.append(audio_tokens)
            attention_mask.extend([1] * audio_tokens.size(1))
        
        # Concatenate all modalities
        if not multimodal_tokens:
            raise ValueError("At least one modality must be provided")
            
        fused_tokens = torch.cat(multimodal_tokens, dim=1)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).expand(fused_tokens.size(0), -1)
        
        # Cross-modal fusion
        for fusion_layer in self.fusion_layers:
            fused_tokens = fusion_layer(fused_tokens, attention_mask)
        
        # Task-specific processing
        if task == 'classification':
            pooled = fused_tokens.mean(dim=1)
            return self.classification_head(pooled)
        elif task == 'generation':
            return self.generation_head(fused_tokens)

class CrossModalFusionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, attention_mask):
        # Self-attention within modalities
        x2, _ = self.self_attention(x, x, x, key_padding_mask=~attention_mask.bool())
        x = self.norm1(x + x2)
        
        # Cross-attention between modalities
        x2, _ = self.cross_attention(x, x, x, key_padding_mask=~attention_mask.bool())
        x = self.norm2(x + x2)
        
        # Feed-forward
        x2 = self.feed_forward(x)
        x = self.norm3(x + x2)
        
        return x
```

#### Q8: "How would you handle missing modalities during training and inference?"

**Expected Solution**:
```python
class RobustMultimodalModel(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.base_model = UnifiedMultimodalModel(d_model)
        
        # Modality reconstruction heads
        self.vision_reconstructor = nn.Linear(d_model, vision_dim)
        self.text_reconstructor = nn.Linear(d_model, text_dim)
        self.audio_reconstructor = nn.Linear(d_model, audio_dim)
        
    def forward(self, batch, training=True):
        # Extract available modalities
        available_modalities = self.get_available_modalities(batch)
        
        if training:
            return self.training_forward(batch, available_modalities)
        else:
            return self.inference_forward(batch, available_modalities)
    
    def training_forward(self, batch, available_modalities):
        """Training with modality dropout and reconstruction."""
        
        # Random modality dropout (simulate missing modalities)
        if random.random() < 0.3:  # 30% chance of dropout
            available_modalities = self.simulate_missing_modalities(available_modalities)
        
        # Forward pass with available modalities
        outputs = self.base_model(
            images=batch.get('images') if 'vision' in available_modalities else None,
            text=batch.get('text') if 'text' in available_modalities else None,
            audio=batch.get('audio') if 'audio' in available_modalities else None
        )
        
        # Reconstruction loss for missing modalities
        reconstruction_loss = 0
        if 'vision' not in available_modalities and 'images' in batch:
            vision_reconstruction = self.vision_reconstructor(outputs.hidden_states)
            reconstruction_loss += F.mse_loss(vision_reconstruction, batch['images'])
        
        # Similar for other modalities...
        
        return {
            'main_output': outputs,
            'reconstruction_loss': reconstruction_loss
        }
    
    def inference_forward(self, batch, available_modalities):
        """Adaptive inference based on available modalities."""
        
        if len(available_modalities) == 0:
            raise ValueError("No modalities available for inference")
        
        # Use available modalities
        outputs = self.base_model(
            images=batch.get('images') if 'vision' in available_modalities else None,
            text=batch.get('text') if 'text' in available_modalities else None,
            audio=batch.get('audio') if 'audio' in available_modalities else None
        )
        
        # Confidence calibration based on number of modalities
        confidence_multiplier = len(available_modalities) / 3.0  # Assuming 3 total modalities
        outputs.confidence *= confidence_multiplier
        
        return outputs

# Training strategy for robustness
def train_with_missing_modalities(model, dataloader, num_epochs):
    """Training strategy that builds robustness to missing modalities."""
    
    missing_patterns = [
        ['vision'],           # Only vision
        ['text'],            # Only text  
        ['audio'],           # Only audio
        ['vision', 'text'],  # Vision + text
        ['vision', 'audio'], # Vision + audio
        ['text', 'audio'],   # Text + audio
        ['vision', 'text', 'audio']  # All modalities
    ]
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Randomly select missing pattern
            available = random.choice(missing_patterns)
            
            # Mask unavailable modalities
            masked_batch = {k: v for k, v in batch.items() if k in available}
            
            # Forward pass
            outputs = model(masked_batch, training=True)
            
            # Compute total loss
            main_loss = compute_main_loss(outputs['main_output'], batch['labels'])
            reconstruction_loss = outputs['reconstruction_loss']
            
            total_loss = main_loss + 0.1 * reconstruction_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

## 🔬 Research & Implementation Questions

### Q9: "How would you design and implement a new attention mechanism for better efficiency?"

**Expected Research Approach**:
```python
class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(n) complexity instead of O(n²).
    Based on kernel trick: softmax(QK^T) ≈ φ(Q)φ(K)^T
    """
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def kernel_function(self, x):
        """
        Kernel function φ(x) to approximate softmax.
        Using ELU + 1 for positivity.
        """
        return F.elu(x) + 1
    
    def forward(self, query, key, value, mask=None):
        B, N, D = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim)  
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim)
        
        # Apply kernel function
        Q = self.kernel_function(Q)  # [B, N, H, D]
        K = self.kernel_function(K)  # [B, M, H, D]
        
        # Linear attention: O(n) complexity
        # Attention = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)
        
        # Compute K^T @ V and K^T @ 1 (denominator)
        KV = torch.einsum('bmhd,bmhf->hdf', K, V)  # [H, D, F]
        K_sum = K.sum(dim=1)  # [B, H, D]
        
        # Compute attention output
        numerator = torch.einsum('bnhd,hdf->bnhf', Q, KV)  # [B, N, H, F]
        denominator = torch.einsum('bnhd,bhd->bnh', Q, K_sum)  # [B, N, H]
        
        attention_output = numerator / (denominator.unsqueeze(-1) + 1e-6)
        
        # Reshape and project
        attention_output = attention_output.view(B, N, D)
        output = self.out_proj(attention_output)
        
        return output

# Experimental validation
def validate_linear_attention():
    """
    Experimental protocol to validate the new attention mechanism.
    """
    experiments = {
        'efficiency_test': {
            'setup': 'Compare time/memory vs standard attention',
            'metrics': ['forward_time', 'backward_time', 'memory_usage'],
            'sequence_lengths': [512, 1024, 2048, 4096, 8192]
        },
        
        'quality_test': {
            'setup': 'Compare performance on downstream tasks',
            'tasks': ['language_modeling', 'machine_translation', 'classification'],
            'metrics': ['perplexity', 'BLEU', 'accuracy']
        },
        
        'ablation_study': {
            'setup': 'Study different kernel functions',
            'variants': ['elu+1', 'relu+1', 'softplus', 'learned_kernel'],
            'metrics': ['performance', 'stability', 'efficiency']
        }
    }
    
    return experiments
```

### Q10: "Design an evaluation framework for a new generative AI model."

**Expected Framework**:
```python
class GenerativeModelEvaluator:
    def __init__(self, model, task_type='text_generation'):
        self.model = model
        self.task_type = task_type
        self.evaluation_suite = self.build_evaluation_suite()
    
    def build_evaluation_suite(self):
        """Comprehensive evaluation framework."""
        return {
            'automatic_metrics': self.setup_automatic_evaluation(),
            'human_evaluation': self.setup_human_evaluation(),
            'robustness_tests': self.setup_robustness_tests(),
            'safety_evaluation': self.setup_safety_evaluation(),
            'efficiency_metrics': self.setup_efficiency_evaluation()
        }
    
    def setup_automatic_evaluation(self):
        """Automatic metrics for different aspects."""
        if self.task_type == 'text_generation':
            return {
                'quality_metrics': {
                    'perplexity': self.compute_perplexity,
                    'bleu': self.compute_bleu,
                    'rouge': self.compute_rouge,
                    'bertscore': self.compute_bertscore
                },
                'diversity_metrics': {
                    'distinct_ngrams': self.compute_distinct_ngrams,
                    'self_bleu': self.compute_self_bleu,
                    'semantic_diversity': self.compute_semantic_diversity
                },
                'coherence_metrics': {
                    'local_coherence': self.compute_local_coherence,
                    'global_coherence': self.compute_global_coherence
                }
            }
    
    def setup_human_evaluation(self):
        """Human evaluation protocol."""
        return {
            'evaluation_dimensions': [
                'fluency',      # How natural/grammatical
                'coherence',    # How logically consistent
                'relevance',    # How relevant to prompt
                'creativity',   # How novel/creative
                'factuality'    # How factually accurate
            ],
            'annotation_protocol': {
                'scale': '1-5 Likert scale',
                'annotators_per_sample': 3,
                'qualification_test': True,
                'inter_annotator_agreement': 'Krippendorff alpha > 0.7'
            },
            'sample_selection': {
                'strategy': 'stratified_random',
                'size': 1000,
                'criteria': ['prompt_type', 'generation_length', 'model_confidence']
            }
        }
    
    def setup_robustness_tests(self):
        """Robustness evaluation."""
        return {
            'adversarial_prompts': {
                'jailbreaking_attempts': self.test_jailbreaking,
                'prompt_injection': self.test_prompt_injection,
                'adversarial_examples': self.test_adversarial_examples
            },
            'distribution_shift': {
                'domain_transfer': self.test_domain_transfer,
                'temporal_shift': self.test_temporal_shift,
                'demographic_shift': self.test_demographic_shift
            },
            'edge_cases': {
                'very_long_prompts': self.test_long_prompts,
                'very_short_prompts': self.test_short_prompts,
                'multilingual_prompts': self.test_multilingual
            }
        }
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation suite."""
        results = {}
        
        # Automatic evaluation
        print("Running automatic evaluation...")
        results['automatic'] = self.run_automatic_evaluation()
        
        # Human evaluation
        print("Running human evaluation...")  
        results['human'] = self.run_human_evaluation()
        
        # Robustness tests
        print("Running robustness tests...")
        results['robustness'] = self.run_robustness_tests()
        
        # Generate comprehensive report
        report = self.generate_evaluation_report(results)
        
        return results, report
    
    def generate_evaluation_report(self, results):
        """Generate structured evaluation report."""
        report = {
            'executive_summary': self.create_executive_summary(results),
            'detailed_results': results,
            'strengths': self.identify_strengths(results),
            'weaknesses': self.identify_weaknesses(results),
            'recommendations': self.generate_recommendations(results),
            'comparison_to_baselines': self.compare_to_baselines(results)
        }
        
        return report
```

## 🛠️ Implementation & Optimization Questions

### Q11: "How would you optimize transformer training for 100B+ parameter models?"

**Expected Implementation**:
```python
class LargeModelTrainer:
    def __init__(self, model_config, hardware_config):
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.setup_distributed_training()
        self.setup_memory_optimization()
        self.setup_computational_optimization()
    
    def setup_distributed_training(self):
        """3D parallelism setup."""
        total_gpus = self.hardware_config['num_nodes'] * self.hardware_config['gpus_per_node']
        
        # Calculate optimal parallelism dimensions
        self.parallelism_config = {
            'data_parallel_size': 4,
            'pipeline_parallel_size': 8,  
            'tensor_parallel_size': 8,
            'total_gpus': total_gpus
        }
        
        # Verify: dp * pp * tp = total_gpus
        assert self.parallelism_config['data_parallel_size'] * \
               self.parallelism_config['pipeline_parallel_size'] * \
               self.parallelism_config['tensor_parallel_size'] == total_gpus
    
    def setup_memory_optimization(self):
        """Memory optimization techniques."""
        self.memory_config = {
            'gradient_checkpointing': True,
            'activation_checkpointing': True,
            'zero_stage': 3,  # ZeRO-3: partition optimizer states, gradients, and parameters
            'offload_optimizer': True,  # Offload to CPU
            'offload_params': False,    # Keep params on GPU for speed
            'mixed_precision': 'bf16',  # Use bfloat16
            'gradient_compression': True
        }
    
    def setup_computational_optimization(self):
        """Computational optimizations."""
        self.compute_config = {
            'flash_attention': True,      # Memory-efficient attention
            'fused_kernels': True,        # Fused operations
            'compile_model': True,        # torch.compile for optimization
            'dynamic_loss_scaling': True, # For mixed precision
            'gradient_clipping': 1.0,     # Prevent gradient explosion
            'learning_rate_schedule': 'cosine_with_warmup'
        }
    
    def train_step(self, batch):
        """Optimized training step."""
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient accumulation check
        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.compute_config['gradient_clipping'])
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Learning rate scheduling
            self.scheduler.step()
    
    def optimize_batch_size(self):
        """Find optimal batch size through binary search."""
        min_batch_size = 1
        max_batch_size = 1024
        optimal_batch_size = min_batch_size
        
        while min_batch_size <= max_batch_size:
            mid_batch_size = (min_batch_size + max_batch_size) // 2
            
            try:
                # Test if this batch size fits in memory
                self.test_batch_size(mid_batch_size)
                optimal_batch_size = mid_batch_size
                min_batch_size = mid_batch_size + 1
            except torch.cuda.OutOfMemoryError:
                max_batch_size = mid_batch_size - 1
        
        return optimal_batch_size
    
    def implement_gradient_checkpointing(self, model):
        """Implement selective gradient checkpointing."""
        
        def checkpoint_wrapper(module):
            def forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    module.forward, *args, **kwargs, use_reentrant=False
                )
            return forward
        
        # Apply checkpointing to transformer blocks
        for i, layer in enumerate(model.transformer.layers):
            if i % 2 == 0:  # Checkpoint every other layer
                layer.forward = checkpoint_wrapper(layer)
```

### Q12: "Implement efficient inference for a large multimodal model."

**Expected Solution**:
```python
class EfficientMultimodalInference:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.setup_optimizations()
    
    def setup_optimizations(self):
        """Setup various inference optimizations."""
        
        # Model optimizations
        self.model.eval()
        self.model = torch.jit.script(self.model)  # TorchScript for speed
        
        # KV Cache for autoregressive generation
        self.kv_cache = self.initialize_kv_cache()
        
        # Batch processing
        self.batch_processor = BatchProcessor(max_batch_size=32)
        
        # Quantization
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    def initialize_kv_cache(self):
        """Initialize KV cache for efficient generation."""
        return {
            'keys': torch.zeros(1, self.model.config.num_heads, 0, self.model.config.head_dim),
            'values': torch.zeros(1, self.model.config.num_heads, 0, self.model.config.head_dim),
            'position': 0
        }
    
    @torch.no_grad()
    def generate_with_kv_cache(self, prompt_tokens, max_length=100):
        """Efficient generation using KV cache."""
        
        generated_tokens = prompt_tokens.clone()
        
        for _ in range(max_length):
            # Only process new token, reuse cached K,V
            if self.kv_cache['position'] == 0:
                # First forward pass - process entire prompt
                input_tokens = generated_tokens
            else:
                # Subsequent passes - only process last token
                input_tokens = generated_tokens[:, -1:]
            
            # Forward pass
            outputs = self.model(
                input_tokens, 
                past_key_values=self.get_past_key_values(),
                use_cache=True
            )
            
            # Update KV cache
            self.update_kv_cache(outputs.past_key_values)
            
            # Sample next token
            logits = outputs.logits[:, -1, :]
            next_token = self.sample_next_token(logits)
            
            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
            
            # Check for end token
            if next_token.item() == self.model.config.eos_token_id:
                break
        
        return generated_tokens
    
    def batch_inference(self, batch_inputs):
        """Efficient batched inference with dynamic batching."""
        
        # Group inputs by modality and sequence length
        grouped_inputs = self.batch_processor.group_inputs(batch_inputs)
        
        results = []
        for group in grouped_inputs:
            # Process each group efficiently
            with torch.cuda.amp.autocast():  # Mixed precision for speed
                group_results = self.process_group(group)
            results.extend(group_results)
        
        return results
    
    def process_group(self, group_inputs):
        """Process a group of similar inputs efficiently."""
        
        # Pad inputs to same length within group
        padded_inputs = self.pad_group_inputs(group_inputs)
        
        # Batch forward pass
        with torch.no_grad():
            outputs = self.model(**padded_inputs)
        
        # Unpad and return individual results
        return self.unpad_outputs(outputs, group_inputs)
    
    def implement_speculative_decoding(self, draft_model, target_model, inputs):
        """
        Speculative decoding for faster generation.
        Draft model generates multiple tokens, target model verifies.
        """
        draft_tokens = []
        accepted_tokens = []
        
        # Draft phase: generate k tokens with small model
        k = 4  # Number of speculative tokens
        draft_outputs = draft_model.generate(inputs, max_new_tokens=k)
        draft_tokens = draft_outputs[:, inputs.shape[1]:]
        
        # Verification phase: check with large model
        verification_input = torch.cat([inputs, draft_tokens], dim=1)
        target_logits = target_model(verification_input).logits
        
        # Accept/reject tokens based on probability ratios
        for i in range(k):
            draft_prob = F.softmax(draft_model(verification_input[:, :inputs.shape[1]+i+1]).logits[:, -1], dim=-1)
            target_prob = F.softmax(target_logits[:, inputs.shape[1]+i-1], dim=-1)
            
            token = draft_tokens[:, i]
            acceptance_ratio = target_prob[:, token] / draft_prob[:, token]
            
            if torch.rand(1) < acceptance_ratio:
                accepted_tokens.append(token)
            else:
                # Reject and sample from adjusted distribution
                adjusted_prob = F.normalize(torch.max(target_prob - draft_prob, torch.zeros_like(target_prob)), p=1, dim=-1)
                rejected_token = torch.multinomial(adjusted_prob, 1)
                accepted_tokens.append(rejected_token)
                break
        
        return torch.stack(accepted_tokens, dim=1)

class BatchProcessor:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
    
    def group_inputs(self, inputs):
        """Group inputs by similarity for efficient batching."""
        
        groups = []
        current_group = []
        
        # Sort by sequence length
        sorted_inputs = sorted(inputs, key=lambda x: len(x['input_ids']))
        
        for inp in sorted_inputs:
            if len(current_group) >= self.max_batch_size:
                groups.append(current_group)
                current_group = [inp]
            else:
                current_group.append(inp)
        
        if current_group:
            groups.append(current_group)
        
        return groups
```

---

## 📝 Answer Quality Framework

### Evaluation Criteria

**Technical Depth (40%)**:
- Mathematical understanding
- Implementation details
- System design considerations
- Optimization awareness

**Problem-Solving Approach (30%)**:
- Structured thinking
- Trade-off analysis
- Edge case consideration
- Scalability planning

**Communication (20%)**:
- Clear explanations
- Appropriate technical language
- Visual aids/diagrams
- Code readability

**Innovation & Research Mindset (10%)**:
- Novel approaches
- Research awareness
- Future directions
- Critical thinking

### Follow-up Strategies

**Probing Questions**:
- "How would this scale to 10x larger models?"
- "What are the failure modes of this approach?"
- "How would you validate this experimentally?"
- "What would you optimize first in production?"

**Code Review Style**:
- "Walk me through this implementation"
- "How would you test this code?"
- "What edge cases are you handling?"
- "How would you monitor this in production?"

---

## 📚 Study Recommendations

### Daily Practice
- Implement key algorithms from scratch
- Solve 1-2 technical problems daily
- Read recent papers and summarize
- Practice explaining complex concepts simply

### Mock Interview Prep
- Record yourself answering questions
- Practice whiteboard coding
- Time your responses
- Get feedback from peers

**Next**: [Coding Challenges →](../04-interview-practice/12-coding-challenges.md)
