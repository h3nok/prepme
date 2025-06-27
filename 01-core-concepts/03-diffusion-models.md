# Diffusion Models

## 🎯 Learning Objectives
- Master the mathematical foundation of diffusion models
- Understand different sampling algorithms and their trade-offs
- Know conditioning mechanisms and guidance techniques
- Implement key components of diffusion models

## 🧮 Mathematical Foundation

### The Diffusion Process

#### Forward Process (Adding Noise)
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Intuition**: Gradually add Gaussian noise over T timesteps until data becomes pure noise.

**Reparameterization Trick**:
```
x_t = √(α̅_t) x_0 + √(1-α̅_t) ε
where α_t = 1 - β_t, α̅_t = ∏_{s=1}^t α_s, ε ~ N(0,I)
```

**Key Properties**:
- **Markovian**: Each step only depends on previous step
- **Gaussian**: Each transition is a Gaussian distribution
- **Closed form**: Can sample x_t directly from x_0

#### Reverse Process (Denoising)
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Goal**: Learn to reverse the noise addition process.

### Training Objective

#### Variational Lower Bound
```
L = E_q[D_KL(q(x_T|x_0) || p(x_T)) + 
        ∑_{t>1} D_KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)) - 
        log p_θ(x_0|x_1)]
```

#### Simplified Training Loss (DDPM)
```
L_simple = E_{t,x_0,ε}[||ε - ε_θ(√(α̅_t) x_0 + √(1-α̅_t) ε, t)||²]
```

**Intuition**: Train neural network to predict the noise that was added.

### Noise Schedules

#### Linear Schedule (Original DDPM)
```python
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
```

#### Cosine Schedule (Improved DDPM)
```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

## 🏗️ Model Architectures

### U-Net Architecture

#### Core Components
```python
class UNet(nn.Module):
    def __init__(self, in_channels, model_channels, num_res_blocks):
        super().__init__()
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, 4 * model_channels),
            nn.SiLU(),
            nn.Linear(4 * model_channels, 4 * model_channels),
        )
        
        # Downsampling path
        self.down_blocks = nn.ModuleList([
            ResBlock(in_channels, model_channels, time_emb_dim=4*model_channels),
            AttentionBlock(model_channels),
            Downsample(model_channels)
        ])
        
        # Upsampling path with skip connections
        self.up_blocks = nn.ModuleList([
            Upsample(model_channels),
            AttentionBlock(model_channels),
            ResBlock(2*model_channels, model_channels, time_emb_dim=4*model_channels)
        ])
```

#### Residual Blocks
```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[..., None, None]  # Broadcast time embedding
        h = self.block2(h)
        return h + self.shortcut(x)
```

#### Self-Attention Blocks
```python
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H*W).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=2)
        
        # Attention computation
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(C), dim=-1)
        h = (attn @ v).reshape(B, C, H, W)
        
        return x + self.proj_out(h)
```

### Timestep Embedding
```python
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
```

## 🎲 Sampling Algorithms

### DDPM Sampling (Slow but Accurate)
```python
def ddpm_sample(model, shape, timesteps=1000):
    device = next(model.parameters()).device
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(timesteps)):
        t_batch = torch.full((shape[0],), t, device=device)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = model(x, t_batch)
        
        # Compute denoising step
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = beta[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise
    
    return x
```

### DDIM Sampling (Fast Deterministic)
```python
def ddim_sample(model, shape, timesteps=50, eta=0.0):
    # Skip timesteps for faster sampling
    skip = len(alpha_bar) // timesteps
    timestep_seq = range(0, len(alpha_bar), skip)
    
    x = torch.randn(shape, device=device)
    
    for i, t in enumerate(reversed(timestep_seq)):
        t_batch = torch.full((shape[0],), t, device=device)
        
        with torch.no_grad():
            predicted_noise = model(x, t_batch)
        
        alpha_bar_t = alpha_bar[t]
        
        if i < len(timestep_seq) - 1:
            t_prev = timestep_seq[-(i+2)]
            alpha_bar_t_prev = alpha_bar[t_prev]
        else:
            alpha_bar_t_prev = 1.0
        
        # DDIM update rule
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # Direction pointing towards x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)) * predicted_noise
        
        # Random noise component
        noise = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev) * torch.randn_like(x)
        
        x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
    
    return x
```

### DPM-Solver (Very Fast)
```python
def dpm_solver_sample(model, shape, steps=20):
    """
    DPM-Solver for fast high-quality sampling.
    Treats diffusion as solving an ODE.
    """
    # Convert to first-order ODE form
    def model_fn(x, t):
        return -model(x, t)  # Negative because we're going backwards
    
    # Use adaptive ODE solver
    from scipy.integrate import solve_ivp
    
    t_span = (1.0, 0.0)  # From t=1 (noise) to t=0 (data)
    t_eval = torch.linspace(1.0, 0.0, steps)
    
    x0 = torch.randn(shape)
    
    # Solve ODE (simplified - real implementation more complex)
    solution = solve_ivp(
        lambda t, x: model_fn(torch.tensor(x), t),
        t_span, x0.flatten(), t_eval=t_eval, method='RK45'
    )
    
    return torch.tensor(solution.y[:, -1]).reshape(shape)
```

## 🎛️ Conditioning Mechanisms

### Class Conditioning
```python
class ClassConditionedUNet(UNet):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, 4 * self.model_channels)
    
    def forward(self, x, timesteps, y=None):
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        if y is not None:
            class_emb = self.class_emb(y)
            time_emb = time_emb + class_emb
        
        return super().forward(x, time_emb)
```

### Text Conditioning (Cross-Attention)
```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x, context):
        B, L, C = x.shape
        
        q = self.to_q(x).reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        
        return x + self.to_out(out)
```

### Classifier-Free Guidance
```python
def classifier_free_guidance_sample(model, shape, text_embeddings, w=7.5):
    """
    Classifier-free guidance for better conditioning.
    w: guidance scale (higher = more conditioning, lower = more diversity)
    """
    # Prepare conditional and unconditional embeddings
    batch_size = shape[0]
    uncond_embeddings = torch.zeros_like(text_embeddings)  # Null conditioning
    
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(timesteps)):
        t_batch = torch.full((batch_size,), t, device=device)
        
        # Predict noise with conditioning
        with torch.no_grad():
            noise_cond = model(x, t_batch, text_embeddings)
            noise_uncond = model(x, t_batch, uncond_embeddings)
        
        # Apply classifier-free guidance
        noise_pred = noise_uncond + w * (noise_cond - noise_uncond)
        
        # Standard DDPM/DDIM denoising step
        x = denoise_step(x, noise_pred, t)
    
    return x
```

## 🎨 Applications & Variations

### Image Generation (DALL-E 2, Stable Diffusion)

#### DALL-E 2 Architecture
```
Text → CLIP Text Encoder → Prior Model → CLIP Image Embeddings → Diffusion Decoder → Image
```

#### Stable Diffusion Architecture
```
Text → CLIP Text Encoder ↘
                           Cross-Attention → Latent Diffusion → VAE Decoder → Image
Latent Space ← VAE Encoder ↗
```

**Latent Diffusion Benefits**:
- **Efficiency**: Work in compressed latent space (8x smaller)
- **Quality**: VAE provides good reconstruction
- **Speed**: Faster than pixel-space diffusion

### Inpainting
```python
def inpainting_sample(model, image, mask, text_embedding):
    """
    Fill masked regions conditioned on text and unmasked regions.
    """
    latents = vae_encode(image)
    
    for t in reversed(range(timesteps)):
        # Predict noise for entire latent
        noise_pred = model(latents, t, text_embedding)
        
        # Denoise
        latents = denoise_step(latents, noise_pred, t)
        
        # Replace unmasked regions with original (noise-corrupted) latents
        if t > 0:
            original_latents_t = add_noise(vae_encode(image), t)
            latents = latents * mask + original_latents_t * (1 - mask)
    
    return vae_decode(latents)
```

### ControlNet
```python
class ControlNet(nn.Module):
    def __init__(self, unet, conditioning_channels):
        super().__init__()
        self.unet = copy.deepcopy(unet)  # Copy of encoder part
        self.control_conv = nn.Conv2d(conditioning_channels, unet.input_channels, 3, padding=1)
        
        # Zero-initialized projection layers
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for channels in unet.skip_channels
        ])
        
        # Initialize to zero
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
    
    def forward(self, x, timesteps, control_input, text_embedding):
        # Process control input
        control_features = self.control_conv(control_input)
        
        # Run through copied encoder
        control_skips = []
        h = control_features
        for layer, zero_conv in zip(self.unet.down_blocks, self.zero_convs):
            h = layer(h, timesteps, text_embedding)
            control_skips.append(zero_conv(h))
        
        # Original U-Net with control skip connections
        return self.unet(x, timesteps, text_embedding, control_skips)
```

## 🔊 Audio Generation

### WaveGrad (Audio Diffusion)
```python
class WaveGrad(nn.Module):
    def __init__(self, mel_channels, residual_layers, residual_channels):
        super().__init__()
        
        # Mel-spectrogram conditioning
        self.mel_conv = nn.Conv1d(mel_channels, residual_channels, 1)
        
        # Dilated convolution blocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, dilation=2**i)
            for i in range(residual_layers)
        ])
        
        self.output_projection = nn.Conv1d(residual_channels, 1, 1)
    
    def forward(self, audio, noise_level, mel_spectrogram):
        # Condition on mel-spectrogram
        mel_emb = self.mel_conv(mel_spectrogram)
        
        # Add noise level embedding
        noise_emb = self.noise_level_mlp(noise_level)
        
        h = audio + mel_emb
        for layer in self.residual_layers:
            h = layer(h, noise_emb)
        
        return self.output_projection(h)
```

## 📊 Training Strategies

### Progressive Training
```python
def progressive_training_schedule(epoch, total_epochs):
    """
    Start with low resolution, gradually increase.
    """
    if epoch < total_epochs * 0.3:
        resolution = 64
    elif epoch < total_epochs * 0.6:
        resolution = 128
    else:
        resolution = 256
    return resolution
```

### Multi-Scale Training
```python
def multiscale_loss(model, x_0, timesteps):
    """
    Train on multiple resolutions simultaneously.
    """
    losses = []
    
    for scale in [0.5, 0.75, 1.0]:
        if scale < 1.0:
            x_scaled = F.interpolate(x_0, scale_factor=scale, mode='bilinear')
        else:
            x_scaled = x_0
        
        # Standard diffusion loss at this scale
        t = torch.randint(0, timesteps, (x_scaled.shape[0],))
        noise = torch.randn_like(x_scaled)
        x_t = add_noise(x_scaled, noise, t)
        
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        losses.append(loss)
    
    return sum(losses) / len(losses)
```

### Noise Prediction vs. X0 Prediction
```python
# Noise prediction (standard)
def noise_prediction_loss(model, x_0, t):
    noise = torch.randn_like(x_0)
    x_t = add_noise(x_0, noise, t)
    predicted_noise = model(x_t, t)
    return F.mse_loss(predicted_noise, noise)

# X0 prediction (alternative)
def x0_prediction_loss(model, x_0, t):
    noise = torch.randn_like(x_0)
    x_t = add_noise(x_0, noise, t)
    predicted_x0 = model(x_t, t)
    return F.mse_loss(predicted_x0, x_0)

# V-parameterization (velocity prediction)
def v_prediction_loss(model, x_0, t):
    noise = torch.randn_like(x_0)
    x_t = add_noise(x_0, noise, t)
    
    # Velocity target
    alpha_bar_t = alpha_bar[t]
    v_target = torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1 - alpha_bar_t) * x_0
    
    predicted_v = model(x_t, t)
    return F.mse_loss(predicted_v, v_target)
```

## 📈 Evaluation Metrics

### Frechet Inception Distance (FID)
```python
def calculate_fid(real_features, generated_features):
    """
    Calculate FID between real and generated image features.
    Lower is better.
    """
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate Frechet distance
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
```

### Inception Score (IS)
```python
def calculate_inception_score(images, splits=10):
    """
    Calculate Inception Score.
    Higher is better (both quality and diversity).
    """
    # Get predictions from Inception model
    preds = inception_model(images)
    
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits):(i + 1) * (len(preds) // splits)]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))
    
    return np.mean(scores), np.std(scores)
```

### CLIP Score (Text-Image Alignment)
```python
def calculate_clip_score(images, texts):
    """
    Calculate CLIP score for text-image alignment.
    Higher is better.
    """
    image_features = clip_model.encode_image(images)
    text_features = clip_model.encode_text(texts)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate cosine similarity
    clip_scores = (image_features * text_features).sum(dim=-1)
    return clip_scores.mean().item()
```

## 🎯 Interview Questions & Answers

### Q1: "Explain the intuition behind diffusion models. Why do they work?"

**Answer Framework**:
1. **Learning data distribution**: Instead of directly modeling complex data distribution, gradually transform it to simple Gaussian
2. **Reverse engineering**: Learn to reverse the noise addition process
3. **Score matching**: Neural network learns the gradient of log probability density
4. **Stable training**: Unlike GANs, no adversarial training or mode collapse
5. **High quality**: Iterative refinement leads to high-quality samples

### Q2: "Compare diffusion models to GANs. When would you choose each?"

**Answer Framework**:

**Diffusion Models**:
- **Pros**: Stable training, high quality, no mode collapse, good sample diversity
- **Cons**: Slow sampling (many steps), high computational cost
- **Use when**: Quality is paramount, training stability important, diversity needed

**GANs**:
- **Pros**: Fast sampling (single forward pass), efficient inference
- **Cons**: Training instability, mode collapse, harder to scale
- **Use when**: Real-time generation needed, computational efficiency important

### Q3: "How would you make diffusion model sampling faster?"

**Answer Framework**:
1. **DDIM sampling**: Deterministic sampling with fewer steps
2. **Advanced ODE solvers**: DPM-Solver, Euler methods
3. **Distillation**: Train smaller model to mimic larger one
4. **Progressive distillation**: Iteratively halve the number of steps
5. **Latent diffusion**: Work in compressed latent space
6. **Consistency models**: Direct mapping from noise to data

### Q4: "Explain classifier-free guidance and why it's important."

**Answer Framework**:
1. **Problem with classifier guidance**: Need separate classifier, can be unstable
2. **Classifier-free approach**: Single model trained conditionally and unconditionally
3. **Guidance formula**: ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
4. **Benefits**: More stable, better conditioning, easier to train
5. **Trade-off**: Higher w gives better conditioning but less diversity

## 🚀 Advanced Topics

### Score-Based Models
```python
def score_matching_loss(model, x):
    """
    Train model to predict score function ∇log p(x).
    """
    x.requires_grad_(True)
    
    # Add small amount of noise for numerical stability
    sigma = 0.01
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    
    # Model predicts score
    score_pred = model(x_noisy)
    
    # True score for Gaussian noise
    score_true = -noise / (sigma ** 2)
    
    return F.mse_loss(score_pred, score_true)
```

### Flow Matching
```python
def flow_matching_loss(model, x_0, x_1, t):
    """
    Flow matching: learn continuous normalizing flows.
    """
    # Linear interpolation path
    x_t = t * x_1 + (1 - t) * x_0
    
    # Target velocity field
    v_target = x_1 - x_0
    
    # Predicted velocity
    v_pred = model(x_t, t)
    
    return F.mse_loss(v_pred, v_target)
```

### Consistency Models
```python
def consistency_training_loss(model, x, timesteps):
    """
    Train model to be consistent across timesteps.
    """
    t1 = torch.randint(1, timesteps, (x.shape[0],))
    t2 = t1 - 1
    
    # Add noise at both timesteps
    x_t1 = add_noise(x, t1)
    x_t2 = add_noise(x, t2)
    
    # Model should predict same clean image from both
    pred_x_t1 = model(x_t1, t1)
    pred_x_t2 = model(x_t2, t2)
    
    return F.mse_loss(pred_x_t1, pred_x_t2)
```

---

## 📝 Study Checklist

- [ ] Understand forward and reverse diffusion processes mathematically
- [ ] Can explain different sampling algorithms (DDPM, DDIM, DPM-Solver)
- [ ] Know conditioning mechanisms (class, text, ControlNet)
- [ ] Understand classifier-free guidance and its benefits
- [ ] Familiar with evaluation metrics (FID, IS, CLIP Score)
- [ ] Can implement basic diffusion model components
- [ ] Know applications (image generation, inpainting, audio)
- [ ] Understand recent advances (consistency models, flow matching)

**Next**: [Multimodal AI →](../01-core-concepts/04-multimodal-ai.md)
