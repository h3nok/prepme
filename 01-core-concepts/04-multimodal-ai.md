# Multimodal AI

## 🎯 Learning Objectives
- Understand vision-language models and their architectures
- Master cross-modal alignment and representation learning
- Know audio-visual models and their applications
- Implement multimodal fusion techniques

## 🌐 Multimodal Landscape

### Why Multimodal AI?
1. **Human-like understanding**: Humans process multiple modalities simultaneously
2. **Richer representations**: Each modality provides complementary information
3. **Better generalization**: Cross-modal knowledge transfer
4. **Real-world applications**: Most practical AI systems need multimodal capabilities

### Core Challenges
1. **Alignment**: Different modalities have different structures and semantics
2. **Fusion**: How to effectively combine information from multiple sources
3. **Missing modalities**: Handling incomplete inputs gracefully
4. **Scalability**: Training on massive multimodal datasets

## 👁️ Vision-Language Models

### CLIP (Contrastive Language-Image Pre-training)

#### Architecture Overview
```
Image → Vision Transformer → Image Embedding ↘
                                              Contrastive Loss
Text → Text Transformer → Text Embedding ↗
```

#### Contrastive Learning Objective
```python
def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    CLIP contrastive loss: maximize similarity between paired image-text,
    minimize similarity between unpaired.
    """
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # Symmetric loss
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    
    loss_i2t = F.cross_entropy(logits, labels)  # Image to text
    loss_t2i = F.cross_entropy(logits.T, labels)  # Text to image
    
    return (loss_i2t + loss_t2i) / 2
```

#### Vision Encoder (ViT)
```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True),
            num_layers
        )
        
        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token as image representation
        x = self.ln_final(x[:, 0])
        x = self.proj(x)
        
        return x
```

#### Text Encoder
```python
class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len=77):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(max_len, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True),
            num_layers
        )
        
        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # Token + position embeddings
        x = self.token_embed(x) + self.pos_embed[:x.size(1)]
        
        # Transformer
        x = self.transformer(x)
        
        # Use [EOS] token representation (last token)
        x = self.ln_final(x[torch.arange(x.size(0)), x.argmax(dim=-1)])
        x = self.proj(x)
        
        return x
```

### DALL-E 2

#### Architecture Pipeline
```
Text → CLIP Text Encoder → Prior Model → CLIP Image Embeddings → Unclip Decoder → Image
```

#### Prior Model (Text → Image Embeddings)
```python
class Prior(nn.Module):
    def __init__(self, clip_embed_dim, prior_embed_dim, num_layers):
        super().__init__()
        
        # Text conditioning
        self.text_proj = nn.Linear(clip_embed_dim, prior_embed_dim)
        
        # Transformer for prior
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(prior_embed_dim, 8, batch_first=True),
            num_layers
        )
        
        # Output projection to CLIP image embedding space
        self.output_proj = nn.Linear(prior_embed_dim, clip_embed_dim)
    
    def forward(self, text_embeddings, num_samples=1):
        batch_size = text_embeddings.shape[0]
        
        # Project text embeddings
        text_cond = self.text_proj(text_embeddings)
        
        # Sample noise and condition on text
        noise = torch.randn(batch_size, num_samples, self.prior_embed_dim)
        
        # Generate image embeddings
        image_embeddings = self.transformer(noise, text_cond)
        image_embeddings = self.output_proj(image_embeddings)
        
        return image_embeddings
```

#### Unclip Decoder (Image Embeddings → Image)
```python
class UncilpDecoder(nn.Module):
    def __init__(self, clip_embed_dim):
        super().__init__()
        
        # Modified U-Net with CLIP embedding conditioning
        self.clip_proj = nn.Linear(clip_embed_dim, 4 * 256)  # Project to time embedding dim
        self.unet = UNet(
            in_channels=3,
            model_channels=256,
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8]
        )
    
    def forward(self, noisy_images, timesteps, clip_embeddings):
        # Add CLIP conditioning to time embeddings
        time_emb = self.unet.time_embed(timestep_embedding(timesteps, 256))
        clip_emb = self.clip_proj(clip_embeddings)
        
        conditioning = time_emb + clip_emb
        
        return self.unet(noisy_images, conditioning)
```

### BLIP (Bootstrapping Language-Image Pre-training)

#### Multi-Task Architecture
```python
class BLIP(nn.Module):
    def __init__(self, vision_width, text_width, embed_dim):
        super().__init__()
        
        # Shared vision encoder
        self.vision_encoder = VisionTransformer(...)
        
        # Text encoder (for ITC and ITM)
        self.text_encoder = TextTransformer(...)
        
        # Text decoder (for captioning)
        self.text_decoder = nn.TransformerDecoder(...)
        
        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim, 8)
        
        # Task-specific heads
        self.itc_head = nn.Linear(embed_dim, embed_dim)  # Image-Text Contrastive
        self.itm_head = nn.Linear(embed_dim * 2, 2)     # Image-Text Matching
    
    def forward(self, images, text, mode='multimodal'):
        # Vision features
        image_embeds = self.vision_encoder(images)
        
        if mode == 'unimodal':
            # Separate encoding for contrastive learning
            text_embeds = self.text_encoder(text)
            return image_embeds, text_embeds
        
        elif mode == 'multimodal':
            # Cross-modal fusion for ITM
            text_embeds = self.text_encoder(text)
            
            # Cross-attention between image and text
            fused_embeds, _ = self.cross_attention(
                text_embeds, image_embeds, image_embeds
            )
            
            # Concatenate for classification
            multimodal_embeds = torch.cat([
                image_embeds.mean(1), fused_embeds.mean(1)
            ], dim=-1)
            
            return self.itm_head(multimodal_embeds)
        
        elif mode == 'generation':
            # Autoregressive text generation conditioned on image
            return self.text_decoder(text, memory=image_embeds)
```

#### CapFilt (Caption and Filter)
```python
def capfilt_bootstrap(model, web_images, web_texts, quality_threshold=0.8):
    """
    Bootstrap high-quality image-text pairs from noisy web data.
    """
    # 1. Caption: Generate captions for images
    with torch.no_grad():
        generated_captions = model.generate_captions(web_images)
    
    # 2. Filter: Score image-text pairs
    caption_scores = model.score_image_text_pairs(web_images, generated_captions)
    text_scores = model.score_image_text_pairs(web_images, web_texts)
    
    # 3. Select high-quality pairs
    high_quality_indices = (caption_scores > quality_threshold) | (text_scores > quality_threshold)
    
    # 4. Create training data
    filtered_images = web_images[high_quality_indices]
    filtered_texts = []
    
    for i, idx in enumerate(high_quality_indices):
        if caption_scores[i] > text_scores[i]:
            filtered_texts.append(generated_captions[i])
        else:
            filtered_texts.append(web_texts[i])
    
    return filtered_images, filtered_texts
```

### LLaVA (Large Language and Vision Assistant)

#### Architecture
```
Image → Vision Encoder → Linear Projection → LLM Input Space
Text → Tokenizer → LLM Input Space
```

#### Vision-Language Connector
```python
class LLaVA(nn.Module):
    def __init__(self, vision_encoder, llm, vision_hidden_size, llm_hidden_size):
        super().__init__()
        
        self.vision_encoder = vision_encoder  # Pre-trained (e.g., CLIP ViT)
        self.llm = llm  # Pre-trained LLM (e.g., Vicuna)
        
        # Simple linear projection to connect vision and language
        self.mm_projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        
        # Special tokens
        self.im_start_token = nn.Parameter(torch.randn(llm_hidden_size))
        self.im_end_token = nn.Parameter(torch.randn(llm_hidden_size))
    
    def encode_images(self, images):
        # Extract image features
        image_features = self.vision_encoder(images)
        
        # Project to LLM space
        image_embeds = self.mm_projector(image_features)
        
        # Add special tokens
        batch_size = image_embeds.shape[0]
        im_start = self.im_start_token.unsqueeze(0).expand(batch_size, 1, -1)
        im_end = self.im_end_token.unsqueeze(0).expand(batch_size, 1, -1)
        
        image_embeds = torch.cat([im_start, image_embeds, im_end], dim=1)
        
        return image_embeds
    
    def forward(self, images, input_ids, attention_mask):
        # Encode images
        image_embeds = self.encode_images(images)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Find image token positions and replace with image embeddings
        # This requires careful handling of the input sequence
        inputs_embeds = self.merge_multimodal_embeddings(text_embeds, image_embeds, input_ids)
        
        # Forward through LLM
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
```

## 🔊 Audio-Visual Models

### Audio-Visual Speech Recognition (AVSR)

#### Architecture
```python
class AudioVisualSpeechRecognizer(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim, vocab_size):
        super().__init__()
        
        # Audio pathway
        self.audio_encoder = nn.LSTM(audio_dim, hidden_dim, batch_first=True)
        
        # Visual pathway (lip reading)
        self.video_cnn = nn.Sequential(
            nn.Conv3d(3, 64, (3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 4, 4))
        )
        self.video_encoder = nn.LSTM(128 * 4 * 4, hidden_dim, batch_first=True)
        
        # Multimodal fusion
        self.fusion = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # Output decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, audio, video):
        # Audio encoding
        audio_features, _ = self.audio_encoder(audio)
        
        # Video encoding
        B, T, C, H, W = video.shape
        video = video.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        video_features = self.video_cnn(video)
        video_features = video_features.view(B, T, -1)
        video_features, _ = self.video_encoder(video_features)
        
        # Cross-modal attention
        fused_features, _ = self.fusion(audio_features, video_features, video_features)
        
        # Decode to text
        decoded, _ = self.decoder(fused_features)
        logits = self.classifier(decoded)
        
        return logits
```

### AudioCLIP

#### Triplet Contrastive Learning
```python
def audio_clip_loss(audio_embeddings, image_embeddings, text_embeddings, temperature=0.07):
    """
    Three-way contrastive learning between audio, image, and text.
    """
    # Normalize all embeddings
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity matrices
    audio_image_sim = torch.matmul(audio_embeddings, image_embeddings.T) / temperature
    audio_text_sim = torch.matmul(audio_embeddings, text_embeddings.T) / temperature
    image_text_sim = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    batch_size = audio_embeddings.shape[0]
    labels = torch.arange(batch_size, device=audio_embeddings.device)
    
    # Six-way symmetric loss
    loss_a2i = F.cross_entropy(audio_image_sim, labels)
    loss_i2a = F.cross_entropy(audio_image_sim.T, labels)
    loss_a2t = F.cross_entropy(audio_text_sim, labels)
    loss_t2a = F.cross_entropy(audio_text_sim.T, labels)
    loss_i2t = F.cross_entropy(image_text_sim, labels)
    loss_t2i = F.cross_entropy(image_text_sim.T, labels)
    
    return (loss_a2i + loss_i2a + loss_a2t + loss_t2a + loss_i2t + loss_t2i) / 6
```

## 🎬 Video Understanding

### Video-Language Models

#### VideoBERT
```python
class VideoBERT(nn.Module):
    def __init__(self, video_vocab_size, text_vocab_size, hidden_size):
        super().__init__()
        
        # Video tokenizer (learned or pre-defined)
        self.video_tokenizer = VideoTokenizer(video_vocab_size)
        
        # Joint embedding space
        self.video_embeddings = nn.Embedding(video_vocab_size, hidden_size)
        self.text_embeddings = nn.Embedding(text_vocab_size, hidden_size)
        
        # Unified transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=12
        )
        
        # Output heads
        self.video_head = nn.Linear(hidden_size, video_vocab_size)
        self.text_head = nn.Linear(hidden_size, text_vocab_size)
    
    def forward(self, video_tokens, text_tokens, modality_mask):
        # Embed tokens
        video_embeds = self.video_embeddings(video_tokens)
        text_embeds = self.text_embeddings(text_tokens)
        
        # Concatenate sequences
        sequence = torch.cat([video_embeds, text_embeds], dim=1)
        
        # Add modality embeddings
        modality_embeds = self.modality_embeddings(modality_mask)
        sequence = sequence + modality_embeds
        
        # Transformer
        hidden_states = self.transformer(sequence)
        
        # Split back to video and text
        video_hidden = hidden_states[:, :video_tokens.size(1)]
        text_hidden = hidden_states[:, video_tokens.size(1):]
        
        # Output logits
        video_logits = self.video_head(video_hidden)
        text_logits = self.text_head(text_hidden)
        
        return video_logits, text_logits
```

#### Video Tokenization
```python
class VideoTokenizer(nn.Module):
    def __init__(self, vocab_size, temporal_dim=16, spatial_dim=14):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 3D CNN for spatiotemporal features
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(64, 128, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((temporal_dim, spatial_dim, spatial_dim))
        )
        
        # Quantization layer
        self.quantize = VectorQuantizer(vocab_size, 128)
    
    def forward(self, video):
        # Extract features
        features = self.feature_extractor(video)  # (B, 128, T, H, W)
        
        # Quantize to discrete tokens
        quantized, tokens, commitment_loss = self.quantize(features)
        
        return tokens, commitment_loss

class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Codebook
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-1/vocab_size, 1/vocab_size)
    
    def forward(self, z):
        # Flatten spatial dimensions
        z_flattened = z.view(-1, self.embed_dim)
        
        # Find closest embedding
        distances = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view_as(z)
        
        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), z)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, encoding_indices, commitment_loss
```

## 🔗 Cross-Modal Fusion Techniques

### Early Fusion
```python
def early_fusion(audio_features, video_features):
    """
    Concatenate features at input level.
    Simple but may not capture complex interactions.
    """
    return torch.cat([audio_features, video_features], dim=-1)
```

### Late Fusion
```python
def late_fusion(audio_logits, video_logits, weights=[0.5, 0.5]):
    """
    Combine predictions from separate modalities.
    Each modality processed independently.
    """
    return weights[0] * audio_logits + weights[1] * video_logits
```

### Cross-Modal Attention
```python
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, query_modality, key_value_modality):
        # Cross-attention
        attended, _ = self.attention(
            query_modality, key_value_modality, key_value_modality
        )
        query_modality = self.norm1(query_modality + attended)
        
        # Feed-forward
        ffn_out = self.ffn(query_modality)
        query_modality = self.norm2(query_modality + ffn_out)
        
        return query_modality
```

### Multimodal Transformer
```python
class MultimodalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        
        # Modality-specific projections
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Modality embeddings
        self.modality_embed = nn.Embedding(3, d_model)  # audio, video, text
        
        # Unified transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
            n_layers
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, audio, video, text):
        # Project to common space
        audio_tokens = self.audio_proj(audio)
        video_tokens = self.video_proj(video)
        text_tokens = self.text_proj(text)
        
        # Add modality embeddings
        audio_tokens += self.modality_embed(torch.zeros(audio.size(0), audio.size(1), dtype=torch.long))
        video_tokens += self.modality_embed(torch.ones(video.size(0), video.size(1), dtype=torch.long))
        text_tokens += self.modality_embed(torch.full((text.size(0), text.size(1)), 2, dtype=torch.long))
        
        # Concatenate all modalities
        multimodal_sequence = torch.cat([audio_tokens, video_tokens, text_tokens], dim=1)
        
        # Process with transformer
        hidden_states = self.transformer(multimodal_sequence)
        
        # Global pooling and classification
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)
```

## 🎯 Training Strategies

### Curriculum Learning for Multimodal
```python
def multimodal_curriculum(epoch, total_epochs):
    """
    Start with unimodal tasks, gradually add multimodal complexity.
    """
    if epoch < total_epochs * 0.2:
        # Phase 1: Unimodal pretraining
        return {
            'modalities': ['text'],
            'tasks': ['language_modeling'],
            'difficulty': 'easy'
        }
    elif epoch < total_epochs * 0.5:
        # Phase 2: Pairwise multimodal
        return {
            'modalities': ['text', 'image'],
            'tasks': ['image_captioning', 'vqa'],
            'difficulty': 'medium'
        }
    else:
        # Phase 3: Full multimodal
        return {
            'modalities': ['text', 'image', 'audio'],
            'tasks': ['multimodal_understanding', 'generation'],
            'difficulty': 'hard'
        }
```

### Modality-Balanced Sampling
```python
class ModalityBalancedSampler:
    def __init__(self, dataset, modality_weights=None):
        self.dataset = dataset
        self.modality_counts = self._count_modalities()
        
        if modality_weights is None:
            # Equal sampling across modalities
            total = sum(self.modality_counts.values())
            self.modality_weights = {k: total/v for k, v in self.modality_counts.items()}
        else:
            self.modality_weights = modality_weights
    
    def _count_modalities(self):
        counts = {}
        for item in self.dataset:
            modality = item['modality_type']
            counts[modality] = counts.get(modality, 0) + 1
        return counts
    
    def __iter__(self):
        # Sample with probability proportional to inverse frequency
        indices = []
        for i, item in enumerate(self.dataset):
            modality = item['modality_type']
            weight = self.modality_weights[modality]
            if random.random() < weight:
                indices.append(i)
        
        random.shuffle(indices)
        return iter(indices)
```

### Gradual Modality Integration
```python
def gradual_modality_integration(model, batch, epoch, total_epochs):
    """
    Gradually introduce modalities during training.
    """
    integration_schedule = {
        0.0: ['text'],
        0.3: ['text', 'image'], 
        0.6: ['text', 'image', 'audio'],
        0.8: ['text', 'image', 'audio', 'video']
    }
    
    progress = epoch / total_epochs
    
    # Find current modalities
    active_modalities = ['text']  # Always include text
    for threshold, modalities in integration_schedule.items():
        if progress >= threshold:
            active_modalities = modalities
    
    # Mask inactive modalities
    masked_batch = {}
    for modality, data in batch.items():
        if modality in active_modalities:
            masked_batch[modality] = data
        else:
            masked_batch[modality] = torch.zeros_like(data)
    
    return model(masked_batch)
```

## 📊 Evaluation Metrics

### Vision-Language Tasks
```python
def evaluate_image_captioning(generated_captions, reference_captions):
    """
    Standard metrics for image captioning.
    """
    from pycocoevalcap.eval import COCOEvalCap
    
    metrics = {}
    
    # BLEU scores
    metrics['bleu'] = corpus_bleu(reference_captions, generated_captions)
    
    # ROUGE scores
    metrics['rouge'] = rouge_score(reference_captions, generated_captions)
    
    # CIDEr (Consensus-based Image Description Evaluation)
    metrics['cider'] = cider_score(reference_captions, generated_captions)
    
    # SPICE (Semantic Propositional Image Caption Evaluation)
    metrics['spice'] = spice_score(reference_captions, generated_captions)
    
    return metrics

def evaluate_vqa(predictions, ground_truth):
    """
    VQA accuracy metric.
    """
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truth):
        # VQA uses soft accuracy (answer might appear multiple times)
        if isinstance(gt, list):
            score = min(gt.count(pred) / 3.0, 1.0)
        else:
            score = 1.0 if pred == gt else 0.0
        correct += score
    
    return correct / total
```

### Cross-Modal Retrieval
```python
def evaluate_retrieval(query_embeddings, database_embeddings, labels, k_values=[1, 5, 10]):
    """
    Evaluate cross-modal retrieval performance.
    """
    # Compute similarity matrix
    similarity = torch.matmul(query_embeddings, database_embeddings.T)
    
    # Get top-k indices
    _, top_k_indices = torch.topk(similarity, max(k_values), dim=1)
    
    metrics = {}
    for k in k_values:
        # Recall@k
        top_k = top_k_indices[:, :k]
        correct = (top_k == labels.unsqueeze(1)).any(dim=1).float()
        metrics[f'recall@{k}'] = correct.mean().item()
        
        # Mean Reciprocal Rank
        ranks = (top_k == labels.unsqueeze(1)).nonzero()[:, 1] + 1
        if len(ranks) > 0:
            mrr = (1.0 / ranks.float()).mean().item()
        else:
            mrr = 0.0
        metrics['mrr'] = mrr
    
    return metrics
```

## 🎯 Interview Questions & Answers

### Q1: "How do you align representations from different modalities?"

**Answer Framework**:
1. **Contrastive learning**: CLIP-style positive/negative pairs
2. **Translation tasks**: Learn mappings between modalities
3. **Shared embedding space**: Project all modalities to common space
4. **Cross-modal attention**: Let modalities attend to each other
5. **Adversarial training**: Domain adaptation techniques

### Q2: "What are the challenges in multimodal fusion?"

**Answer Framework**:
1. **Modality imbalance**: Different modalities may dominate
2. **Temporal alignment**: Synchronizing sequences of different lengths
3. **Missing modalities**: Handling incomplete inputs gracefully
4. **Scale differences**: Modalities may have different dynamic ranges
5. **Semantic gaps**: Different levels of abstraction across modalities

### Q3: "How would you handle missing modalities during inference?"

**Answer Framework**:
1. **Modality dropout**: Train with random modality masking
2. **Reconstruction**: Learn to reconstruct missing modalities
3. **Adaptive fusion**: Dynamically weight available modalities
4. **Modality-specific networks**: Fall back to unimodal processing
5. **Uncertainty modeling**: Express confidence based on available information

### Q4: "Explain the trade-offs between early, late, and intermediate fusion."

**Answer Framework**:

**Early Fusion**:
- Pros: Simple, learns joint representations from start
- Cons: May not capture complex cross-modal interactions

**Late Fusion**:
- Pros: Modality-specific optimization, interpretable
- Cons: Limited cross-modal interaction

**Intermediate Fusion**:
- Pros: Best of both worlds, flexible interaction
- Cons: More complex, requires careful design

## 🚀 Advanced Topics

### Multimodal Chain-of-Thought
```python
def multimodal_chain_of_thought(model, image, question):
    """
    Generate reasoning steps using multiple modalities.
    """
    prompt = f"""
    Image: [IMAGE]
    Question: {question}
    
    Let me think step by step:
    1. What do I see in the image?
    2. What does the question ask?
    3. How does the visual information relate to the question?
    4. What is my final answer?
    """
    
    response = model.generate(
        image=image,
        text=prompt,
        max_length=200
    )
    
    return response
```

### Tool-Augmented Multimodal Models
```python
class ToolAugmentedMultimodalModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Available tools
        self.tools = {
            'object_detector': ObjectDetector(),
            'ocr': OCREngine(),
            'depth_estimator': DepthEstimator(),
            'calculator': Calculator()
        }
        
        self.tool_selector = nn.Linear(base_model.hidden_size, len(self.tools))
    
    def forward(self, inputs):
        # Generate initial response
        initial_response = self.base_model(inputs)
        
        # Decide if tools are needed
        tool_scores = self.tool_selector(initial_response.last_hidden_state)
        
        if tool_scores.max() > threshold:
            # Use appropriate tool
            tool_name = list(self.tools.keys())[tool_scores.argmax()]
            tool_output = self.tools[tool_name](inputs)
            
            # Incorporate tool output
            augmented_inputs = self.incorporate_tool_output(inputs, tool_output)
            final_response = self.base_model(augmented_inputs)
            
            return final_response
        else:
            return initial_response
```

---

## 📝 Study Checklist

- [ ] Understand CLIP architecture and contrastive learning
- [ ] Know vision-language model variants (DALL-E 2, BLIP, LLaVA)
- [ ] Can explain different fusion techniques and their trade-offs
- [ ] Familiar with audio-visual models and applications
- [ ] Understand video-language models and tokenization
- [ ] Know evaluation metrics for multimodal tasks
- [ ] Can design training strategies for multimodal models
- [ ] Understand challenges and solutions for missing modalities

**Next**: [Research Methodology →](../02-research-implementation/05-research-methodology.md)
