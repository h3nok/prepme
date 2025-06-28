import { LearningModule } from '../types/LearningModule';

export const genAIInterviewModule: LearningModule = {
  id: 'genai-interview',
  title: 'GenAI Interview Mastery',
  description: 'Comprehensive preparation for Generative AI interviews. Master the concepts, architectures, and techniques that interviewers ask about most frequently.',
  color: '#8b5cf6',
  icon: 'Zap',
  progress: 0,
  estimatedHours: 20,
  prerequisites: ['fundamentals', 'transformers', 'llms'],
  difficulty: 'Expert',
  concepts: [
    {
      id: 'transformer-deep-dive',
      title: 'Transformer Architecture Interview Deep Dive',
      description: 'Master every component of transformers that interviewers love to ask about',
      slides: [
        {
          id: 'attention-mechanism-interview',
          title: 'Self-Attention: The Heart of Modern AI',
          content: {
            tier1: "Self-attention allows each position to attend to all positions in the sequence. This enables parallel processing and captures long-range dependencies that RNNs struggle with.",
            tier2: "The mechanism computes queries, keys, and values from the input, then uses dot-product attention with scaling to compute attention weights. The softmax ensures weights sum to 1.",
            tier3: "Multi-head attention runs multiple attention functions in parallel, allowing the model to jointly attend to information from different representation subspaces at different positions."
          },
          mathNotations: [
            {
              id: 'attention-formula',
              latex: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
              explanation: 'Core attention formula. The √d_k scaling prevents softmax saturation in high dimensions',
              interactive: true
            },
            {
              id: 'multihead-attention',
              latex: '\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O',
              explanation: 'Multi-head attention concatenates multiple attention heads and projects the result',
              interactive: false
            }
          ],
          keyPoints: [
            "Self-attention enables parallel processing vs sequential RNNs",
            "Scaling by √d_k prevents gradient vanishing in high dimensions",
            "Multi-head attention captures different types of relationships",
            "Causal masking in decoders prevents looking ahead",
            "Attention weights provide interpretability into model focus"
          ],
          interviewTips: [
            "Always mention the computational complexity: O(n²d) for sequence length n",
            "Explain why RNNs are insufficient for long sequences (vanishing gradients)",
            "Discuss the parallelization advantage over recurrent architectures",
            "Be ready to explain the intuition behind multi-head attention"
          ],
          practiceQuestions: [
            "Why do we scale attention weights by √d_k?",
            "How does self-attention handle variable-length sequences?",
            "What's the computational complexity of self-attention?",
            "Explain the difference between encoder and decoder attention"
          ]
        },
        {
          id: 'positional-encoding-interview',
          title: 'Positional Encoding: Teaching Position to Transformers',
          content: {
            tier1: "Transformers process all positions simultaneously, losing sequence order information. Positional encoding adds position information to token embeddings.",
            tier2: "Sinusoidal encoding uses sine and cosine functions of different frequencies. This allows the model to learn relative positions and extrapolate to longer sequences.",
            tier3: "Modern alternatives include learned positional embeddings, relative position encoding (T5), and rotary position embedding (RoPE) which shows better length extrapolation."
          },
          mathNotations: [
            {
              id: 'sinusoidal-encoding',
              latex: 'PE_{(pos, 2i)} = \\sin(pos/10000^{2i/d_{model}})',
              explanation: 'Sinusoidal positional encoding for even dimensions',
              interactive: true
            },
            {
              id: 'rope-formula',
              latex: 'f_q(x_m, m) = R_m W_q x_m',
              explanation: 'RoPE applies rotation matrices based on position m',
              interactive: true
            }
          ],
          keyPoints: [
            "Positional encoding is added to input embeddings, not concatenated",
            "Sinusoidal encoding allows extrapolation to unseen lengths",
            "RoPE shows better performance on long sequences",
            "Absolute vs relative position encoding trade-offs",
            "Position information is crucial for language understanding"
          ],
          interviewTips: [
            "Explain why transformers need explicit position information",
            "Compare different positional encoding schemes",
            "Discuss length extrapolation capabilities",
            "Be ready to draw the sinusoidal encoding pattern"
          ],
          practiceQuestions: [
            "Why can't transformers handle sequences without positional encoding?",
            "How does RoPE improve upon sinusoidal encoding?",
            "What happens if you remove positional encoding?",
            "Explain the periodicity in sinusoidal encodings"
          ]
        },
        {
          id: 'layer-normalization-interview',
          title: 'Layer Normalization & Training Stability',
          content: {
            tier1: "Layer normalization normalizes inputs across features for each sample, stabilizing training. It's applied before or after each sub-layer in transformers.",
            tier2: "Pre-norm vs post-norm placement affects training dynamics. Pre-norm (used in GPT) is more stable, while post-norm (original Transformer) can achieve slightly better performance.",
            tier3: "RMSNorm removes the mean centering operation for efficiency. Other variants like GroupNorm and LayerScale further improve training stability and performance."
          },
          mathNotations: [
            {
              id: 'layer-norm',
              latex: '\\text{LayerNorm}(x) = \\gamma \\frac{x - \\mu}{\\sigma} + \\beta',
              explanation: 'Layer normalization formula with learnable scale γ and shift β parameters',
              interactive: false
            }
          ],
          keyPoints: [
            "Layer norm operates across features, batch norm across samples",
            "Pre-norm placement improves training stability",
            "Residual connections enable deep network training",
            "RMSNorm provides computational efficiency",
            "Normalization placement affects gradient flow"
          ],
          interviewTips: [
            "Explain the difference between layer norm and batch norm",
            "Discuss why batch norm doesn't work well for transformers",
            "Compare pre-norm vs post-norm architectures",
            "Mention recent innovations like RMSNorm"
          ],
          practiceQuestions: [
            "Why is layer norm preferred over batch norm in transformers?",
            "What's the difference between pre-norm and post-norm?",
            "How do residual connections help with training?",
            "Explain the vanishing gradient problem layer norm solves"
          ]
        }
      ]
    },
    {
      id: 'llm-training-interview',
      title: 'LLM Training & Optimization Interview Focus',
      description: 'Deep dive into training methodologies, scaling laws, and optimization techniques',
      slides: [
        {
          id: 'pretraining-strategies',
          title: 'Pre-training Strategies & Data Engineering',
          content: {
            tier1: "Pre-training on massive text corpora teaches language understanding and world knowledge. Data quality, diversity, and preprocessing are critical for model performance.",
            tier2: "Modern pre-training involves sophisticated data pipelines: deduplication (exact and fuzzy), quality filtering, toxicity removal, and careful dataset mixing ratios.",
            tier3: "Advanced techniques include curriculum learning (easy to hard examples), data selection based on model perplexity, and multi-lingual training strategies."
          },
          keyPoints: [
            "Data quality matters more than quantity alone",
            "Deduplication prevents overfitting and improves generalization",
            "Dataset mixing ratios affect final model capabilities",
            "Tokenization choices impact model efficiency and performance",
            "Compute-optimal training balances model size and data size"
          ],
          interviewTips: [
            "Discuss the importance of data preprocessing pipelines",
            "Explain different deduplication strategies (exact, fuzzy, semantic)",
            "Mention specific datasets like Common Crawl, Books, Wikipedia",
            "Be ready to discuss data contamination and evaluation concerns"
          ],
          practiceQuestions: [
            "How do you ensure training data quality at scale?",
            "What are the challenges with web-scraped training data?",
            "Explain different tokenization strategies and their trade-offs",
            "How do you prevent data leakage in evaluation?"
          ]
        },
        {
          id: 'scaling-laws-interview',
          title: 'Scaling Laws: The Science Behind Model Size',
          content: {
            tier1: "Scaling laws predict how model performance improves with size, data, and compute. They guide fundamental decisions about resource allocation in training.",
            tier2: "Key findings: Performance scales as power laws in model parameters, dataset size, and compute. Chinchilla revealed most models were undertrained on data.",
            tier3: "Optimal compute allocation follows N_opt ∝ C^0.5 and D_opt ∝ C^0.5, meaning model size and data should scale equally with compute budget."
          },
          mathNotations: [
            {
              id: 'scaling-law-formula',
              latex: 'L(N, D, C) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta} + \\frac{G}{C^\\gamma}',
              explanation: 'Generalized scaling law including compute dependence',
              interactive: true
            },
            {
              id: 'chinchilla-optimal',
              latex: 'N_{opt} = G \\cdot C^{a}, \\quad D_{opt} = H \\cdot C^{b}',
              explanation: 'Chinchilla optimal scaling with a ≈ b ≈ 0.5',
              interactive: true
            }
          ],
          keyPoints: [
            "Power law relationships hold across orders of magnitude",
            "Compute-optimal training requires balanced scaling",
            "Chinchilla showed models were data-undertrained",
            "Emergent capabilities appear at predictable scales",
            "Inference costs scale with model size, not training cost"
          ],
          interviewTips: [
            "Explain the Chinchilla findings and their implications",
            "Discuss the trade-off between training and inference costs",
            "Mention specific scaling exponents (α ≈ 0.076, β ≈ 0.095)",
            "Be ready to explain emergent capabilities and their thresholds"
          ],
          practiceQuestions: [
            "Why did Chinchilla challenge previous scaling assumptions?",
            "How do you determine optimal model size for a given budget?",
            "What causes emergent capabilities to appear suddenly?",
            "Explain the difference between training and inference scaling"
          ]
        },
        {
          id: 'optimization-techniques',
          title: 'Advanced Optimization & Training Techniques',
          content: {
            tier1: "Training large language models requires sophisticated optimization techniques: learning rate scheduling, gradient clipping, and mixed precision training.",
            tier2: "Key techniques include warmup schedules, cosine annealing, gradient accumulation for large batch sizes, and careful initialization schemes.",
            tier3: "Advanced methods: gradient checkpointing for memory efficiency, model parallelism strategies, and techniques like gradient centralization and LAMB optimizer."
          },
          mathNotations: [
            {
              id: 'adam-optimizer',
              latex: 'm_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t, \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2',
              explanation: 'Adam optimizer momentum and velocity updates',
              interactive: false
            },
            {
              id: 'gradient-clipping',
              latex: '\\tilde{g} = \\min\\left(1, \\frac{\\tau}{||g||_2}\\right) \\cdot g',
              explanation: 'Gradient clipping prevents training instability',
              interactive: false
            }
          ],
          keyPoints: [
            "Learning rate warmup prevents early training instability",
            "Gradient clipping is essential for transformer training",
            "Mixed precision training speeds up training and saves memory",
            "Batch size scaling requires careful learning rate adjustment",
            "Model parallelism enables training larger models"
          ],
          interviewTips: [
            "Explain why transformers need gradient clipping",
            "Discuss the benefits and challenges of mixed precision",
            "Compare different learning rate schedules",
            "Mention specific optimizers like AdamW and their advantages"
          ],
          practiceQuestions: [
            "Why is gradient clipping necessary for transformer training?",
            "How does mixed precision training work?",
            "Explain the learning rate warmup schedule and its purpose",
            "What are the challenges with very large batch sizes?"
          ]
        }
      ]
    },
    {
      id: 'fine-tuning-alignment',
      title: 'Fine-tuning & Alignment Interview Deep Dive',
      description: 'Master parameter-efficient fine-tuning, RLHF, and alignment techniques',
      slides: [
        {
          id: 'parameter-efficient-finetuning',
          title: 'Parameter-Efficient Fine-tuning (PEFT)',
          content: {
            tier1: "PEFT methods fine-tune only a small subset of parameters while keeping the base model frozen. This enables efficient adaptation to new tasks with minimal computational cost.",
            tier2: "LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices. Adapters add small bottleneck layers. Prompt tuning optimizes only the input embeddings.",
            tier3: "Advanced PEFT includes QLoRA (quantized LoRA), AdaLoRA (adaptive rank allocation), and IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)."
          },
          mathNotations: [
            {
              id: 'lora-decomposition',
              latex: 'W = W_0 + \\Delta W = W_0 + BA',
              explanation: 'LoRA represents weight updates as product of low-rank matrices B and A',
              interactive: true
            },
            {
              id: 'lora-forward',
              latex: 'h = W_0 x + \\Delta W x = W_0 x + B A x',
              explanation: 'Forward pass with LoRA adaptation',
              interactive: false
            }
          ],
          keyPoints: [
            "LoRA reduces trainable parameters by 1000x with minimal performance loss",
            "Rank r determines the trade-off between efficiency and expressivity",
            "Multiple LoRA adapters can be trained for different tasks",
            "QLoRA enables fine-tuning of larger models on consumer hardware",
            "PEFT methods preserve the base model's general capabilities"
          ],
          interviewTips: [
            "Explain the mathematical intuition behind low-rank decomposition",
            "Compare LoRA with other PEFT methods like adapters",
            "Discuss the rank selection strategy in LoRA",
            "Mention practical applications like multi-task learning"
          ],
          practiceQuestions: [
            "How does LoRA achieve parameter efficiency?",
            "What determines the optimal rank in LoRA?",
            "Compare LoRA, adapters, and prompt tuning",
            "How does QLoRA combine quantization with LoRA?"
          ]
        },
        {
          id: 'rlhf-deep-dive',
          title: 'Reinforcement Learning from Human Feedback (RLHF)',
          content: {
            tier1: "RLHF aligns language models with human preferences through a three-stage process: supervised fine-tuning, reward model training, and RL optimization.",
            tier2: "Stage 1: SFT on high-quality demonstrations. Stage 2: Train reward model on human preference data. Stage 3: Use PPO to optimize the policy against the reward model.",
            tier3: "Key challenges include reward hacking, distributional shift, and scaling oversight. Advanced techniques include constitutional AI, debate, and recursive reward modeling."
          },
          mathNotations: [
            {
              id: 'rlhf-objective',
              latex: 'J(\\pi) = \\mathbb{E}_{x \\sim D, y \\sim \\pi(\\cdot|x)}[r(x, y)] - \\beta \\cdot D_{KL}(\\pi(y|x) || \\pi_{ref}(y|x))',
              explanation: 'RLHF objective balancing reward and KL penalty from reference model',
              interactive: true
            },
            {
              id: 'reward-model',
              latex: 'r_\\theta(x, y) = \\text{Transformer}_\\theta(x, y)',
              explanation: 'Reward model parameterized as a transformer',
              interactive: false
            }
          ],
          keyPoints: [
            "Human preferences are more accessible than rewards for language tasks",
            "KL regularization prevents the model from drifting too far from base",
            "Reward model quality critically affects final performance",
            "PPO provides stable policy optimization for language models",
            "Constitutional AI scales human oversight using AI feedback"
          ],
          interviewTips: [
            "Explain each stage of the RLHF pipeline clearly",
            "Discuss the importance of KL regularization",
            "Mention specific implementations like InstructGPT",
            "Compare RLHF with supervised fine-tuning approaches"
          ],
          practiceQuestions: [
            "Why is RLHF necessary beyond supervised fine-tuning?",
            "How does the KL penalty prevent reward hacking?",
            "What are the key challenges in reward model training?",
            "Explain the role of PPO in RLHF"
          ]
        },
        {
          id: 'constitutional-ai',
          title: 'Constitutional AI & Advanced Alignment',
          content: {
            tier1: "Constitutional AI (CAI) uses AI feedback instead of human feedback for alignment. The model learns to follow a constitution of principles through self-critique and revision.",
            tier2: "CAI has two phases: Supervised learning to critique and revise responses, and RL from AI feedback using the constitutional principles as reward signal.",
            tier3: "Advanced alignment research includes debate (models argue different sides), amplification (break down complex tasks), and scalable oversight for superintelligent systems."
          },
          keyPoints: [
            "Constitutional AI reduces dependence on human labelers",
            "Self-critique and revision improve response quality",
            "AI feedback can scale better than human feedback",
            "Constitutional principles provide interpretable alignment",
            "Debate and amplification aim to solve scalable oversight"
          ],
          interviewTips: [
            "Explain how CAI differs from traditional RLHF",
            "Discuss the advantages of AI feedback over human feedback",
            "Mention specific constitutional principles used",
            "Connect to broader AI safety and alignment goals"
          ],
          practiceQuestions: [
            "How does Constitutional AI work without human feedback?",
            "What are the advantages of AI feedback scaling?",
            "Explain the concept of scalable oversight",
            "How do debate and amplification contribute to alignment?"
          ]
        }
      ]
    },
    {
      id: 'generative-models-interview',
      title: 'Generative Models: Diffusion, VAEs, GANs',
      description: 'Comprehensive coverage of generative modeling approaches for interviews',
      slides: [
        {
          id: 'diffusion-models-interview',
          title: 'Diffusion Models: The New King of Generation',
          content: {
            tier1: "Diffusion models learn to reverse a gradual noising process. They start with data, add noise until it becomes Gaussian, then learn to denoise step by step.",
            tier2: "DDPM defines a forward process q(x_t|x_{t-1}) that adds Gaussian noise, and learns a reverse process p_θ(x_{t-1}|x_t) to generate samples.",
            tier3: "Key innovations include variance scheduling, improved sampling (DDIM), guided generation, and latent diffusion for computational efficiency."
          },
          mathNotations: [
            {
              id: 'ddpm-forward',
              latex: 'q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)',
              explanation: 'Forward diffusion process with noise schedule β_t',
              interactive: true
            },
            {
              id: 'ddpm-loss',
              latex: 'L = \\mathbb{E}_{t, x_0, \\epsilon} [||\\epsilon - \\epsilon_\\theta(x_t, t)||^2]',
              explanation: 'DDPM training objective: predict the noise added at each step',
              interactive: true
            }
          ],
          keyPoints: [
            "Diffusion models achieve state-of-the-art image generation quality",
            "Training is stable compared to GANs (no adversarial training)",
            "Sampling requires many denoising steps (slow inference)",
            "Classifier-free guidance enables conditional generation",
            "Latent diffusion reduces computational requirements"
          ],
          interviewTips: [
            "Explain the intuition behind the forward and reverse processes",
            "Compare diffusion models with GANs and VAEs",
            "Discuss the trade-off between quality and sampling speed",
            "Mention recent improvements like DDIM and LDM"
          ],
          practiceQuestions: [
            "How do diffusion models differ from GANs?",
            "Why is the noise schedule important in diffusion models?",
            "Explain classifier-free guidance",
            "What makes latent diffusion more efficient?"
          ]
        },
        {
          id: 'model-comparison-interview',
          title: 'Generative Models: VAEs vs GANs vs Diffusion',
          content: {
            tier1: "Each generative model type has distinct advantages: VAEs for smooth interpolation, GANs for sharp images, and diffusion for stable training and high quality.",
            tier2: "VAEs optimize a tractable lower bound but produce blurry images. GANs generate sharp images through adversarial training but suffer from mode collapse. Diffusion provides stability and quality.",
            tier3: "Modern approaches combine strengths: VQ-VAE for discrete representations, StyleGAN for controllable generation, and guided diffusion for conditional synthesis."
          },
          keyPoints: [
            "VAEs: Explicit likelihood, smooth latent space, blurry outputs",
            "GANs: Sharp outputs, adversarial training, mode collapse issues",
            "Diffusion: Stable training, high quality, slow sampling",
            "Each model suits different applications and requirements",
            "Hybrid approaches combine advantages of multiple paradigms"
          ],
          interviewTips: [
            "Create a clear comparison table of the three approaches",
            "Discuss when to use each type of model",
            "Mention specific architectures (StyleGAN, β-VAE, DDPM)",
            "Explain the fundamental trade-offs in generative modeling"
          ],
          practiceQuestions: [
            "When would you choose a VAE over a GAN?",
            "What causes mode collapse in GANs?",
            "How do diffusion models achieve training stability?",
            "Compare the latent spaces of VAEs and GANs"
          ]
        }
      ]
    },
    {
      id: 'practical-deployment',
      title: 'Production ML & Model Deployment',
      description: 'Real-world deployment challenges and solutions for GenAI systems',
      slides: [
        {
          id: 'model-serving-interview',
          title: 'Model Serving & Inference Optimization',
          content: {
            tier1: "Serving large language models requires careful optimization for latency, throughput, and cost. Key techniques include batching, caching, and model compression.",
            tier2: "Dynamic batching groups requests with similar lengths. KV caching stores attention keys/values. Model pruning, quantization, and distillation reduce model size.",
            tier3: "Advanced techniques include speculative decoding, parallel sampling, and hardware-specific optimizations like tensor parallelism and pipeline parallelism."
          },
          keyPoints: [
            "Dynamic batching improves GPU utilization for variable-length inputs",
            "KV caching is essential for efficient autoregressive generation",
            "Quantization can reduce model size by 4x with minimal quality loss",
            "Model parallelism enables serving models larger than single GPU memory",
            "Caching strategies significantly reduce computational costs"
          ],
          interviewTips: [
            "Explain the memory requirements for large model inference",
            "Discuss different quantization techniques (INT8, INT4, FP16)",
            "Compare model parallelism strategies",
            "Mention specific frameworks like vLLM, TensorRT-LLM"
          ],
          practiceQuestions: [
            "How does KV caching work in transformer inference?",
            "What are the trade-offs of model quantization?",
            "Explain tensor parallelism vs pipeline parallelism",
            "How do you optimize batch processing for variable-length sequences?"
          ]
        },
        {
          id: 'monitoring-evaluation',
          title: 'Monitoring & Evaluation in Production',
          content: {
            tier1: "Production GenAI systems require comprehensive monitoring: model performance, data drift, safety violations, and user satisfaction metrics.",
            tier2: "Key metrics include perplexity for language models, BLEU/ROUGE for generation tasks, and human evaluation for alignment. Online A/B testing validates improvements.",
            tier3: "Advanced monitoring includes prompt injection detection, toxicity filtering, factual accuracy checking, and real-time bias detection across demographic groups."
          },
          keyPoints: [
            "Automated metrics don't capture all aspects of model quality",
            "Human evaluation remains crucial for subjective tasks",
            "Safety monitoring includes toxicity, bias, and prompt injection",
            "Data drift can degrade model performance over time",
            "A/B testing validates model improvements in production"
          ],
          interviewTips: [
            "Discuss the limitations of automated evaluation metrics",
            "Explain different types of drift (data, concept, model)",
            "Mention specific safety concerns for LLMs",
            "Compare offline vs online evaluation strategies"
          ],
          practiceQuestions: [
            "How do you evaluate the quality of generated text?",
            "What is prompt injection and how do you detect it?",
            "Explain different types of model drift",
            "How do you design A/B tests for generative models?"
          ]
        }
      ]
    }
  ]
};
