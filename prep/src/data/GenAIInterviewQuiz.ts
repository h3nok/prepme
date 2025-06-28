import { Quiz } from '../types/LearningModule';

export const genAIInterviewQuiz: Quiz = {
  id: 'genai-interview-quiz',
  title: 'GenAI Interview Questions',
  description: 'Comprehensive quiz covering the most common GenAI interview questions across all major topics',
  moduleId: 'genai-interview',
  timeLimit: 45,
  passingScore: 80,
  questions: [
    // Transformer Architecture Questions
    {
      id: 'attention-complexity',
      type: 'multiple-choice',
      question: 'What is the computational complexity of self-attention for a sequence of length n with model dimension d?',
      options: [
        'O(n * d)',
        'O(n² * d)',
        'O(n * d²)',
        'O(n³)'
      ],
      correctAnswer: 1,
      explanation: 'Self-attention computes n×n attention weights (O(n²)) and for each position computes d-dimensional values, resulting in O(n²×d) complexity. This quadratic scaling is why long sequences are computationally expensive.',
      difficulty: 'Intermediate',
      category: 'Architecture',
      interviewFrequency: 'Very High'
    },
    {
      id: 'attention-scaling',
      type: 'multiple-choice',
      question: 'Why do we scale attention weights by √d_k in the attention mechanism?',
      options: [
        'To make the softmax function smoother',
        'To prevent gradients from vanishing',
        'To prevent the dot products from becoming too large',
        'All of the above'
      ],
      correctAnswer: 3,
      explanation: 'Scaling by √d_k prevents dot products from becoming too large (which would push softmax into saturation), keeps gradients well-behaved, and makes the softmax distribution smoother. All these effects contribute to better training stability.',
      difficulty: 'Intermediate',
      category: 'Architecture',
      interviewFrequency: 'Very High'
    },
    {
      id: 'positional-encoding',
      type: 'multiple-choice',
      question: 'What is the main advantage of sinusoidal positional encoding over learned positional embeddings?',
      options: [
        'Better performance on downstream tasks',
        'Ability to extrapolate to sequences longer than training',
        'Faster training convergence',
        'Lower memory requirements'
      ],
      correctAnswer: 1,
      explanation: 'Sinusoidal encoding allows the model to extrapolate to longer sequences than seen during training because the patterns are mathematically defined rather than learned for specific positions.',
      difficulty: 'Intermediate',
      category: 'Architecture',
      interviewFrequency: 'High'
    },
    {
      id: 'layer-norm-placement',
      type: 'multiple-choice',
      question: 'In modern transformer architectures, where is layer normalization typically placed and why?',
      options: [
        'After each sub-layer (post-norm) for better performance',
        'Before each sub-layer (pre-norm) for training stability',
        'Only at the beginning and end of the model',
        'It depends on the specific task'
      ],
      correctAnswer: 1,
      explanation: 'Pre-norm (LayerNorm before sub-layers) has become standard because it provides better training stability, especially for deep models, even though post-norm can sometimes achieve slightly better final performance.',
      difficulty: 'Advanced',
      category: 'Architecture',
      interviewFrequency: 'High'
    },
    
    // LLM Training Questions
    {
      id: 'chinchilla-scaling',
      type: 'multiple-choice',
      question: 'What was the key finding of the Chinchilla scaling laws paper?',
      options: [
        'Larger models always perform better',
        'Data quality matters more than quantity',
        'Most models were undertrained on data relative to their size',
        'Training time is the main bottleneck'
      ],
      correctAnswer: 2,
      explanation: 'Chinchilla found that for optimal performance, models should be trained on about 20 tokens per parameter, much more than previous models. Most large models were undertrained on data.',
      difficulty: 'Advanced',
      category: 'Training',
      interviewFrequency: 'Very High'
    },
    {
      id: 'gradient-clipping',
      type: 'multiple-choice',
      question: 'Why is gradient clipping particularly important when training transformers?',
      options: [
        'To speed up convergence',
        'To prevent exploding gradients due to attention mechanisms',
        'To reduce memory usage',
        'To improve final model performance'
      ],
      correctAnswer: 1,
      explanation: 'Transformers, especially with attention mechanisms, are prone to exploding gradients. Gradient clipping prevents training instability by capping the gradient norm.',
      difficulty: 'Intermediate',
      category: 'Training',
      interviewFrequency: 'High'
    },
    {
      id: 'emergent-capabilities',
      type: 'multiple-choice',
      question: 'At approximately what scale do in-context learning capabilities typically emerge in language models?',
      options: [
        '100M parameters',
        '1B parameters', 
        '10B parameters',
        '100B parameters'
      ],
      correctAnswer: 1,
      explanation: 'In-context learning typically emerges around 1B parameters, though the exact threshold varies by task and architecture. This is one of the most well-documented emergent capabilities.',
      difficulty: 'Advanced',
      category: 'Capabilities',
      interviewFrequency: 'High'
    },
    {
      id: 'mixed-precision',
      type: 'multiple-choice',
      question: 'What is the main benefit of mixed precision training for large language models?',
      options: [
        'Improved model accuracy',
        'Reduced training time and memory usage',
        'Better gradient flow',
        'Increased model interpretability'
      ],
      correctAnswer: 1,
      explanation: 'Mixed precision (FP16/BF16) reduces memory usage by ~50% and speeds up training on modern GPUs, while maintaining model quality through careful handling of gradients.',
      difficulty: 'Intermediate',
      category: 'Training',
      interviewFrequency: 'High'
    },
    
    // Fine-tuning and RLHF Questions
    {
      id: 'lora-parameters',
      type: 'multiple-choice',
      question: 'In LoRA fine-tuning, what does the rank parameter r control?',
      options: [
        'The learning rate',
        'The number of trainable parameters',
        'The batch size',
        'The sequence length'
      ],
      correctAnswer: 1,
      explanation: 'The rank r determines the dimensions of the low-rank matrices A and B. Higher rank means more trainable parameters and expressivity, but also more memory and computation.',
      difficulty: 'Intermediate',
      category: 'Fine-tuning',
      interviewFrequency: 'Very High'
    },
    {
      id: 'rlhf-stages',
      type: 'multiple-choice',
      question: 'What are the three main stages of RLHF (Reinforcement Learning from Human Feedback)?',
      options: [
        'Pre-training, Fine-tuning, Evaluation',
        'Data Collection, Model Training, Deployment',
        'Supervised Fine-tuning, Reward Model Training, RL Optimization',
        'Tokenization, Training, Inference'
      ],
      correctAnswer: 2,
      explanation: 'RLHF consists of: 1) Supervised fine-tuning on demonstrations, 2) Training a reward model on human preferences, 3) Using PPO to optimize the policy against the reward model.',
      difficulty: 'Advanced',
      category: 'Alignment',
      interviewFrequency: 'Very High'
    },
    {
      id: 'kl-penalty',
      type: 'multiple-choice',
      question: 'Why is a KL penalty included in the RLHF objective function?',
      options: [
        'To speed up training',
        'To prevent the model from drifting too far from the reference model',
        'To improve sample efficiency',
        'To reduce computational cost'
      ],
      correctAnswer: 1,
      explanation: 'The KL penalty prevents the policy from deviating too much from the reference model, avoiding reward hacking and maintaining the model\'s original capabilities.',
      difficulty: 'Advanced',
      category: 'Alignment',
      interviewFrequency: 'High'
    },
    {
      id: 'constitutional-ai',
      type: 'multiple-choice',
      question: 'How does Constitutional AI differ from traditional RLHF?',
      options: [
        'It uses larger models',
        'It replaces human feedback with AI feedback based on principles',
        'It requires more computational resources',
        'It only works for text generation tasks'
      ],
      correctAnswer: 1,
      explanation: 'Constitutional AI uses AI feedback based on a set of principles (constitution) instead of human feedback, making the alignment process more scalable.',
      difficulty: 'Advanced',
      category: 'Alignment',
      interviewFrequency: 'Medium'
    },
    
    // Diffusion Models Questions
    {
      id: 'ddpm-objective',
      type: 'multiple-choice',
      question: 'What does the DDPM training objective optimize for?',
      options: [
        'Maximizing data likelihood',
        'Minimizing reconstruction error',
        'Predicting the noise added at each timestep',
        'Minimizing the discriminator loss'
      ],
      correctAnswer: 2,
      explanation: 'DDPM training objective is simplified to predicting the noise ε that was added at each timestep, which is equivalent to learning the score function.',
      difficulty: 'Intermediate',
      category: 'Generative Models',
      interviewFrequency: 'High'
    },
    {
      id: 'ddim-advantage',
      type: 'multiple-choice',
      question: 'What is the main advantage of DDIM over DDPM sampling?',
      options: [
        'Better sample quality',
        'Faster sampling with fewer steps',
        'Lower memory requirements',
        'Easier training'
      ],
      correctAnswer: 1,
      explanation: 'DDIM enables faster sampling by making the reverse process deterministic and allowing larger timestep skips, reducing from 1000+ to 50-100 steps.',
      difficulty: 'Intermediate',
      category: 'Generative Models',
      interviewFrequency: 'High'
    },
    {
      id: 'classifier-free-guidance',
      type: 'multiple-choice',
      question: 'What is the key innovation of classifier-free guidance over classifier guidance?',
      options: [
        'Better sample quality',
        'No need for a separate classifier model',
        'Faster inference',
        'Lower training cost'
      ],
      correctAnswer: 1,
      explanation: 'Classifier-free guidance trains a single model to handle both conditional and unconditional generation, eliminating the need for separate classifier training.',
      difficulty: 'Advanced',
      category: 'Generative Models',
      interviewFrequency: 'High'
    },
    {
      id: 'latent-diffusion-benefit',
      type: 'multiple-choice',
      question: 'Why do Latent Diffusion Models perform diffusion in latent space rather than pixel space?',
      options: [
        'Better sample quality',
        'Reduced computational cost',
        'Easier training',
        'Better controllability'
      ],
      correctAnswer: 1,
      explanation: 'Operating in latent space reduces computational requirements by 8-64x while maintaining generation quality, enabling high-resolution image generation.',
      difficulty: 'Intermediate',
      category: 'Generative Models',
      interviewFrequency: 'High'
    },
    
    // Production and Deployment Questions
    {
      id: 'kv-caching',
      type: 'multiple-choice',
      question: 'What is the purpose of KV caching in transformer inference?',
      options: [
        'To reduce model size',
        'To avoid recomputing attention keys and values for previous tokens',
        'To improve model accuracy',
        'To enable parallel processing'
      ],
      correctAnswer: 1,
      explanation: 'KV caching stores the keys and values from previous tokens in autoregressive generation, avoiding redundant computation and significantly speeding up inference.',
      difficulty: 'Intermediate',
      category: 'Deployment',
      interviewFrequency: 'Very High'
    },
    {
      id: 'model-quantization',
      type: 'multiple-choice',
      question: 'What is the main trade-off when using INT8 quantization for model deployment?',
      options: [
        'Increased latency vs reduced accuracy',
        'Reduced memory usage vs potential accuracy loss',
        'Faster training vs slower inference',
        'Better interpretability vs higher cost'
      ],
      correctAnswer: 1,
      explanation: 'Quantization reduces memory usage and can speed up inference, but may cause some accuracy degradation. The trade-off is generally favorable for deployment.',
      difficulty: 'Intermediate',
      category: 'Deployment',
      interviewFrequency: 'High'
    },
    {
      id: 'dynamic-batching',
      type: 'multiple-choice',
      question: 'Why is dynamic batching particularly useful for serving language models?',
      options: [
        'It reduces model size',
        'It handles variable-length sequences efficiently',
        'It improves model accuracy',
        'It simplifies the codebase'
      ],
      correctAnswer: 1,
      explanation: 'Dynamic batching groups requests with similar sequence lengths, maximizing GPU utilization and throughput for the variable-length nature of text data.',
      difficulty: 'Intermediate',
      category: 'Deployment',
      interviewFrequency: 'High'
    },
    
    // Advanced Concepts
    {
      id: 'rope-encoding',
      type: 'multiple-choice',
      question: 'What advantage does RoPE (Rotary Position Embedding) provide over sinusoidal encoding?',
      options: [
        'Lower computational cost',
        'Better length extrapolation',
        'Easier implementation',
        'Smaller model size'
      ],
      correctAnswer: 1,
      explanation: 'RoPE provides better extrapolation to sequence lengths longer than those seen during training, making it superior for handling variable-length inputs.',
      difficulty: 'Advanced',
      category: 'Architecture',
      interviewFrequency: 'Medium'
    },
    {
      id: 'moe-architecture',
      type: 'multiple-choice',
      question: 'What is the main benefit of Mixture of Experts (MoE) architectures?',
      options: [
        'Reduced training time',
        'Increased model capacity without proportional compute increase',
        'Better interpretability',
        'Simpler deployment'
      ],
      correctAnswer: 1,
      explanation: 'MoE allows scaling model capacity by using sparse routing to activate only a subset of experts for each input, increasing parameters without proportional compute cost.',
      difficulty: 'Advanced',
      category: 'Architecture',
      interviewFrequency: 'Medium'
    }
  ]
};
