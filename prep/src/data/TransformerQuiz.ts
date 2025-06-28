export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correct: number;
  explanation: string;
  difficulty: 'easy' | 'medium' | 'hard';
  concept: string;
}

export const transformerQuiz: QuizQuestion[] = [
  // Attention Mechanism Questions
  {
    id: 'attention-1',
    question: 'What is the main innovation of the Transformer architecture?',
    options: [
      'Convolutional layers for sequence processing',
      'Self-attention mechanism for parallel processing',
      'Recurrent connections for memory',
      'Pooling layers for dimensionality reduction'
    ],
    correct: 1,
    explanation: 'The Transformer\'s key innovation is the self-attention mechanism that allows parallel processing of sequences, eliminating the sequential bottleneck of RNNs.',
    difficulty: 'easy',
    concept: 'attention-mechanism'
  },
  {
    id: 'attention-2',
    question: 'In the attention formula, what does the √d_k normalization factor prevent?',
    options: [
      'Overfitting during training',
      'Gradient vanishing problems',
      'Softmax saturation in high dimensions',
      'Memory overflow issues'
    ],
    correct: 2,
    explanation: 'The √d_k factor prevents the dot products from becoming too large in high dimensions, which would cause the softmax to saturate and produce very small gradients.',
    difficulty: 'hard',
    concept: 'attention-mechanism'
  },
  {
    id: 'attention-3',
    question: 'What are the three main components in the attention mechanism?',
    options: [
      'Input, Output, Hidden',
      'Query, Key, Value',
      'Source, Target, Context',
      'Encoder, Decoder, Bridge'
    ],
    correct: 1,
    explanation: 'The attention mechanism uses Query (what we\'re looking for), Key (what information is available), and Value (the actual content to retrieve).',
    difficulty: 'medium',
    concept: 'attention-mechanism'
  },
  {
    id: 'attention-4',
    question: 'What is the computational complexity of self-attention for a sequence of length n?',
    options: [
      'O(n)',
      'O(n log n)',
      'O(n²)',
      'O(n³)'
    ],
    correct: 2,
    explanation: 'Self-attention has O(n²) complexity because each position attends to all other positions, creating an n×n attention matrix.',
    difficulty: 'medium',
    concept: 'attention-mechanism'
  },

  // Multi-Head Attention Questions
  {
    id: 'multihead-1',
    question: 'Why do Transformers use multiple attention heads?',
    options: [
      'To increase model capacity',
      'To capture different types of relationships simultaneously',
      'To reduce computational complexity',
      'To improve gradient flow'
    ],
    correct: 1,
    explanation: 'Multiple heads allow the model to simultaneously capture different types of relationships: syntactic, semantic, positional, and long-range dependencies.',
    difficulty: 'medium',
    concept: 'multi-head-attention'
  },
  {
    id: 'multihead-2',
    question: 'How is the dimension split among attention heads?',
    options: [
      'Each head gets the full dimension',
      'The dimension is divided equally among heads',
      'Heads get random dimensions',
      'The first head gets most of the dimension'
    ],
    correct: 1,
    explanation: 'If d_model is the total dimension and h is the number of heads, each head operates in d_model/h dimensions.',
    difficulty: 'medium',
    concept: 'multi-head-attention'
  },
  {
    id: 'multihead-3',
    question: 'What happens to the outputs of multiple attention heads?',
    options: [
      'They are averaged together',
      'They are concatenated and linearly transformed',
      'They are multiplied together',
      'The maximum value is selected'
    ],
    correct: 1,
    explanation: 'The outputs from all heads are concatenated and then passed through a linear transformation (W^O) to produce the final output.',
    difficulty: 'hard',
    concept: 'multi-head-attention'
  },

  // Positional Encoding Questions
  {
    id: 'positional-1',
    question: 'Why do Transformers need positional encoding?',
    options: [
      'To reduce computational complexity',
      'To provide sequence order information',
      'To prevent overfitting',
      'To enable parallel training'
    ],
    correct: 1,
    explanation: 'Since attention is permutation-invariant, positional encodings are added to give the model information about the position of tokens in the sequence.',
    difficulty: 'easy',
    concept: 'positional-encoding'
  },
  {
    id: 'positional-2',
    question: 'What type of functions are used in sinusoidal positional encoding?',
    options: [
      'Linear functions',
      'Sinusoidal functions (sine and cosine)',
      'Exponential functions',
      'Polynomial functions'
    ],
    correct: 1,
    explanation: 'Sinusoidal positional encoding uses sine and cosine functions with different frequencies to encode position information.',
    difficulty: 'medium',
    concept: 'positional-encoding'
  },
  {
    id: 'positional-3',
    question: 'What is a key advantage of sinusoidal positional encoding?',
    options: [
      'It requires fewer parameters',
      'It can extrapolate to longer sequences',
      'It is computationally faster',
      'It provides better gradients'
    ],
    correct: 1,
    explanation: 'Sinusoidal encoding can handle sequences longer than those seen during training, making it useful for extrapolation.',
    difficulty: 'hard',
    concept: 'positional-encoding'
  },
  {
    id: 'positional-4',
    question: 'What is the main difference between learned and sinusoidal positional encodings?',
    options: [
      'Learned encodings are faster to compute',
      'Sinusoidal encodings require more memory',
      'Learned encodings cannot extrapolate beyond training length',
      'Sinusoidal encodings are not differentiable'
    ],
    correct: 2,
    explanation: 'Learned positional embeddings are trainable parameters that cannot handle sequences longer than the maximum length seen during training.',
    difficulty: 'medium',
    concept: 'positional-encoding'
  },

  // Architecture Questions
  {
    id: 'architecture-1',
    question: 'What is the typical structure of a Transformer layer?',
    options: [
      'Attention → Feed-forward → Normalization',
      'Normalization → Attention → Feed-forward → Normalization',
      'Feed-forward → Attention → Normalization',
      'Attention → Normalization → Feed-forward → Normalization'
    ],
    correct: 1,
    explanation: 'A typical Transformer layer uses pre-norm: LayerNorm → Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm.',
    difficulty: 'medium',
    concept: 'transformer-architecture'
  },
  {
    id: 'architecture-2',
    question: 'What is the purpose of residual connections in Transformers?',
    options: [
      'To reduce the number of parameters',
      'To enable gradient flow in deep networks',
      'To increase model capacity',
      'To speed up computation'
    ],
    correct: 1,
    explanation: 'Residual connections help gradients flow through deep networks by providing direct paths from later layers to earlier layers.',
    difficulty: 'medium',
    concept: 'transformer-architecture'
  },
  {
    id: 'architecture-3',
    question: 'What is the difference between pre-norm and post-norm in Transformers?',
    options: [
      'Pre-norm is more computationally expensive',
      'Post-norm provides better training stability',
      'Pre-norm normalizes before the sublayer, post-norm after',
      'Pre-norm is only used in small models'
    ],
    correct: 2,
    explanation: 'Pre-norm applies layer normalization before the attention/FFN sublayers, while post-norm applies it after, affecting training stability.',
    difficulty: 'hard',
    concept: 'transformer-architecture'
  },

  // Transformer Variants Questions
  {
    id: 'variants-1',
    question: 'What type of attention does BERT use?',
    options: [
      'Causal attention (can only see previous tokens)',
      'Bidirectional attention (can see all tokens)',
      'Sparse attention (only attends to nearby tokens)',
      'Local attention (fixed window size)'
    ],
    correct: 1,
    explanation: 'BERT uses bidirectional attention, allowing each token to attend to all other tokens in the sequence, making it suitable for understanding tasks.',
    difficulty: 'easy',
    concept: 'transformer-variants'
  },
  {
    id: 'variants-2',
    question: 'What type of attention does GPT use?',
    options: [
      'Bidirectional attention',
      'Causal attention with triangular mask',
      'Sparse attention',
      'Cross-attention only'
    ],
    correct: 1,
    explanation: 'GPT uses causal attention with a triangular mask, ensuring each token can only attend to previous tokens, making it suitable for generation.',
    difficulty: 'medium',
    concept: 'transformer-variants'
  },
  {
    id: 'variants-3',
    question: 'What is the main advantage of encoder-decoder models like T5?',
    options: [
      'They are faster to train',
      'They can handle different input and output modalities',
      'They require fewer parameters',
      'They have better gradient flow'
    ],
    correct: 1,
    explanation: 'Encoder-decoder models can handle tasks where input and output are different (translation, summarization) by using separate encoding and decoding phases.',
    difficulty: 'medium',
    concept: 'transformer-variants'
  },
  {
    id: 'variants-4',
    question: 'What is the key difference between BERT and GPT pre-training objectives?',
    options: [
      'BERT uses next token prediction, GPT uses masked language modeling',
      'BERT uses masked language modeling, GPT uses next token prediction',
      'Both use the same pre-training objective',
      'BERT uses translation, GPT uses classification'
    ],
    correct: 1,
    explanation: 'BERT is pre-trained using masked language modeling (predicting masked tokens), while GPT uses next token prediction (autoregressive generation).',
    difficulty: 'hard',
    concept: 'transformer-variants'
  },

  // Advanced Questions
  {
    id: 'advanced-1',
    question: 'What is the relationship between model size and performance in Transformers?',
    options: [
      'Performance scales linearly with model size',
      'Performance scales logarithmically with model size',
      'Performance scales with the square root of model size',
      'There is no consistent relationship'
    ],
    correct: 1,
    explanation: 'Empirical studies show that Transformer performance scales roughly logarithmically with model size, following scaling laws.',
    difficulty: 'hard',
    concept: 'transformer-architecture'
  },
  {
    id: 'advanced-2',
    question: 'What is the main computational bottleneck in large Transformer models?',
    options: [
      'Feed-forward network computation',
      'Attention matrix multiplication',
      'Layer normalization',
      'Embedding lookup'
    ],
    correct: 1,
    explanation: 'Attention matrix multiplication (O(n²)) becomes the main bottleneck in large models, leading to research on sparse attention and other optimizations.',
    difficulty: 'hard',
    concept: 'attention-mechanism'
  },
  {
    id: 'advanced-3',
    question: 'What is the purpose of the feed-forward network in Transformer layers?',
    options: [
      'To add non-linearity and increase model capacity',
      'To reduce the sequence length',
      'To normalize the attention outputs',
      'To compute attention weights'
    ],
    correct: 0,
    explanation: 'The feed-forward network adds non-linearity and increases the model\'s capacity to learn complex transformations of the attention outputs.',
    difficulty: 'medium',
    concept: 'transformer-architecture'
  }
]; 