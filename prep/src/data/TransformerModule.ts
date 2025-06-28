import { LearningModule } from '../types/LearningModule';

export const transformerModule: LearningModule = {
  id: 'transformers',
  title: 'Transformer Architecture',
  description: 'Master the revolutionary architecture that powers modern AI. From attention mechanisms to positional encoding, understand every component through interactive visualizations and hands-on practice.',
  color: '#7c3aed',
  icon: 'Brain',
  progress: 0,
  estimatedHours: 10,
  prerequisites: ['fundamentals'],
  difficulty: 'Intermediate',
  concepts: [
    {
      id: 'attention-mechanism',
      title: 'Self-Attention Mechanism',
      description: 'The revolutionary mechanism that enables parallel processing and captures long-range dependencies',
      slides: [
        {
          id: 'attention-intuition',
          title: 'The Intuition Behind Attention',
          content: {
            tier1: "Attention is like having a smart spotlight that can focus on different parts of a sentence to understand context. Instead of reading word-by-word like humans, it can look at all words at once and decide which ones are most important for understanding each word.",
            tier2: "Think of it as a soft database lookup: each word (query) searches through all other words (keys) to find the most relevant information (values). The attention weights tell us how much each word should 'pay attention' to every other word.",
            tier3: "Mathematically, this creates a weighted combination where each token's representation becomes a context-aware mixture of all other tokens, with weights determined by learned compatibility functions."
          },
          mathNotations: [
            {
              id: 'attention-formula',
              latex: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
              explanation: 'The fundamental attention formula where Q=queries, K=keys, V=values, and √d_k prevents gradient vanishing',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'attention-demo',
              type: 'interactive',
              component: 'AttentionVisualization',
              data: { 
                sentence: "The cat sat on the mat",
                focusWord: "cat",
                showWeights: true
              },
              controls: [
                {
                  id: 'focus-word',
                  type: 'dropdown',
                  label: 'Focus Word',
                  options: ['The', 'cat', 'sat', 'on', 'the', 'mat'],
                  defaultValue: 'cat'
                },
                {
                  id: 'show-weights',
                  type: 'toggle',
                  label: 'Show Attention Weights',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'attention-components',
          title: 'Queries, Keys, and Values',
          content: {
            tier1: "Every word in the sequence gets transformed into three different representations: Query (Q), Key (K), and Value (V). These serve different purposes in the attention mechanism.",
            tier2: "Query (Q): 'What am I looking for?' - represents the current word's needs. Key (K): 'What information is available?' - represents what each word can offer. Value (V): 'What is the actual content?' - the information to be retrieved.",
            tier3: "The Q, K, V matrices are learned linear transformations of the input embeddings. Each has its own weight matrix (W_Q, W_K, W_V) that gets optimized during training to capture different types of relationships."
          },
          mathNotations: [
            {
              id: 'qkv-transform',
              latex: 'Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V',
              explanation: 'Linear transformations that project input embeddings into query, key, and value spaces',
              interactive: false
            }
          ],
          visualizations: [
            {
              id: 'qkv-transformation',
              type: 'interactive',
              component: 'QKVTransformation',
              data: { 
                inputTokens: ['The', 'cat', 'sat'],
                embeddingDim: 4,
                showMatrices: true
              },
              controls: [
                {
                  id: 'show-matrices',
                  type: 'toggle',
                  label: 'Show Weight Matrices',
                  defaultValue: true
                },
                {
                  id: 'embedding-dim',
                  type: 'slider',
                  label: 'Embedding Dimension',
                  range: [2, 8],
                  defaultValue: 4
                }
              ]
            }
          ]
        },
        {
          id: 'attention-computation',
          title: 'Step-by-Step Attention Computation',
          content: {
            tier1: "Attention computation happens in 5 key steps: 1) Compute attention scores, 2) Scale by √d_k, 3) Apply softmax, 4) Weight the values, 5) Sum up the weighted values.",
            tier2: "The attention scores (QK^T) measure how compatible each query is with each key. Higher scores mean stronger attention. The √d_k scaling prevents the dot products from becoming too large, which would cause softmax saturation.",
            tier3: "The softmax converts scores to probabilities that sum to 1, ensuring the attention weights are properly normalized. The final weighted sum creates a context vector that represents the most relevant information for each position."
          },
          mathNotations: [
            {
              id: 'attention-steps',
              latex: '\\text{scores} = QK^T, \\quad \\text{weights} = \\text{softmax}(\\frac{\\text{scores}}{\\sqrt{d_k}}), \\quad \\text{output} = \\text{weights} \\cdot V',
              explanation: 'The three-step computation: compute scores, normalize with softmax, weight the values',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'attention-steps-demo',
              type: 'interactive',
              component: 'AttentionSteps',
              data: { 
                sequence: ['I', 'love', 'machine', 'learning'],
                step: 1
              },
              controls: [
                {
                  id: 'step',
                  type: 'slider',
                  label: 'Computation Step',
                  range: [1, 5],
                  defaultValue: 1
                },
                {
                  id: 'auto-play',
                  type: 'toggle',
                  label: 'Auto-play Steps',
                  defaultValue: false
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'multi-head-attention',
      title: 'Multi-Head Attention',
      description: 'Multiple attention mechanisms running in parallel to capture different types of relationships',
      slides: [
        {
          id: 'multi-head-intuition',
          title: 'Why Multiple Heads?',
          content: {
            tier1: "A single attention head might focus on one type of relationship (like subject-verb agreement). Multiple heads can simultaneously capture different types of relationships: syntax, semantics, position, and more.",
            tier2: "Think of it like having multiple specialists working on the same problem. One head might focus on grammatical relationships, another on semantic meaning, another on positional patterns, and so on.",
            tier3: "Each head operates in a lower-dimensional space (d_model/h) but can specialize in different aspects. The final output combines all these specialized views through a learned linear transformation."
          },
          mathNotations: [
            {
              id: 'multi-head-formula',
              latex: '\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O',
              explanation: 'Multi-head attention concatenates outputs from multiple attention heads and applies a final linear transformation',
              interactive: true
            },
            {
              id: 'head-formula',
              latex: '\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)',
              explanation: 'Each head has its own learned weight matrices for Q, K, V transformations',
              interactive: false
            }
          ],
          visualizations: [
            {
              id: 'multi-head-demo',
              type: 'interactive',
              component: 'MultiHeadAttention',
              data: { 
                sentence: "The quick brown fox jumps over the lazy dog",
                numHeads: 8,
                showSpecializations: true
              },
              controls: [
                {
                  id: 'num-heads',
                  type: 'slider',
                  label: 'Number of Heads',
                  range: [1, 12],
                  defaultValue: 8
                },
                {
                  id: 'show-specializations',
                  type: 'toggle',
                  label: 'Show Head Specializations',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'head-specializations',
          title: 'What Different Heads Learn',
          content: {
            tier1: "Different attention heads naturally specialize in different types of relationships. Some focus on syntactic patterns, others on semantic relationships, positional dependencies, or long-range connections.",
            tier2: "Research shows that heads often develop interpretable specializations: some attend to adjacent words (local syntax), others to subject-verb relationships, some to semantic similarity, and others to long-distance dependencies.",
            tier3: "This specialization emerges during training without explicit supervision. The model learns to distribute different types of attention patterns across heads to maximize the overall performance on the task."
          },
          visualizations: [
            {
              id: 'head-patterns',
              type: 'interactive',
              component: 'HeadPatterns',
              data: { 
                patterns: ['syntactic', 'semantic', 'positional', 'long-range'],
                sentence: "The scientist discovered a new particle in the laboratory"
              },
              controls: [
                {
                  id: 'pattern-type',
                  type: 'dropdown',
                  label: 'Attention Pattern Type',
                  options: ['Syntactic', 'Semantic', 'Positional', 'Long-range'],
                  defaultValue: 'Syntactic'
                },
                {
                  id: 'show-heatmap',
                  type: 'toggle',
                  label: 'Show Attention Heatmap',
                  defaultValue: true
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'positional-encoding',
      title: 'Positional Encoding',
      description: 'Injecting sequence order information into the permutation-invariant attention mechanism',
      slides: [
        {
          id: 'positional-problem',
          title: 'The Positional Problem',
          content: {
            tier1: "Attention is permutation-invariant - it treats words the same regardless of their position. But word order matters in language! 'The cat sat' means something different than 'Sat the cat'.",
            tier2: "Without positional information, the model can't distinguish between different word orders. We need to inject position information so the model knows where each word appears in the sequence.",
            tier3: "The challenge is to create positional encodings that can generalize to sequences of different lengths and that allow the model to learn relative positions effectively."
          },
          visualizations: [
            {
              id: 'permutation-demo',
              type: 'interactive',
              component: 'PermutationDemo',
              data: { 
                original: "The cat sat on the mat",
                permuted: "Sat the cat the on mat"
              },
              controls: [
                {
                  id: 'show-attention',
                  type: 'toggle',
                  label: 'Show Attention Patterns',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'sinusoidal-encoding',
          title: 'Sinusoidal Positional Encoding',
          content: {
            tier1: "The original Transformer uses sinusoidal functions to encode position information. Each position gets a unique encoding based on sine and cosine functions of different frequencies.",
            tier2: "Sinusoidal encoding has special properties: it's deterministic (same position always gets same encoding), it can handle sequences longer than training, and relative positions can be computed as linear combinations.",
            tier3: "The encoding uses different frequencies for different dimensions, allowing the model to learn both absolute and relative positional relationships. Lower frequencies capture long-range dependencies, higher frequencies capture local patterns."
          },
          mathNotations: [
            {
              id: 'sinusoidal-formula',
              latex: 'PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)',
              explanation: 'Sinusoidal encoding for even dimensions',
              interactive: true
            },
            {
              id: 'cosine-formula',
              latex: 'PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)',
              explanation: 'Sinusoidal encoding for odd dimensions',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'sinusoidal-demo',
              type: 'interactive',
              component: 'SinusoidalEncoding',
              data: { 
                sequenceLength: 20,
                embeddingDim: 8,
                showFrequencies: true
              },
              controls: [
                {
                  id: 'sequence-length',
                  type: 'slider',
                  label: 'Sequence Length',
                  range: [5, 50],
                  defaultValue: 20
                },
                {
                  id: 'embedding-dim',
                  type: 'slider',
                  label: 'Embedding Dimension',
                  range: [4, 16],
                  defaultValue: 8
                },
                {
                  id: 'show-frequencies',
                  type: 'toggle',
                  label: 'Show Frequency Patterns',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'modern-encodings',
          title: 'Modern Positional Encodings',
          content: {
            tier1: "While sinusoidal encoding works well, researchers have developed newer approaches that offer better performance and flexibility: learned positional embeddings, rotary positional encoding (RoPE), and attention with linear biases (ALiBi).",
            tier2: "Learned embeddings are trainable parameters for each position, offering better performance on fixed-length sequences but no extrapolation. RoPE encodes positions as rotations in complex space, providing excellent length extrapolation.",
            tier3: "ALiBi adds position-dependent biases directly to attention scores, making it very efficient and providing strong extrapolation properties. Each approach has trade-offs between performance, computational efficiency, and generalization."
          },
          visualizations: [
            {
              id: 'encoding-comparison',
              type: 'interactive',
              component: 'EncodingComparison',
              data: { 
                encodings: ['sinusoidal', 'learned', 'rope', 'alibi'],
                sequenceLength: 30
              },
              controls: [
                {
                  id: 'encoding-type',
                  type: 'dropdown',
                  label: 'Encoding Type',
                  options: ['Sinusoidal', 'Learned', 'RoPE', 'ALiBi'],
                  defaultValue: 'Sinusoidal'
                },
                {
                  id: 'extrapolation-test',
                  type: 'toggle',
                  label: 'Test Length Extrapolation',
                  defaultValue: false
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'transformer-architecture',
      title: 'Complete Transformer Architecture',
      description: 'Putting it all together: the full encoder-decoder architecture with all components',
      slides: [
        {
          id: 'architecture-overview',
          title: 'The Complete Architecture',
          content: {
            tier1: "A Transformer consists of an encoder and decoder, each made up of multiple identical layers. Each layer contains multi-head attention, feed-forward networks, layer normalization, and residual connections.",
            tier2: "The encoder processes the input sequence bidirectionally, creating context-aware representations. The decoder generates the output sequence autoregressively, using both self-attention and cross-attention to the encoder.",
            tier3: "The architecture is highly modular and parallelizable. Each component serves a specific purpose: attention captures relationships, FFNs add non-linearity, normalization stabilizes training, and residuals enable gradient flow."
          },
          visualizations: [
            {
              id: 'architecture-diagram',
              type: 'interactive',
              component: 'TransformerArchitecture',
              data: { 
                numLayers: 6,
                showDataFlow: true,
                highlightComponent: 'attention'
              },
              controls: [
                {
                  id: 'num-layers',
                  type: 'slider',
                  label: 'Number of Layers',
                  range: [1, 12],
                  defaultValue: 6
                },
                {
                  id: 'highlight-component',
                  type: 'dropdown',
                  label: 'Highlight Component',
                  options: ['Attention', 'FFN', 'Norm', 'Residual'],
                  defaultValue: 'Attention'
                },
                {
                  id: 'show-data-flow',
                  type: 'toggle',
                  label: 'Show Data Flow',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'layer-components',
          title: 'Key Architectural Components',
          content: {
            tier1: "Each Transformer layer contains several key components: multi-head attention, feed-forward networks, layer normalization, and residual connections. Each serves a crucial role in the model's success.",
            tier2: "Layer normalization stabilizes training by normalizing activations within each layer. Residual connections enable gradient flow through deep networks. Feed-forward networks add non-linearity and increase model capacity.",
            tier3: "The order of operations matters: pre-norm (normalize before attention/FFN) vs post-norm (normalize after) affects training stability. Modern architectures often use pre-norm for better convergence."
          },
          mathNotations: [
            {
              id: 'layer-norm',
              latex: '\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta',
              explanation: 'Layer normalization with learnable parameters γ and β',
              interactive: true
            },
            {
              id: 'residual',
              latex: '\\text{output} = x + \\text{Sublayer}(x)',
              explanation: 'Residual connection that enables gradient flow',
              interactive: false
            }
          ],
          visualizations: [
            {
              id: 'component-demo',
              type: 'interactive',
              component: 'LayerComponents',
              data: { 
                components: ['attention', 'ffn', 'norm', 'residual'],
                showGradients: true
              },
              controls: [
                {
                  id: 'component',
                  type: 'dropdown',
                  label: 'Component',
                  options: ['Attention', 'FFN', 'Layer Norm', 'Residual'],
                  defaultValue: 'Attention'
                },
                {
                  id: 'show-gradients',
                  type: 'toggle',
                  label: 'Show Gradient Flow',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'training-inference',
          title: 'Training vs Inference',
          content: {
            tier1: "During training, the model sees the entire input sequence and learns to predict the next token. During inference, it generates tokens one by one, using previously generated tokens as context.",
            tier2: "Training uses teacher forcing where the model sees the correct previous tokens. Inference uses autoregressive generation where each prediction depends on the model's own previous outputs.",
            tier3: "This creates a discrepancy between training and inference (exposure bias). Techniques like scheduled sampling and reinforcement learning help bridge this gap."
          },
          visualizations: [
            {
              id: 'training-inference-demo',
              type: 'interactive',
              component: 'TrainingInference',
              data: { 
                mode: 'training',
                sequence: "The quick brown fox",
                showPredictions: true
              },
              controls: [
                {
                  id: 'mode',
                  type: 'dropdown',
                  label: 'Mode',
                  options: ['Training', 'Inference'],
                  defaultValue: 'Training'
                },
                {
                  id: 'show-predictions',
                  type: 'toggle',
                  label: 'Show Predictions',
                  defaultValue: true
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'transformer-variants',
      title: 'Transformer Variants',
      description: 'Understanding different transformer architectures: BERT, GPT, T5, and their use cases',
      slides: [
        {
          id: 'encoder-only',
          title: 'Encoder-Only Models (BERT-style)',
          content: {
            tier1: "Encoder-only models like BERT use bidirectional attention, meaning each token can attend to all other tokens in the sequence. This makes them excellent for understanding tasks where you have complete context.",
            tier2: "BERT is pre-trained using masked language modeling (predicting masked tokens) and next sentence prediction. This creates rich contextual representations that can be fine-tuned for various downstream tasks.",
            tier3: "The bidirectional nature makes BERT great for classification, named entity recognition, question answering, and other tasks where understanding the full context is crucial."
          },
          visualizations: [
            {
              id: 'bert-demo',
              type: 'interactive',
              component: 'BERTDemo',
              data: { 
                task: 'classification',
                sentence: "The movie was [MASK] and entertaining",
                showAttention: true
              },
              controls: [
                {
                  id: 'task',
                  type: 'dropdown',
                  label: 'Task',
                  options: ['Classification', 'NER', 'QA', 'MLM'],
                  defaultValue: 'Classification'
                },
                {
                  id: 'show-attention',
                  type: 'toggle',
                  label: 'Show Bidirectional Attention',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'decoder-only',
          title: 'Decoder-Only Models (GPT-style)',
          content: {
            tier1: "Decoder-only models like GPT use causal attention, meaning each token can only attend to previous tokens. This makes them excellent for text generation and completion tasks.",
            tier2: "GPT is pre-trained using next token prediction, learning to predict the most likely next word given the previous context. This creates a strong language model that can generate coherent text.",
            tier3: "The autoregressive nature makes GPT great for text generation, completion, dialogue, and other tasks where you need to generate text sequentially."
          },
          visualizations: [
            {
              id: 'gpt-demo',
              type: 'interactive',
              component: 'GPTDemo',
              data: { 
                prompt: "The future of artificial intelligence",
                maxTokens: 20,
                showCausalMask: true
              },
              controls: [
                {
                  id: 'prompt',
                  type: 'button',
                  label: 'Prompt',
                  defaultValue: 'The future of artificial intelligence'
                },
                {
                  id: 'max-tokens',
                  type: 'slider',
                  label: 'Max Tokens',
                  range: [5, 50],
                  defaultValue: 20
                },
                {
                  id: 'show-causal-mask',
                  type: 'toggle',
                  label: 'Show Causal Mask',
                  defaultValue: true
                }
              ]
            }
          ]
        },
        {
          id: 'encoder-decoder',
          title: 'Encoder-Decoder Models (T5-style)',
          content: {
            tier1: "Encoder-decoder models like T5 have separate encoder and decoder components. The encoder processes the input, and the decoder generates the output while attending to the encoder's representations.",
            tier2: "T5 is pre-trained using text-to-text transfer, converting all tasks to text generation. This unified approach allows the same model to handle translation, summarization, question answering, and more.",
            tier3: "The separation of encoding and decoding makes these models ideal for tasks where input and output are different (translation, summarization) or where you need structured generation."
          },
          visualizations: [
            {
              id: 't5-demo',
              type: 'interactive',
              component: 'T5Demo',
              data: { 
                task: 'translation',
                input: "Hello world",
                target: "Hola mundo",
                showCrossAttention: true
              },
              controls: [
                {
                  id: 'task',
                  type: 'dropdown',
                  label: 'Task',
                  options: ['Translation', 'Summarization', 'QA', 'Classification'],
                  defaultValue: 'Translation'
                },
                {
                  id: 'show-cross-attention',
                  type: 'toggle',
                  label: 'Show Cross-Attention',
                  defaultValue: true
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}; 