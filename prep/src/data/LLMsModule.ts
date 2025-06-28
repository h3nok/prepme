import { LearningModule } from '../types/LearningModule';

export const llmsModule: LearningModule = {
  id: 'llms',
  title: 'Large Language Models',
  description: 'Master the architecture, training, and deployment of modern language models. From pre-training to fine-tuning, understand how LLMs work and how to use them effectively.',
  color: '#059669',
  icon: 'Brain',
  progress: 0,
  estimatedHours: 12,
  prerequisites: ['fundamentals', 'transformers'],
  difficulty: 'Advanced',
  concepts: [
    {
      id: 'llm-architecture',
      title: 'LLM Architecture Fundamentals',
      description: 'Understanding the core architecture decisions that make modern language models work',
      slides: [
        {
          id: 'architecture-overview',
          title: 'Modern LLM Architecture',
          content: {
            tier1: "Modern Large Language Models are built on Transformer architecture but with key innovations: massive scale, efficient training techniques, and sophisticated pre-training objectives.",
            tier2: "The architecture follows a decoder-only pattern (like GPT) with causal attention, allowing for autoregressive text generation. Key innovations include improved positional encodings, better normalization strategies, and optimized attention mechanisms.",
            tier3: "Architectural decisions are driven by scaling laws: performance scales predictably with model size, data size, and compute. This has led to models with hundreds of billions of parameters trained on trillions of tokens."
          },
          mathNotations: [
            {
              id: 'scaling-law',
              latex: 'L(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}',
              explanation: 'Chinchilla scaling law where L is loss, N is model size, D is data size, and α, β are scaling exponents',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'llm-architecture-demo',
              type: 'interactive',
              component: 'LLMArchitecture',
              data: { 
                modelSize: '175B',
                numLayers: 96,
                attentionHeads: 96,
                showScaling: true
              },
              controls: [
                {
                  id: 'model-size',
                  type: 'dropdown',
                  label: 'Model Size',
                  options: ['1.3B', '6.7B', '13B', '30B', '65B', '175B'],
                  defaultValue: '175B'
                },
                {
                  id: 'show-scaling',
                  type: 'toggle',
                  label: 'Show Scaling Laws',
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
