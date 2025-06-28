import { LearningModule } from '../types/LearningModule';

export const llmsModule: LearningModule = {
  id: 'llms',
  title: 'Large Language Models',
  description: 'Master the architecture, training, and deployment of modern language models. From pre-training to fine-tuning, understand how LLMs work and how to use them effectively.',
  color: '#059669',
  icon: 'Brain',
  progress: 0,
  estimatedHours: 15,
  prerequisites: ['fundamentals', 'transformers'],
  difficulty: 'Advanced',
  concepts: [
    {
      id: 'llm-architecture',
      title: 'LLM Architecture & Scaling Laws',
      description: 'Deep dive into modern LLM architectures and scaling principles',
      slides: [
        {
          id: 'architecture-evolution',
          title: 'Evolution from BERT to GPT to Modern LLMs',
          content: {
            tier1: "The journey from BERT's bidirectional encoding to GPT's autoregressive generation represents a fundamental shift toward generative AI. Modern LLMs combine the best of both worlds.",
            tier2: "Key architectural milestones: BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder), and modern innovations like RoPE, SwiGLU, and grouped query attention.",
            tier3: "Modern architectures optimize for efficiency at scale: techniques like gradient checkpointing, mixed precision training, and model parallelism enable training trillion-parameter models."
          },
          mathNotations: [
            {
              id: 'autoregressive-objective',
              latex: 'P(x_1, ..., x_T) = \\prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})',
              explanation: 'Autoregressive language modeling objective used in decoder-only models',
              interactive: true
            }
          ],
          keyPoints: [
            "BERT: Masked Language Model (MLM) for understanding",
            "GPT: Causal Language Model (CLM) for generation", 
            "Modern LLMs: Unified architectures for both tasks",
            "Scale drives emergent capabilities"
          ]
        },
        {
          id: 'scaling-laws-deep',
          title: 'Scaling Laws: The Science of Model Size',
          content: {
            tier1: "Scaling laws predict how model performance improves with size, data, and compute. They guide fundamental decisions about resource allocation.",
            tier2: "Chinchilla findings: Models were undertrained on data. Optimal compute allocation requires more tokens than previously thought (20 tokens per parameter).",
            tier3: "Emergent abilities appear at specific scales: in-context learning (~1B params), chain-of-thought reasoning (~10B params), and advanced reasoning (~100B+ params)."
          },
          mathNotations: [
            {
              id: 'chinchilla-law',
              latex: 'N_{optimal} \\propto C^{0.5}, \\quad D_{optimal} \\propto C^{0.5}',
              explanation: 'Chinchilla scaling: optimal model size N and data size D both scale as square root of compute C',
              interactive: true
            },
            {
              id: 'loss-prediction',
              latex: 'L(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}',
              explanation: 'Power law for predicting loss given model size N and dataset size D',
              interactive: false
            }
          ],
          keyPoints: [
            "Compute-optimal training requires balanced scaling",
            "Emergent abilities appear at predictable scales",
            "Data quality matters more than quantity",
            "Inference efficiency vs training efficiency tradeoffs"
          ]
        },
        {
          id: 'modern-innovations',
          title: 'Modern LLM Architectural Innovations',
          content: {
            tier1: "Recent innovations focus on efficiency, stability, and capability. Key areas include attention mechanisms, positional encodings, and activation functions.",
            tier2: "Rotary Position Embedding (RoPE), Grouped Query Attention (GQA), and SwiGLU activations are now standard. These improve efficiency while maintaining or improving performance.",
            tier3: "Architecture search and neural scaling laws guide design choices. Models like LLaMA, PaLM, and GPT-4 represent different optimization points in the efficiency-capability space."
          },
          mathNotations: [
            {
              id: 'rope-encoding',
              latex: 'f_q(x_m, m) = (W_q x_m) \\otimes e^{im\\theta}',
              explanation: 'RoPE encoding applies rotation based on position, enabling length extrapolation',
              interactive: true
            }
          ],
          keyPoints: [
            "RoPE: Better positional encoding for long sequences",
            "GQA: Reduces KV cache memory in inference",
            "SwiGLU: More efficient activation function",
            "LayerNorm placement affects training stability"
          ]
        }
      ]
    },
    {
      id: 'training-methods',
      title: 'Training Methodologies & Optimization',
      description: 'Comprehensive training techniques from pre-training to alignment',
      slides: [
        {
          id: 'pretraining-overview',
          title: 'Pre-training: Learning from the Internet',
          content: {
            tier1: "Pre-training on vast text corpora teaches models language understanding and world knowledge. The data quality and diversity are crucial for model capabilities.",
            tier2: "Modern pre-training uses sophisticated data processing: deduplication, filtering, and careful dataset mixing. Training dynamics require careful learning rate scheduling and gradient clipping.",
            tier3: "Compute-optimal training balances model size, data size, and training time. Recent trends emphasize data quality over quantity and efficient training techniques."
          },
          mathNotations: [
            {
              id: 'gradient-clipping',
              latex: '\\tilde{g} = \\min\\left(1, \\frac{\\tau}{||g||_2}\\right) \\cdot g',
              explanation: 'Gradient clipping prevents instability during training by capping gradient norms',
              interactive: false
            }
          ],
          keyPoints: [
            "Data preprocessing is critical for model quality",
            "Learning rate schedules prevent training instability",
            "Compute-optimal training maximizes performance per FLOPs",
            "Curriculum learning can improve training efficiency"
          ]
        },
        {
          id: 'finetuning-techniques',
          title: 'Fine-tuning: Specializing Pre-trained Models',
          content: {
            tier1: "Fine-tuning adapts pre-trained models to specific tasks. Methods range from full fine-tuning to parameter-efficient approaches like LoRA.",
            tier2: "Parameter-efficient fine-tuning (PEFT) methods modify only a small subset of parameters. LoRA, adapters, and prompt tuning achieve strong performance with minimal parameter updates.",
            tier3: "Advanced techniques include instruction tuning, few-shot prompting, and in-context learning. The choice depends on data availability, computational constraints, and target performance."
          },
          mathNotations: [
            {
              id: 'lora-update',
              latex: 'W = W_0 + \\Delta W = W_0 + BA',
              explanation: 'LoRA represents weight updates as low-rank decomposition, reducing trainable parameters',
              interactive: true
            }
          ],
          keyPoints: [
            "LoRA: Low-rank adaptation for efficient fine-tuning",
            "Instruction tuning improves following complex commands",
            "Few-shot learning leverages pre-trained knowledge",
            "Task-specific architectures may outperform general models"
          ]
        },
        {
          id: 'alignment-rlhf',
          title: 'Alignment & RLHF: Making Models Helpful and Safe',
          content: {
            tier1: "Alignment ensures models behave helpfully, harmlessly, and honestly. RLHF uses human feedback to train reward models that guide model behavior.",
            tier2: "The RLHF process: collect human preferences, train reward model, optimize policy with PPO. Constitutional AI provides an alternative approach using AI feedback.",
            tier3: "Advanced alignment techniques include recursive reward modeling, debate, and amplification. The goal is scalable oversight for superintelligent systems."
          },
          mathNotations: [
            {
              id: 'rlhf-objective',
              latex: 'J(\\pi) = \\mathbb{E}[r(x, y)] - \\beta \\cdot D_{KL}(\\pi(y|x) || \\pi_{ref}(y|x))',
              explanation: 'RLHF objective balances reward maximization with KL penalty from reference model',
              interactive: true
            }
          ],
          keyPoints: [
            "Human feedback improves model alignment",
            "Reward models must generalize beyond training data",
            "KL regularization prevents reward hacking",
            "Constitutional AI scales oversight with AI feedback"
          ]
        }
      ]
    },
    {
      id: 'capabilities-limitations',
      title: 'Capabilities, Limitations & Future Directions',
      description: 'Understanding what LLMs can and cannot do, and where the field is heading',
      slides: [
        {
          id: 'emergent-capabilities',
          title: 'Emergent Capabilities at Scale',
          content: {
            tier1: "Emergent capabilities appear suddenly at certain scales: in-context learning, chain-of-thought reasoning, and instruction following weren't explicitly trained but emerge from scale.",
            tier2: "Key emergent abilities: few-shot learning (~1B params), complex reasoning (~10B), coding (~20B), advanced reasoning (~100B+). These thresholds are approximate and task-dependent.",
            tier3: "The mechanism behind emergence is debated: is it smooth improvement that appears sudden due to evaluation metrics, or genuine phase transitions in model behavior?"
          },
          keyPoints: [
            "In-context learning: Learning from examples in the prompt",
            "Chain-of-thought: Step-by-step reasoning improves accuracy",
            "Instruction following: Understanding and executing commands",
            "Few-shot generalization: Adapting to new tasks quickly"
          ]
        },
        {
          id: 'current-limitations',
          title: 'Fundamental Limitations of Current LLMs',
          content: {
            tier1: "LLMs have significant limitations: hallucination, lack of grounding, poor mathematical reasoning, and inability to learn from experience during inference.",
            tier2: "Key challenges: knowledge cutoffs, factual inconsistency, difficulty with multi-step reasoning, and lack of true understanding vs pattern matching.",
            tier3: "Research directions: retrieval-augmented generation, tool use, multi-modal learning, and improved training objectives aim to address these limitations."
          },
          keyPoints: [
            "Hallucination: Generating false but plausible information",
            "Knowledge cutoff: Limited to training data timeframe", 
            "Reasoning errors: Especially in math and logic",
            "Context limitations: Fixed context window constraints"
          ]
        },
        {
          id: 'future-directions',
          title: 'Future Directions & Next-Generation Models',
          content: {
            tier1: "Future LLMs will likely be multi-modal, continuously learning, and capable of using tools. Integration with robotics and scientific discovery are key applications.",
            tier2: "Technical directions: sparse models (MoE), longer context windows, better reasoning architectures, and integration with symbolic AI systems.",
            tier3: "Potential breakthroughs: AGI-level reasoning, scientific discovery acceleration, and seamless human-AI collaboration across all domains."
          },
          keyPoints: [
            "Multi-modal: Vision, audio, and text integration",
            "Tool use: API calls, code execution, web browsing",
            "Continuous learning: Updating knowledge post-training",
            "Reasoning architectures: Beyond next-token prediction"
          ]
        }
      ]
    }
  ]
};
