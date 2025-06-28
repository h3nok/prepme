import { QuizQuestion } from '../types/LearningModule';

export const llmsQuiz: QuizQuestion[] = [
  {
    id: 'llm-1',
    question: 'What is the primary architectural pattern used in modern Large Language Models like GPT?',
    options: [
      'Encoder-only (like BERT)',
      'Decoder-only (like GPT)',
      'Encoder-decoder (like T5)',
      'Recurrent Neural Network'
    ],
    correctAnswer: 1,
    explanation: 'Modern LLMs like GPT use decoder-only architecture with causal attention, meaning each token can only attend to previous tokens. This enables autoregressive text generation.',
    difficulty: 'Medium',
    concept: 'llm-architecture'
  },
  {
    id: 'llm-2',
    question: 'According to the Chinchilla scaling law, how does optimal model size scale with data size?',
    options: [
      'N_opt ∝ D^0.5',
      'N_opt ∝ D^0.74',
      'N_opt ∝ D^1.0',
      'N_opt ∝ D^1.5'
    ],
    correctAnswer: 1,
    explanation: 'The Chinchilla scaling law shows that optimal model size scales as N_opt ∝ D^0.74 where N is parameters and D is training tokens. This has led to more efficient training strategies.',
    difficulty: 'Hard',
    concept: 'llm-architecture'
  },
  {
    id: 'llm-3',
    question: 'What is the core pre-training objective for language models?',
    options: [
      'Masked Language Modeling',
      'Next Token Prediction',
      'Sequence-to-Sequence Translation',
      'Question Answering'
    ],
    correctAnswer: 1,
    explanation: 'Next token prediction is the core objective: given a sequence of tokens, predict the most likely next token. This simple objective teaches the model to understand language patterns.',
    difficulty: 'Easy',
    concept: 'pre-training'
  },
  {
    id: 'llm-4',
    question: 'What is the mathematical formulation of the next token prediction loss?',
    options: [
      'L = -Σ log P(x_i | x_{<i})',
      'L = -Σ log P(x_i | x_{>i})',
      'L = Σ P(x_i | x_{<i})',
      'L = -Σ P(x_i | x_{<i})'
    ],
    correctAnswer: 0,
    explanation: 'The cross-entropy loss for next token prediction is L = -Σ log P(x_i | x_{<i}) where the model learns to minimize this loss across training examples.',
    difficulty: 'Medium',
    concept: 'pre-training'
  },
  {
    id: 'llm-5',
    question: 'Which technique allows fine-tuning large models on limited hardware by updating only a small fraction of parameters?',
    options: [
      'Full Fine-tuning',
      'LoRA (Low-Rank Adaptation)',
      'Prompt Tuning',
      'Prefix Tuning'
    ],
    correctAnswer: 1,
    explanation: 'LoRA adds low-rank adaptation matrices to attention layers, updating only a small fraction of parameters. This reduces memory requirements and enables fine-tuning on consumer hardware.',
    difficulty: 'Medium',
    concept: 'fine-tuning'
  },
  {
    id: 'llm-6',
    question: 'What is the mathematical formulation of LoRA parameterization?',
    options: [
      'W = W_0 + α · BA',
      'W = W_0 + α · (B + A)',
      'W = W_0 · BA',
      'W = W_0 + α · (B ⊗ A)'
    ],
    correctAnswer: 0,
    explanation: 'LoRA parameterization is W = W_0 + α · BA where B and A are low-rank matrices and α is a scaling factor.',
    difficulty: 'Hard',
    concept: 'fine-tuning'
  },
  {
    id: 'llm-7',
    question: 'What does RLHF stand for and what is its primary purpose?',
    options: [
      'Reinforcement Learning from Human Feedback - to improve model performance on specific tasks',
      'Reinforcement Learning from Human Feedback - to align models with human preferences',
      'Recurrent Learning from Human Feedback - to improve memory retention',
      'Random Learning from Human Feedback - to increase model diversity'
    ],
    correctAnswer: 1,
    explanation: 'RLHF (Reinforcement Learning from Human Feedback) uses human feedback to align models with human preferences, making them helpful, harmless, and honest.',
    difficulty: 'Medium',
    concept: 'fine-tuning'
  },
  {
    id: 'llm-8',
    question: 'What is the PPO objective used in RLHF?',
    options: [
      'L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]',
      'L^CLIP(θ) = E[r_t(θ)A_t]',
      'L^CLIP(θ) = E[clip(r_t(θ), 1-ε, 1+ε)]',
      'L^CLIP(θ) = E[A_t]'
    ],
    correctAnswer: 0,
    explanation: 'The PPO clipped objective is L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] which prevents large policy updates.',
    difficulty: 'Hard',
    concept: 'fine-tuning'
  },
  {
    id: 'llm-9',
    question: 'What is in-context learning?',
    options: [
      'Learning by updating model parameters during inference',
      'Learning new tasks from examples provided in the prompt without parameter updates',
      'Learning through reinforcement learning',
      'Learning by fine-tuning on new data'
    ],
    correctAnswer: 1,
    explanation: 'In-context learning allows models to learn new tasks from examples provided in the prompt, without any parameter updates. The model learns patterns from the examples and applies them to new inputs.',
    difficulty: 'Medium',
    concept: 'prompt-engineering'
  },
  {
    id: 'llm-10',
    question: 'What is the purpose of chain-of-thought prompting?',
    options: [
      'To make the model generate longer responses',
      'To encourage the model to show its reasoning process step-by-step',
      'To improve the model\'s vocabulary',
      'To reduce response time'
    ],
    correctAnswer: 1,
    explanation: 'Chain-of-thought prompting encourages models to show their reasoning process step-by-step, improving performance on complex reasoning tasks by breaking down problems into intermediate steps.',
    difficulty: 'Medium',
    concept: 'prompt-engineering'
  },
  {
    id: 'llm-11',
    question: 'Which benchmark is commonly used to evaluate language model knowledge across multiple domains?',
    options: [
      'GLUE',
      'MMLU (Massive Multitask Language Understanding)',
      'SuperGLUE',
      'SQuAD'
    ],
    correctAnswer: 1,
    explanation: 'MMLU (Massive Multitask Language Understanding) evaluates language models across 57 tasks in various domains including STEM, humanities, social sciences, and more.',
    difficulty: 'Easy',
    concept: 'evaluation-metrics'
  },
  {
    id: 'llm-12',
    question: 'What is the primary challenge in evaluating LLM safety?',
    options: [
      'Defining what constitutes harm',
      'Computational cost',
      'Data availability',
      'Model size limitations'
    ],
    correctAnswer: 0,
    explanation: 'The primary challenge in evaluating LLM safety is defining what constitutes harm, as this varies across cultures and contexts, making it difficult to create universal safety metrics.',
    difficulty: 'Medium',
    concept: 'evaluation-metrics'
  },
  {
    id: 'llm-13',
    question: 'What is the relationship between model size and sample efficiency according to scaling laws?',
    options: [
      'Larger models are less sample-efficient',
      'Larger models are more sample-efficient',
      'Model size has no effect on sample efficiency',
      'Sample efficiency decreases with model size'
    ],
    correctAnswer: 1,
    explanation: 'According to scaling laws, larger models are more sample-efficient, meaning they can achieve the same performance with fewer training examples.',
    difficulty: 'Medium',
    concept: 'llm-architecture'
  },
  {
    id: 'llm-14',
    question: 'What is the purpose of causal attention in decoder-only models?',
    options: [
      'To allow bidirectional understanding',
      'To ensure each token can only attend to previous tokens',
      'To reduce computational complexity',
      'To improve memory efficiency'
    ],
    correctAnswer: 1,
    explanation: 'Causal attention ensures each token can only attend to previous tokens, creating a triangular attention pattern that enables autoregressive generation.',
    difficulty: 'Medium',
    concept: 'llm-architecture'
  },
  {
    id: 'llm-15',
    question: 'What is the main advantage of parameter-efficient fine-tuning methods like LoRA?',
    options: [
      'They improve model performance',
      'They reduce memory requirements and enable fine-tuning on limited hardware',
      'They speed up training',
      'They increase model accuracy'
    ],
    correctAnswer: 1,
    explanation: 'Parameter-efficient fine-tuning methods like LoRA reduce memory requirements and enable fine-tuning of large models on consumer hardware by updating only a small fraction of parameters.',
    difficulty: 'Easy',
    concept: 'fine-tuning'
  }
]; 