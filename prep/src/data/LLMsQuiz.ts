import { Quiz } from '../types/LearningModule';

export const llmsQuiz: Quiz = {
  id: 'llms-quiz',
  title: 'Large Language Models Quiz',
  description: 'Test your understanding of LLM architecture, training, and deployment',
  moduleId: 'llms',
  timeLimit: 35,
  passingScore: 80,
  questions: [
    {
      id: 'llm-1',
      type: 'multiple-choice',
      question: 'What is the primary architectural pattern used in modern Large Language Models like GPT?',
      options: [
        'Encoder-only (like BERT)',
        'Decoder-only (like GPT)',
        'Encoder-decoder (like T5)',
        'Recurrent Neural Network'
      ],
      correctAnswer: 1,
      explanation: 'Modern LLMs like GPT use decoder-only architecture with causal attention, meaning each token can only attend to previous tokens. This enables autoregressive text generation.',
      difficulty: 'Intermediate',
      category: 'llm-architecture',
      interviewFrequency: 'Very High'
    },
    {
      id: 'llm-2',
      type: 'multiple-choice',
      question: 'What is the core pre-training objective for language models?',
      options: [
        'Masked language modeling',
        'Next token prediction',
        'Sentence classification',
        'Sequence-to-sequence translation'
      ],
      correctAnswer: 1,
      explanation: 'Next token prediction is the core objective: given a sequence of tokens, predict the most likely next token. This simple objective teaches the model to understand language patterns.',
      difficulty: 'Beginner',
      category: 'pre-training',
      interviewFrequency: 'Very High'
    }
  ]
};
