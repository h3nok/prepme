import { Quiz } from '../types/LearningModule';

export const transformerQuiz: Quiz = {
  id: 'transformer-quiz',
  title: 'Transformer Architecture Quiz',
  description: 'Test your understanding of transformer architecture, attention mechanisms, and key innovations',
  moduleId: 'transformers',
  timeLimit: 30,
  passingScore: 75,
  questions: [
    {
      id: 'attention-1',
      type: 'multiple-choice',
      question: 'What is the main innovation of the Transformer architecture?',
      options: [
        'Convolutional layers',
        'Self-attention mechanism',
        'Recurrent connections',
        'Pooling operations'
      ],
      correctAnswer: 1,
      explanation: 'The self-attention mechanism is the key innovation that allows Transformers to process sequences in parallel and capture long-range dependencies efficiently.',
      difficulty: 'Beginner',
      category: 'attention',
      interviewFrequency: 'Very High'
    },
    {
      id: 'attention-2',
      type: 'multiple-choice',
      question: 'What is the computational complexity of self-attention for a sequence of length n?',
      options: [
        'O(n)',
        'O(n log n)',
        'O(n²)',
        'O(n³)'
      ],
      correctAnswer: 2,
      explanation: 'Self-attention has O(n²) complexity because it computes attention weights between all pairs of positions in the sequence.',
      difficulty: 'Intermediate',
      category: 'attention',
      interviewFrequency: 'Very High'
    }
  ]
};
