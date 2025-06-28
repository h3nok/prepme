import { Quiz } from '../types/LearningModule';

export const diffusionQuiz: Quiz = {
  id: 'diffusion-quiz',
  title: 'Diffusion Models Quiz',
  description: 'Test your understanding of diffusion models, from DDPM fundamentals to advanced sampling techniques',
  moduleId: 'diffusion',
  timeLimit: 30,
  passingScore: 75,
  questions: [
    {
      id: 'diffusion-1',
      type: 'multiple-choice',
      question: 'What is the core principle behind diffusion models?',
      options: [
        'Learning to generate data directly from noise',
        'Learning to reverse a gradual noising process',
        'Learning to discriminate between real and fake data',
        'Learning to compress data efficiently'
      ],
      correctAnswer: 1,
      explanation: 'Diffusion models learn to reverse a gradual noising process. They start with clean data, gradually add noise until it becomes pure noise, then learn to reverse this process to generate new samples.',
      difficulty: 'Beginner',
      category: 'diffusion-fundamentals',
      interviewFrequency: 'High'
    },
    {
      id: 'diffusion-2',
      type: 'multiple-choice',
      question: 'What is the mathematical formulation of the forward process in diffusion models?',
      options: [
        'q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)',
        'q(x_t | x_{t-1}) = N(x_t; x_{t-1}, β_t I)',
        'q(x_t | x_{t-1}) = N(x_t; sqrt(β_t) x_{t-1}, (1-β_t) I)',
        'q(x_t | x_{t-1}) = N(x_t; x_{t-1} + β_t, I)'
      ],
      correctAnswer: 0,
      explanation: 'The forward process is q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I) where β_t is the noise schedule that controls how much noise is added at each step.',
      difficulty: 'Intermediate',
      category: 'diffusion-fundamentals',
      interviewFrequency: 'High'
    }
  ]
};
