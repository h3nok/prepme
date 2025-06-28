import { QuizQuestion } from '../types/LearningModule';

export const diffusionQuiz: QuizQuestion[] = [
  {
    id: 'diffusion-1',
    question: 'What is the core principle behind diffusion models?',
    options: [
      'Learning to generate data directly from noise',
      'Learning to reverse a gradual noising process',
      'Learning to discriminate between real and fake data',
      'Learning to compress data efficiently'
    ],
    correctAnswer: 1,
    explanation: 'Diffusion models learn to reverse a gradual noising process. They start with clean data, gradually add noise until it becomes pure noise, then learn to reverse this process to generate new samples.',
    difficulty: 'Easy',
    concept: 'diffusion-fundamentals'
  },
  {
    id: 'diffusion-2',
    question: 'What is the mathematical formulation of the forward process in diffusion models?',
    options: [
      'q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)',
      'q(x_t | x_{t-1}) = N(x_t; x_{t-1}, β_t I)',
      'q(x_t | x_{t-1}) = N(x_t; sqrt(β_t) x_{t-1}, (1-β_t) I)',
      'q(x_t | x_{t-1}) = N(x_t; x_{t-1} + β_t, I)'
    ],
    correctAnswer: 0,
    explanation: 'The forward process is q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I) where β_t is the noise schedule that controls how much noise is added at each step.',
    difficulty: 'Medium',
    concept: 'diffusion-fundamentals'
  },
  {
    id: 'diffusion-3',
    question: 'What is the closed-form expression for q(x_t | x_0)?',
    options: [
      'q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1-ᾱ_t)I)',
      'q(x_t | x_0) = N(x_t; x_0, ᾱ_t I)',
      'q(x_t | x_0) = N(x_t; sqrt(1-ᾱ_t) x_0, ᾱ_t I)',
      'q(x_t | x_0) = N(x_t; x_0 + ᾱ_t, I)'
    ],
    correctAnswer: 0,
    explanation: 'The closed-form expression is q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1-ᾱ_t)I) where ᾱ_t = ∏(1-β_i) is the cumulative product of (1-β_i) up to step t.',
    difficulty: 'Hard',
    concept: 'diffusion-fundamentals'
  },
  {
    id: 'diffusion-4',
    question: 'What is the training objective for DDPM (Denoising Diffusion Probabilistic Models)?',
    options: [
      'MSE between predicted and actual noise',
      'Cross-entropy loss between predicted and actual images',
      'Adversarial loss between generator and discriminator',
      'KL divergence between forward and reverse processes'
    ],
    correctAnswer: 0,
    explanation: 'DDPM trains a neural network to predict the noise that was added during the forward process. The loss is the MSE between predicted and actual noise: L = E[||ε - ε_θ(x_t, t)||²].',
    difficulty: 'Medium',
    concept: 'ddpm'
  },
  {
    id: 'diffusion-5',
    question: 'What is the mathematical formulation of the DDPM training loss?',
    options: [
      'L = E[||ε - ε_θ(x_t, t)||²]',
      'L = E[||x_0 - x_θ(x_t, t)||²]',
      'L = E[||x_t - x_θ(x_0, t)||²]',
      'L = E[||ε_θ(x_t, t)||²]'
    ],
    correctAnswer: 0,
    explanation: 'The DDPM training loss is L = E[||ε - ε_θ(x_t, t)||²] where the model learns to predict the noise ε that was added during the forward process.',
    difficulty: 'Medium',
    concept: 'ddpm'
  },
  {
    id: 'diffusion-6',
    question: 'What is the key innovation of DDIM (Denoising Diffusion Implicit Models)?',
    options: [
      'Using a different noise schedule',
      'Making the reverse process deterministic',
      'Adding more layers to the neural network',
      'Using a different loss function'
    ],
    correctAnswer: 1,
    explanation: 'DDIM makes the reverse process deterministic by removing the stochastic noise component. This allows for much faster sampling with fewer steps while maintaining quality.',
    difficulty: 'Medium',
    concept: 'ddim'
  },
  {
    id: 'diffusion-7',
    question: 'What is the DDIM sampling formula?',
    options: [
      'x_{t-1} = sqrt(ᾱ_{t-1}) x̂_0 + sqrt(1-ᾱ_{t-1}) ε_θ(x_t, t)',
      'x_{t-1} = μ_θ(x_t, t) + σ_t z',
      'x_{t-1} = x_t - ε_θ(x_t, t)',
      'x_{t-1} = x_t + ε_θ(x_t, t)'
    ],
    correctAnswer: 0,
    explanation: 'The DDIM sampling formula is x_{t-1} = sqrt(ᾱ_{t-1}) x̂_0 + sqrt(1-ᾱ_{t-1}) ε_θ(x_t, t) where x̂_0 is the predicted clean image.',
    difficulty: 'Hard',
    concept: 'ddim'
  },
  {
    id: 'diffusion-8',
    question: 'What is classifier-free guidance?',
    options: [
      'Using a separate classifier to guide generation',
      'Using two diffusion models (conditioned and unconditioned)',
      'Using reinforcement learning to guide generation',
      'Using a discriminator network'
    ],
    correctAnswer: 1,
    explanation: 'Classifier-free guidance uses two diffusion models: one conditioned on a prompt and one unconditioned. The difference between their predictions guides the generation process.',
    difficulty: 'Medium',
    concept: 'classifier-guidance'
  },
  {
    id: 'diffusion-9',
    question: 'What is the formula for classifier-free guidance?',
    options: [
      'ε_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + s(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))',
      'ε_θ(x_t, t, c) = ε_θ(x_t, t, c) + s ε_θ(x_t, t, ∅)',
      'ε_θ(x_t, t, c) = s ε_θ(x_t, t, c)',
      'ε_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + ε_θ(x_t, t, c)'
    ],
    correctAnswer: 0,
    explanation: 'The classifier-free guidance formula is ε_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + s(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅)) where s is the guidance scale.',
    difficulty: 'Hard',
    concept: 'classifier-guidance'
  },
  {
    id: 'diffusion-10',
    question: 'What is the main advantage of Latent Diffusion Models (LDM)?',
    options: [
      'Better image quality',
      'Reduced computational cost',
      'Faster training',
      'More stable training'
    ],
    correctAnswer: 1,
    explanation: 'Latent Diffusion Models operate in a compressed latent space instead of pixel space, dramatically reducing computational cost while maintaining quality.',
    difficulty: 'Medium',
    concept: 'advanced-topics'
  },
  {
    id: 'diffusion-11',
    question: 'What is the key insight of Consistency Models?',
    options: [
      'Learning to map any point on the diffusion trajectory directly to clean data',
      'Using a different noise schedule',
      'Adding more conditioning information',
      'Using a different architecture'
    ],
    correctAnswer: 0,
    explanation: 'Consistency models learn to map any point on the diffusion trajectory directly to the clean data, enabling single-step generation instead of many iterative steps.',
    difficulty: 'Medium',
    concept: 'advanced-topics'
  },
  {
    id: 'diffusion-12',
    question: 'What is the consistency function in Consistency Models?',
    options: [
      'f_θ(x_t, t) = x_0',
      'f_θ(x_t, t) = x_{t-1}',
      'f_θ(x_t, t) = ε',
      'f_θ(x_t, t) = μ_θ(x_t, t)'
    ],
    correctAnswer: 0,
    explanation: 'The consistency function f_θ(x_t, t) = x_0 maps any noisy point x_t directly to the clean data x_0, enabling single-step generation.',
    difficulty: 'Hard',
    concept: 'advanced-topics'
  },
  {
    id: 'diffusion-13',
    question: 'What is the typical noise schedule used in diffusion models?',
    options: [
      'Linear schedule from 0.0001 to 0.02',
      'Exponential schedule',
      'Constant noise level',
      'Random noise level'
    ],
    correctAnswer: 0,
    explanation: 'The typical noise schedule is linear from β_1 = 0.0001 to β_T = 0.02, ensuring smooth transition from clean data to pure noise.',
    difficulty: 'Easy',
    concept: 'diffusion-fundamentals'
  },
  {
    id: 'diffusion-14',
    question: 'What is the main advantage of diffusion models over GANs?',
    options: [
      'Faster generation',
      'More stable training',
      'Lower computational cost',
      'Better image quality'
    ],
    correctAnswer: 1,
    explanation: 'Diffusion models have more stable training compared to GANs, which often suffer from mode collapse and training instability.',
    difficulty: 'Easy',
    concept: 'diffusion-fundamentals'
  },
  {
    id: 'diffusion-15',
    question: 'What is the typical number of sampling steps for DDPM?',
    options: [
      '10-50 steps',
      '100-500 steps',
      '1000 steps',
      '10000 steps'
    ],
    correctAnswer: 2,
    explanation: 'DDPM typically uses 1000 sampling steps, which is why it was initially slow. DDIM and other improvements have reduced this to 10-50 steps.',
    difficulty: 'Easy',
    concept: 'ddpm'
  }
]; 