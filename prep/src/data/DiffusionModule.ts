import { LearningModule } from '../types/LearningModule';

export const diffusionModule: LearningModule = {
  id: 'diffusion',
  title: 'Diffusion Models',
  description: 'Master the theory and implementation of diffusion models, from DDPM to DDIM and beyond. Understand how these models generate high-quality images and their applications in generative AI.',
  color: '#dc2626',
  icon: 'Image',
  progress: 0,
  estimatedHours: 10,
  prerequisites: ['fundamentals', 'transformers'],
  difficulty: 'Advanced',
  concepts: [
    {
      id: 'diffusion-fundamentals',
      title: 'Diffusion Model Fundamentals',
      description: 'Understanding the core principles of diffusion models and the forward/reverse process',
      slides: [
        {
          id: 'diffusion-overview',
          title: 'What are Diffusion Models?',
          content: {
            tier1: "Diffusion models are generative models that learn to reverse a gradual noising process. They start with data and gradually add noise until it becomes pure noise, then learn to reverse this process.",
            tier2: "The key insight is that if we can learn to reverse a simple forward process (adding noise), we can generate high-quality samples. The forward process is fixed and the reverse process is learned.",
            tier3: "Diffusion models have achieved state-of-the-art results in image generation, surpassing GANs in many benchmarks. They are also more stable to train and can generate diverse, high-quality samples."
          },
          mathNotations: [
            {
              id: 'forward-process',
              latex: 'q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)',
              explanation: 'Forward process: gradually adding Gaussian noise with variance schedule β_t',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'diffusion-process-demo',
              type: 'interactive',
              component: 'DiffusionProcess',
              data: { 
                steps: 1000,
                showForward: true,
                showReverse: true
              },
              controls: [
                {
                  id: 'steps',
                  type: 'slider',
                  label: 'Number of Steps',
                  range: [10, 1000],
                  defaultValue: 1000
                },
                {
                  id: 'show-forward',
                  type: 'toggle',
                  label: 'Show Forward Process',
                  defaultValue: true
                },
                {
                  id: 'show-reverse',
                  type: 'toggle',
                  label: 'Show Reverse Process',
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
