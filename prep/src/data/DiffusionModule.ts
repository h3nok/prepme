import { LearningModule } from '../types/LearningModule';

export const diffusionModule: LearningModule = {
  id: 'diffusion',
  title: 'Diffusion Models',
  description: 'Master the theory and implementation of diffusion models, from DDPM to DDIM and beyond. Understand how these models generate high-quality images and their applications in generative AI.',
  color: '#dc2626',
  icon: 'Image',
  progress: 0,
  estimatedHours: 12,
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
            },
            {
              id: 'reverse-process',
              latex: 'p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\Sigma_\\theta(x_t, t))',
              explanation: 'Reverse process: learned denoising with neural network μ_θ',
              interactive: true
            }
          ],
          keyPoints: [
            "Forward process is fixed Markov chain that adds Gaussian noise",
            "Reverse process is learned to denoise step by step",
            "Training is stable compared to adversarial methods",
            "Can generate diverse, high-quality samples",
            "Slower inference due to iterative sampling process"
          ],
          interviewTips: [
            "Emphasize the stability advantage over GANs",
            "Explain the intuition behind learning to reverse noise",
            "Mention the trade-off between quality and sampling speed",
            "Compare with other generative models"
          ],
          practiceQuestions: [
            "Why is the forward process fixed while reverse is learned?",
            "How do diffusion models avoid mode collapse?",
            "What makes diffusion training more stable than GANs?",
            "Explain the role of the noise schedule"
          ]
        },
        {
          id: 'ddpm-training',
          title: 'DDPM Training Objective & Implementation',
          content: {
            tier1: "DDPM training involves predicting the noise added at each timestep. The model learns to estimate ε_θ(x_t, t) - the noise that was added to create x_t from x_0.",
            tier2: "The loss function is simplified to ||ε - ε_θ(x_t, t)||², where ε is the actual noise and ε_θ is the predicted noise. This makes training straightforward and stable.",
            tier3: "Key implementation details: noise scheduling (β_t), timestep embedding, U-Net architecture with attention, and techniques for stable training at different noise levels."
          },
          mathNotations: [
            {
              id: 'ddpm-loss',
              latex: 'L_{simple} = \\mathbb{E}_{t, x_0, \\epsilon} [||\\epsilon - \\epsilon_\\theta(x_t, t)||^2]',
              explanation: 'Simplified DDPM loss: predict the noise added at timestep t',
              interactive: true
            },
            {
              id: 'reparameterization',
              latex: 'x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon',
              explanation: 'Reparameterization trick allows direct sampling at any timestep',
              interactive: true
            }
          ],
          keyPoints: [
            "Training objective is simple: predict added noise",
            "Reparameterization enables efficient training",
            "U-Net architecture with attention works well",
            "Noise schedule affects generation quality",
            "Timestep embedding helps model understand noise level"
          ],
          interviewTips: [
            "Explain the intuition behind predicting noise vs data",
            "Discuss the importance of the reparameterization trick",
            "Mention the role of the U-Net architecture",
            "Compare with VAE and GAN training objectives"
          ],
          practiceQuestions: [
            "Why predict noise instead of the clean image?",
            "How does the reparameterization trick help training?",
            "What architecture choices work best for diffusion?",
            "Explain the role of timestep conditioning"
          ]
        },
        {
          id: 'sampling-methods',
          title: 'Sampling Methods: DDPM to DDIM and Beyond',
          content: {
            tier1: "DDPM requires 1000+ sampling steps for high quality. DDIM reduces this to 50-100 steps by making the process deterministic and skipping timesteps.",
            tier2: "DDIM uses the same trained model but changes the sampling process to be non-Markovian. This allows larger steps and faster generation with minimal quality loss.",
            tier3: "Advanced sampling includes DPM-Solver, Euler methods, and learned variance schedules. Recent work on consistency models and progressive distillation further reduces steps."
          },
          mathNotations: [
            {
              id: 'ddim-update',
              latex: 'x_{t-1} = \\sqrt{\\alpha_{t-1}} \\left(\\frac{x_t - \\sqrt{1-\\alpha_t}\\epsilon_\\theta(x_t,t)}{\\sqrt{\\alpha_t}}\\right) + \\sqrt{1-\\alpha_{t-1}}\\epsilon_\\theta(x_t,t)',
              explanation: 'DDIM sampling formula for deterministic generation',
              interactive: true
            }
          ],
          keyPoints: [
            "DDIM enables fast sampling with fewer steps",
            "Deterministic sampling allows reproducible generation",
            "Advanced ODE solvers further improve efficiency",
            "Classifier-free guidance improves conditional generation",
            "Consistency models promise single-step generation"
          ],
          interviewTips: [
            "Explain the speed vs quality trade-off in sampling",
            "Compare DDPM and DDIM sampling procedures",
            "Mention recent advances in fast sampling",
            "Discuss practical considerations for deployment"
          ],
          practiceQuestions: [
            "How does DDIM achieve faster sampling?",
            "What's the difference between DDPM and DDIM sampling?",
            "Explain classifier-free guidance",
            "How do consistency models work?"
          ]
        }
      ]
    },
    {
      id: 'conditional-generation',
      title: 'Conditional Generation & Guidance',
      description: 'Techniques for controlled generation with diffusion models',
      slides: [
        {
          id: 'classifier-guidance',
          title: 'Classifier Guidance for Conditional Generation',
          content: {
            tier1: "Classifier guidance uses a separately trained classifier to steer the diffusion process toward desired classes. The gradient of the classifier provides guidance.",
            tier2: "At each denoising step, we add the gradient ∇_x log p(y|x_t) scaled by guidance strength. This biases generation toward the target class y.",
            tier3: "Classifier guidance can achieve strong conditioning but requires training additional classifiers and can reduce sample diversity with high guidance scales."
          },
          mathNotations: [
            {
              id: 'classifier-guidance-formula',
              latex: '\\tilde{\\epsilon}_\\theta(x_t, t) = \\epsilon_\\theta(x_t, t) - \\sqrt{1-\\bar{\\alpha}_t} \\cdot s \\cdot \\nabla_{x_t} \\log p_\\phi(y|x_t)',
              explanation: 'Classifier guidance modifies noise prediction with classifier gradients',
              interactive: true
            }
          ],
          keyPoints: [
            "Requires training separate classifier on noisy images",
            "Guidance strength s controls conditioning vs diversity trade-off",
            "Can condition on any property the classifier recognizes",
            "Computational overhead from classifier evaluation",
            "May reduce sample diversity at high guidance scales"
          ],
          interviewTips: [
            "Explain why classifiers need to work on noisy images",
            "Discuss the guidance strength trade-off",
            "Compare with unconditional generation quality",
            "Mention computational considerations"
          ]
        },
        {
          id: 'classifier-free-guidance',
          title: 'Classifier-Free Guidance: The Modern Standard',
          content: {
            tier1: "Classifier-free guidance trains a single model to handle both conditional and unconditional generation, eliminating the need for separate classifiers.",
            tier2: "During training, conditioning information is randomly dropped (e.g., 10% of the time). At inference, we interpolate between conditional and unconditional predictions.",
            tier3: "The guidance formula amplifies the difference between conditional and unconditional outputs, effectively steering generation without external classifiers."
          },
          mathNotations: [
            {
              id: 'cfg-formula',
              latex: '\\tilde{\\epsilon}_\\theta(x_t, t, c) = \\epsilon_\\theta(x_t, t, \\emptyset) + s \\cdot (\\epsilon_\\theta(x_t, t, c) - \\epsilon_\\theta(x_t, t, \\emptyset))',
              explanation: 'Classifier-free guidance interpolates conditional and unconditional predictions',
              interactive: true
            }
          ],
          keyPoints: [
            "Single model handles both conditional and unconditional tasks",
            "No need for additional classifier training",
            "Random conditioning dropout during training",
            "Guidance scale controls conditioning strength",
            "Widely adopted in modern text-to-image models"
          ],
          interviewTips: [
            "Explain the advantage over classifier guidance",
            "Describe the training procedure with dropout",
            "Discuss why this became the standard approach",
            "Mention applications in text-to-image models"
          ]
        }
      ]
    },
    {
      id: 'latent-diffusion',
      title: 'Latent Diffusion Models',
      description: 'Efficient diffusion in compressed latent space',
      slides: [
        {
          id: 'latent-space-diffusion',
          title: 'Why Diffuse in Latent Space?',
          content: {
            tier1: "Latent Diffusion Models (LDMs) perform diffusion in a compressed latent space rather than raw pixel space, dramatically reducing computational requirements.",
            tier2: "A pre-trained autoencoder (VAE) compresses images to latent space. Diffusion happens in this space, then the decoder reconstructs final images.",
            tier3: "This approach maintains generation quality while reducing compute by 8-64x. It enables high-resolution generation and makes diffusion models more accessible."
          },
          mathNotations: [
            {
              id: 'latent-encoding',
              latex: 'z = E(x), \\quad x = D(z)',
              explanation: 'Encoder E compresses to latent z, decoder D reconstructs image x',
              interactive: false
            }
          ],
          keyPoints: [
            "Massive computational savings (8-64x reduction)",
            "Enables high-resolution image generation",
            "Pre-trained VAE provides good latent representation",
            "Maintains generation quality in compressed space",
            "Foundation for Stable Diffusion and similar models"
          ],
          interviewTips: [
            "Emphasize the computational efficiency gains",
            "Explain the role of the pre-trained VAE",
            "Compare computational costs with pixel-space diffusion",
            "Mention specific implementations like Stable Diffusion"
          ]
        },
        {
          id: 'stable-diffusion-architecture',
          title: 'Stable Diffusion: Architecture & Components',
          content: {
            tier1: "Stable Diffusion combines a VAE for image compression, a U-Net for denoising in latent space, and CLIP for text conditioning.",
            tier2: "The architecture includes cross-attention layers that inject text embeddings into the U-Net, enabling text-to-image generation with rich semantic control.",
            tier3: "Key innovations include efficient attention mechanisms, conditioning techniques, and safety filters. The model can be fine-tuned for specific domains and styles."
          },
          keyPoints: [
            "VAE + U-Net + CLIP text encoder architecture",
            "Cross-attention enables text conditioning",
            "Latent space diffusion for efficiency",
            "Safety filtering for responsible deployment",
            "Extensible through fine-tuning and LoRA"
          ],
          interviewTips: [
            "Explain each component's role clearly",
            "Describe how text conditioning works",
            "Mention the importance of safety considerations",
            "Discuss fine-tuning and customization options"
          ]
        }
      ]
    }
  ]
};
