import { LearningModule } from '../types/LearningModule';

export const fundamentalsModule: LearningModule = {
  id: 'fundamentals',
  title: 'Fundamentals of Machine Learning',
  description: 'Master the mathematical foundations: statistics, probability, linear algebra, and core ML concepts that power modern AI systems.',
  color: '#ff6b35',
  icon: 'Calculator',
  progress: 0,
  estimatedHours: 12,
  prerequisites: [],
  difficulty: 'Beginner',
  concepts: [
    {
      id: 'probability-basics',
      title: 'Probability Theory Foundations',
      description: 'Core probability concepts that underpin all machine learning algorithms',
      slides: [
        {
          id: 'prob-intro',
          title: 'What is Probability?',
          content: {
            tier1: "Probability measures uncertainty - it quantifies how likely events are to occur on a scale from 0 (impossible) to 1 (certain).",
            tier2: "In ML, we use probability to model uncertainty in data, make predictions with confidence levels, and handle noisy real-world information.",
            tier3: "Formally, for a sample space Ω and event A ⊆ Ω, P(A) satisfies: P(Ω) = 1, P(∅) = 0, and P(A∪B) = P(A) + P(B) for disjoint events."
          },
          mathNotations: [
            {
              id: 'basic-prob',
              latex: 'P(A) = \\frac{\\text{Number of favorable outcomes}}{\\text{Total number of outcomes}}',
              explanation: 'Classical probability definition for equally likely outcomes',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'prob-wheel',
              type: 'interactive',
              component: 'ProbabilityWheel',
              data: { events: ['Heads', 'Tails'], probabilities: [0.5, 0.5] },
              controls: [
                {
                  id: 'bias',
                  type: 'slider',
                  label: 'Coin Bias',
                  range: [0, 1],
                  defaultValue: 0.5
                }
              ]
            }
          ]
        },
        {
          id: 'conditional-prob',
          title: 'Conditional Probability',
          content: {
            tier1: "Conditional probability P(A|B) tells us the probability of event A occurring given that event B has already occurred.",
            tier2: "This is crucial in ML for updating beliefs with new evidence - the foundation of Bayesian learning and many classification algorithms.",
            tier3: "Mathematically: P(A|B) = P(A∩B)/P(B), assuming P(B) > 0. This leads to Bayes' theorem: P(A|B) = P(B|A)P(A)/P(B)."
          },
          mathNotations: [
            {
              id: 'conditional',
              latex: 'P(A|B) = \\frac{P(A \\cap B)}{P(B)}',
              explanation: 'Definition of conditional probability',
              interactive: false
            },
            {
              id: 'bayes',
              latex: 'P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}',
              explanation: "Bayes' theorem - fundamental to machine learning",
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'venn-conditional',
              type: 'interactive',
              component: 'VennDiagram',
              data: { setA: 'Disease', setB: 'Positive Test' },
              controls: [
                {
                  id: 'prevalence',
                  type: 'slider',
                  label: 'Disease Prevalence',
                  range: [0.01, 0.1],
                  defaultValue: 0.05
                },
                {
                  id: 'sensitivity',
                  type: 'slider',
                  label: 'Test Sensitivity',
                  range: [0.8, 0.99],
                  defaultValue: 0.95
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'distributions',
      title: 'Probability Distributions',
      description: 'Key probability distributions used throughout machine learning',
      slides: [
        {
          id: 'normal-dist',
          title: 'The Normal Distribution',
          content: {
            tier1: "The normal (Gaussian) distribution is the most important distribution in statistics and ML, describing many natural phenomena.",
            tier2: "Its bell curve shape is defined by two parameters: mean (μ) and variance (σ²). About 68% of values fall within 1 standard deviation of the mean.",
            tier3: "PDF: f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²)). Central Limit Theorem explains why it's so common in nature and ML."
          },
          mathNotations: [
            {
              id: 'normal-pdf',
              latex: 'f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}',
              explanation: 'Probability density function of normal distribution',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'normal-curve',
              type: 'interactive',
              component: 'NormalDistribution',
              data: { mu: 0, sigma: 1 },
              controls: [
                {
                  id: 'mean',
                  type: 'slider',
                  label: 'Mean (μ)',
                  range: [-3, 3],
                  defaultValue: 0
                },
                {
                  id: 'std',
                  type: 'slider',
                  label: 'Standard Deviation (σ)',
                  range: [0.5, 3],
                  defaultValue: 1
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'linear-algebra',
      title: 'Linear Algebra for ML',
      description: 'Vectors, matrices, and operations that form the backbone of machine learning',
      slides: [
        {
          id: 'vectors-basics',
          title: 'Vectors and Vector Operations',
          content: {
            tier1: "Vectors represent data points, features, and model parameters. They're ordered lists of numbers that can be added, scaled, and multiplied.",
            tier2: "In ML, each data sample is often represented as a vector in high-dimensional space. Vector operations let us compute similarities, distances, and transformations.",
            tier3: "Key operations: addition (component-wise), scalar multiplication, dot product (a·b = Σaᵢbᵢ), and norms (||a|| = √(Σaᵢ²))."
          },
          mathNotations: [
            {
              id: 'dot-product',
              latex: '\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^n a_i b_i = |\\mathbf{a}||\\mathbf{b}|\\cos\\theta',
              explanation: 'Dot product measures similarity between vectors',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'vector-space',
              type: 'interactive',
              component: 'VectorVisualization',
              data: { dimension: 2 },
              controls: [
                {
                  id: 'vector-a',
                  type: 'slider',
                  label: 'Vector A components',
                  range: [-5, 5],
                  defaultValue: [3, 2]
                },
                {
                  id: 'vector-b',
                  type: 'slider',
                  label: 'Vector B components',
                  range: [-5, 5],
                  defaultValue: [1, 4]
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'optimization',
      title: 'Optimization Fundamentals',
      description: 'How machines learn through optimization - finding the best parameters',
      slides: [
        {
          id: 'gradient-descent',
          title: 'Gradient Descent',
          content: {
            tier1: "Gradient descent is how neural networks learn - it finds the minimum of a function by following the steepest downward slope.",
            tier2: "The gradient points in the direction of steepest increase, so we move in the opposite direction. Learning rate controls how big steps we take.",
            tier3: "Update rule: θ = θ - α∇f(θ), where α is learning rate and ∇f is the gradient. Variants include SGD, momentum, and Adam."
          },
          mathNotations: [
            {
              id: 'gradient-update',
              latex: '\\theta_{t+1} = \\theta_t - \\alpha \\nabla f(\\theta_t)',
              explanation: 'Basic gradient descent update rule',
              interactive: true
            }
          ],
          visualizations: [
            {
              id: 'gradient-descent-viz',
              type: 'animation',
              component: 'GradientDescentAnimation',
              data: { function: 'quadratic' },
              controls: [
                {
                  id: 'learning-rate',
                  type: 'slider',
                  label: 'Learning Rate',
                  range: [0.01, 0.5],
                  defaultValue: 0.1
                },
                {
                  id: 'start-point',
                  type: 'slider',
                  label: 'Starting Point',
                  range: [-5, 5],
                  defaultValue: 3
                }
              ]
            }
          ]
        }
      ]
    },
    {
      id: 'bias-variance',
      title: 'Bias-Variance Tradeoff',
      description: 'Understanding the fundamental tradeoff in machine learning model performance',
      slides: [
        {
          id: 'bias-variance-intro',
          title: 'The Bias-Variance Decomposition',
          content: {
            tier1: "Every ML model faces a tradeoff: bias (systematic error) vs variance (sensitivity to training data). Understanding this guides model selection.",
            tier2: "High bias = underfitting (too simple), high variance = overfitting (too complex). Sweet spot minimizes total error = bias² + variance + noise.",
            tier3: "Mathematically: E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ². This decomposition is fundamental to model selection and regularization."
          },
          mathNotations: [
            {
              id: 'bias-variance-decomp',
              latex: '\\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Noise}',
              explanation: 'The fundamental bias-variance decomposition',
              interactive: false
            }
          ],
          visualizations: [
            {
              id: 'bias-variance-demo',
              type: 'interactive',
              component: 'BiasVarianceDemo',
              data: { dataPoints: 100 },
              controls: [
                {
                  id: 'model-complexity',
                  type: 'slider',
                  label: 'Model Complexity',
                  range: [1, 15],
                  defaultValue: 3
                },
                {
                  id: 'noise-level',
                  type: 'slider',
                  label: 'Noise Level',
                  range: [0, 1],
                  defaultValue: 0.2
                }
              ]
            }
          ]
        }
      ]
    }
  ]
};
