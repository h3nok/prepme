import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowLeft, Image, Layers, Zap, CheckCircle, Book } from 'lucide-react';
import { Link } from 'react-router-dom';

import Card from '../components/Card';
import Math from '../components/Math';
import Quiz from '../components/Quiz';

const PageHeader = styled.div`
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const BackButton = styled(Link)`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textSecondary};
  text-decoration: none;
  margin-bottom: ${props => props.theme.spacing.md};
  transition: color 0.2s ease;

  &:hover {
    color: ${props => props.theme.colors.primary};
  }
`;

const PageTitle = styled.h1`
  font-size: 2.5rem;
  font-weight: 800;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
  
  span {
    color: ${props => props.theme.colors.primary};
  }
`;

const PageDescription = styled.p`
  font-size: 1.2rem;
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
  max-width: 800px;
`;

const ContentSection = styled.section`
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const SectionTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.lg};
  font-size: 1.75rem;
  font-weight: 700;
  
  &:before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.75rem;
    background: ${props => props.theme.colors.primary};
    margin-right: ${props => props.theme.spacing.md};
    vertical-align: bottom;
  }
`;

const ConceptGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const FormulaCard = styled(Card)`
  margin: ${props => props.theme.spacing.lg} 0;
  
  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.md};
  }
`;

const ProcessDiagram = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.lg} 0;
  text-align: center;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.lg};
  }

  .process-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: ${props => props.theme.spacing.md};
    margin-top: ${props => props.theme.spacing.lg};
  }

  .step {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.md};
    border-radius: ${props => props.theme.radii.md};
    border: 1px solid ${props => props.theme.colors.border};

    .step-number {
      background: ${props => props.theme.colors.primary};
      color: white;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      font-weight: 600;
      margin: 0 auto ${props => props.theme.spacing.sm};
    }

    .step-title {
      font-weight: 600;
      color: ${props => props.theme.colors.primary};
      font-size: 0.9rem;
      margin-bottom: ${props => props.theme.spacing.xs};
    }

    .step-desc {
      font-size: 0.8rem;
      color: ${props => props.theme.colors.textSecondary};
    }
  }
`;

const KeyPoints = styled.ul`
  list-style: none;
  padding: 0;
  
  li {
    display: flex;
    align-items: flex-start;
    gap: ${props => props.theme.spacing.sm};
    margin-bottom: ${props => props.theme.spacing.sm};
    padding: ${props => props.theme.spacing.sm};
    background: ${props => props.theme.colors.surface};
    border-radius: ${props => props.theme.radii.md};
    border-left: 3px solid ${props => props.theme.colors.primary};
    
    svg {
      color: ${props => props.theme.colors.success};
      margin-top: 2px;
      flex-shrink: 0;
    }
  }
`;

const diffusionQuestions = [
  {
    id: "diffusion-1",
    question: "What is the core principle behind diffusion models?",
    options: [
      "Learning to compress data efficiently",
      "Learning to reverse a gradual noise addition process",
      "Learning to classify images directly",
      "Learning to encode images into latent space"
    ],
    correct: 1,
    explanation: "Diffusion models learn to reverse a forward diffusion process that gradually adds noise to data until it becomes pure noise, then learns to denoise step by step.",
    difficulty: "medium" as const
  },
  {
    id: "diffusion-2",
    question: "What is the key advantage of the reparameterization trick in DDPM?",
    options: [
      "It reduces memory usage",
      "It allows training on any timestep directly",
      "It speeds up inference",
      "It improves image quality"
    ],
    correct: 1,
    explanation: "The reparameterization trick allows the model to predict the noise added at any timestep t directly, rather than having to apply the forward process sequentially, enabling efficient training.",
    difficulty: "hard" as const
  },
  {
    id: "diffusion-3",
    question: "How does classifier-free guidance work in diffusion models?",
    options: [
      "It removes the need for any conditioning",
      "It interpolates between conditional and unconditional predictions",
      "It uses a separate classifier for guidance",
      "It only works with text conditioning"
    ],
    correct: 1,
    explanation: "Classifier-free guidance combines conditional and unconditional model predictions: ε_guided = ε_uncond + w(ε_cond - ε_uncond), where w controls guidance strength.",
    difficulty: "medium" as const
  }
];

const DiffusionPage: React.FC = () => {
  const [showQuiz, setShowQuiz] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <PageHeader>
        <BackButton to="/">
          <ArrowLeft size={20} />
          Back to Home
        </BackButton>
        <PageTitle>
          🎨 <span>Diffusion</span> Models
        </PageTitle>
        <PageDescription>
          Master the generative powerhouse behind DALL-E, Midjourney, and Stable Diffusion. 
          From mathematical foundations to practical implementation, understand how diffusion 
          models revolutionized AI-generated content.
        </PageDescription>
      </PageHeader>

      <ContentSection>
        <SectionTitle>Core Concepts</SectionTitle>
        <ConceptGrid>
          <Card>
            <h4><Layers />Forward Diffusion Process</h4>
            <p>
              Gradually adds Gaussian noise to data over T timesteps until the original 
              signal becomes indistinguishable from pure noise. This process is fixed and analytical.
            </p>
          </Card>
          
          <Card variant="accent">
            <h4><Zap />Reverse Diffusion Process</h4>
            <p>
              A learned neural network that reverses the forward process, gradually 
              denoising pure noise back into clean data samples.
            </p>
          </Card>
          
          <Card variant="purple">
            <h4><Image />Score-Based Generation</h4>
            <p>
              Models learn to estimate the score function (gradient of log probability), 
              enabling high-quality sample generation through iterative denoising.
            </p>
          </Card>
        </ConceptGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Mathematical Foundation</SectionTitle>
        
        <FormulaCard>
          <h4>Forward Diffusion Process (DDPM)</h4>
          <p>Gradually corrupts data with Gaussian noise over T steps:</p>
          <Math block>
            {"q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)"}
          </Math>
          <p>Where β_t is a variance schedule that controls noise addition rate.</p>
        </FormulaCard>

        <FormulaCard>
          <h4>Reparameterization Trick</h4>
          <p>Allows direct sampling at any timestep t:</p>
          <Math block>
            {"x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon"}
          </Math>
          <p>Where ε ~ N(0,I) and ᾱ_t = ∏_i α_i with α_t = 1 - β_t.</p>
        </FormulaCard>

        <FormulaCard>
          <h4>Training Objective</h4>
          <p>Simplified loss function for training the denoising network:</p>
          <Math block>
            {"L = \\mathbb{E}_{x_0, \\epsilon, t} \\left[ \\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2 \\right]"}
          </Math>
          <p>The model learns to predict the noise ε that was added at timestep t.</p>
        </FormulaCard>

        <FormulaCard>
          <h4>Classifier-Free Guidance</h4>
          <p>Interpolates between conditional and unconditional predictions:</p>
          <Math block>
            {"\\tilde{\\epsilon}_\\theta(x_t, t, c) = \\epsilon_\\theta(x_t, t, \\emptyset) + w \\cdot (\\epsilon_\\theta(x_t, t, c) - \\epsilon_\\theta(x_t, t, \\emptyset))"}
          </Math>
          <p>Where w is the guidance scale that controls conditioning strength.</p>
        </FormulaCard>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Diffusion Process Visualization</SectionTitle>
        <ProcessDiagram>
          <h4>From Noise to Image: The Reverse Process</h4>
          <div className="process-steps">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-title">Pure Noise</div>
              <div className="step-desc">Start with random Gaussian noise</div>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-title">Early Structure</div>
              <div className="step-desc">Rough shapes and forms emerge</div>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-title">Refined Details</div>
              <div className="step-desc">Features become more defined</div>
            </div>
            <div className="step">
              <div className="step-number">4</div>
              <div className="step-title">Fine Details</div>
              <div className="step-desc">Textures and details appear</div>
            </div>
            <div className="step">
              <div className="step-number">5</div>
              <div className="step-title">Final Image</div>
              <div className="step-desc">High-quality generated sample</div>
            </div>
          </div>
        </ProcessDiagram>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Advanced Techniques</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Latent Diffusion (Stable Diffusion):</strong> Perform diffusion in compressed latent space for efficiency</li>
          <li><CheckCircle size={16} /><strong>Conditioning Mechanisms:</strong> Text, class labels, images, or any modality can guide generation</li>
          <li><CheckCircle size={16} /><strong>Noise Schedules:</strong> Linear, cosine, and learned schedules for optimal training dynamics</li>
          <li><CheckCircle size={16} /><strong>Accelerated Sampling:</strong> DDIM, DPM-Solver for faster inference with fewer steps</li>
          <li><CheckCircle size={16} /><strong>Inpainting & Editing:</strong> Modify specific regions while preserving context</li>
          <li><CheckCircle size={16} /><strong>Score-Based SDEs:</strong> Continuous-time formulation with stochastic differential equations</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Model Architectures</SectionTitle>
        <ConceptGrid>
          <Card>
            <h4>U-Net Architecture</h4>
            <p>
              Encoder-decoder with skip connections, adapted for diffusion with timestep 
              and conditioning embeddings. ResNet blocks with attention layers.
            </p>
          </Card>
          
          <Card variant="accent">
            <h4>Attention Mechanisms</h4>
            <p>
              Self-attention for spatial consistency, cross-attention for conditioning 
              on text or other modalities. Multi-scale attention at different resolutions.
            </p>
          </Card>
          
          <Card variant="success">
            <h4>Timestep Embedding</h4>
            <p>
              Sinusoidal position embeddings for timestep t, enabling the model to know 
              how much noise is present at each step.
            </p>
          </Card>
        </ConceptGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Interview Focus Areas</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Training Efficiency:</strong> How to scale diffusion model training to large datasets and high resolutions</li>
          <li><CheckCircle size={16} /><strong>Inference Speed:</strong> Trade-offs between sample quality and generation speed</li>
          <li><CheckCircle size={16} /><strong>Evaluation Metrics:</strong> FID, IS, CLIP Score, human evaluation for assessing generation quality</li>
          <li><CheckCircle size={16} /><strong>Conditioning Strategies:</strong> Different ways to incorporate guidance signals effectively</li>
          <li><CheckCircle size={16} /><strong>Safety Considerations:</strong> Content filtering, bias mitigation, preventing harmful generations</li>
          <li><CheckCircle size={16} /><strong>Production Deployment:</strong> Optimizing for real-world applications and user experience</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Test Your Knowledge</SectionTitle>
        {!showQuiz ? (
          <Card>
            <h4><Book />Ready for the Quiz?</h4>
            <p>Test your understanding of diffusion models with these challenging questions.</p>
            <button
              onClick={() => setShowQuiz(true)}
              style={{
                background: '#ff6b35',
                color: 'white',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '600',
                cursor: 'pointer',
                marginTop: '1rem'
              }}
            >
              Start Quiz
            </button>
          </Card>
        ) : (
          <Quiz 
            questions={diffusionQuestions}
          />
        )}
      </ContentSection>
    </motion.div>
  );
};

export default DiffusionPage;
