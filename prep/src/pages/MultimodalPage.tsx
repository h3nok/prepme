import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowLeft, Eye, MessageSquare, Zap, CheckCircle, Book } from 'lucide-react';
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

const ArchitectureDiagram = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.lg} 0;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.lg};
    text-align: center;
  }

  .fusion-stages {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: ${props => props.theme.spacing.lg};
    margin-top: ${props => props.theme.spacing.lg};
  }

  .stage {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.lg};
    border-radius: ${props => props.theme.radii.md};
    border: 1px solid ${props => props.theme.colors.border};
    text-align: center;

    .stage-title {
      font-weight: 600;
      color: ${props => props.theme.colors.primary};
      margin-bottom: ${props => props.theme.spacing.md};
    }

    .modality {
      background: ${props => props.theme.colors.primary}20;
      padding: ${props => props.theme.spacing.sm};
      margin: ${props => props.theme.spacing.xs} 0;
      border-radius: ${props => props.theme.radii.sm};
      font-size: 0.9rem;
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

const ModelTimeline = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.lg} 0;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.lg};
    text-align: center;
  }

  .timeline {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: ${props => props.theme.spacing.md};
    margin-top: ${props => props.theme.spacing.lg};
  }

  .model {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.lg};
    border-radius: ${props => props.theme.radii.md};
    border: 1px solid ${props => props.theme.colors.border};

    .model-name {
      font-weight: 600;
      color: ${props => props.theme.colors.primary};
      margin-bottom: ${props => props.theme.spacing.sm};
    }

    .model-desc {
      font-size: 0.9rem;
      color: ${props => props.theme.colors.textSecondary};
      line-height: 1.4;
    }

    .model-year {
      background: ${props => props.theme.colors.primary};
      color: white;
      padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
      border-radius: ${props => props.theme.radii.sm};
      font-size: 0.8rem;
      font-weight: 600;
      display: inline-block;
      margin-bottom: ${props => props.theme.spacing.sm};
    }
  }
`;

const multimodalQuestions = [
  {
    id: "multimodal-1",
    question: "What is the key innovation of CLIP (Contrastive Language-Image Pre-training)?",
    options: [
      "Joint training on image-text pairs using contrastive learning",
      "Using transformers for image classification",
      "Generating images from text descriptions",
      "Cross-modal attention between vision and language"
    ],
    correct: 0,
    explanation: "CLIP learns joint representations by contrasting positive image-text pairs against negative pairs, enabling zero-shot transfer to many vision tasks without task-specific training data.",
    difficulty: "medium" as const
  },
  {
    id: "multimodal-2",
    question: "In vision-language models, what is the purpose of cross-modal attention?",
    options: [
      "To reduce computational complexity",
      "To enable interaction between visual and textual features",
      "To compress the model size",
      "To speed up training"
    ],
    correct: 1,
    explanation: "Cross-modal attention allows different modalities (vision and language) to attend to each other, enabling the model to understand relationships between visual elements and text.",
    difficulty: "medium" as const
  },
  {
    id: "multimodal-3",
    question: "What challenge does the 'modality gap' present in multimodal models?",
    options: [
      "Different modalities have different update frequencies",
      "Features from different modalities lie in different regions of embedding space",
      "Some modalities require more compute than others",
      "Different modalities use different architectures"
    ],
    correct: 1,
    explanation: "The modality gap refers to the tendency for embeddings from different modalities to cluster in separate regions of the joint embedding space, making cross-modal retrieval and reasoning more challenging.",
    difficulty: "hard" as const
  }
];

const MultimodalPage: React.FC = () => {
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
          👁️ <span>Multimodal</span> AI
        </PageTitle>
        <PageDescription>
          Discover how AI systems understand and connect different types of data. From CLIP to 
          GPT-4V, explore the techniques that enable AI to process vision, language, audio, 
          and other modalities together.
        </PageDescription>
      </PageHeader>

      <ContentSection>
        <SectionTitle>Core Concepts</SectionTitle>
        <ConceptGrid>
          <Card>
            <h4><Eye />Vision-Language Models</h4>
            <p>
              Systems that understand both images and text, enabling tasks like image 
              captioning, visual question answering, and cross-modal retrieval.
            </p>
          </Card>
          
          <Card variant="accent">
            <h4><MessageSquare />Cross-Modal Attention</h4>
            <p>
              Attention mechanisms that allow different modalities to interact, 
              enabling rich understanding of relationships across data types.
            </p>
          </Card>
          
          <Card variant="purple">
            <h4><Zap />Contrastive Learning</h4>
            <p>
              Training approach that learns to align similar examples and separate 
              dissimilar ones across modalities, as seen in CLIP.
            </p>
          </Card>
        </ConceptGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Mathematical Foundation</SectionTitle>
        
        <FormulaCard>
          <h4>Contrastive Loss (CLIP)</h4>
          <p>Learns joint embeddings by maximizing similarity of positive pairs:</p>
          <Math block>
            {"\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(\\text{sim}(I_i, T_i) / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(I_i, T_j) / \\tau)}"}
          </Math>
          <p>Where sim(I,T) is cosine similarity between image and text embeddings, τ is temperature.</p>
        </FormulaCard>

        <FormulaCard>
          <h4>Cross-Modal Attention</h4>
          <p>Attention mechanism between different modalities:</p>
          <Math block>
            {"\\text{Attention}(Q_v, K_t, V_t) = \\text{softmax}\\left(\\frac{Q_v K_t^T}{\\sqrt{d_k}}\\right) V_t"}
          </Math>
          <p>Where Q_v are visual queries and K_t, V_t are textual keys and values.</p>
        </FormulaCard>

        <FormulaCard>
          <h4>Multimodal Fusion</h4>
          <p>Simple concatenation or more sophisticated fusion strategies:</p>
          <Math block>
            {"h_{\\text{fused}} = f(h_v \\oplus h_t \\oplus h_a)"}
          </Math>
          <p>Where ⊕ represents fusion operation (concat, sum, attention) across vision, text, audio.</p>
        </FormulaCard>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Fusion Strategies</SectionTitle>
        <ArchitectureDiagram>
          <h4>Multimodal Fusion Approaches</h4>
          <div className="fusion-stages">
            <div className="stage">
              <div className="stage-title">Early Fusion</div>
              <div className="modality">Raw Features</div>
              <div className="modality">Concatenation</div>
              <div className="modality">Joint Processing</div>
            </div>
            <div className="stage">
              <div className="stage-title">Late Fusion</div>
              <div className="modality">Separate Processing</div>
              <div className="modality">Modal Embeddings</div>
              <div className="modality">Final Combination</div>
            </div>
            <div className="stage">
              <div className="stage-title">Hybrid Fusion</div>
              <div className="modality">Multi-level Interaction</div>
              <div className="modality">Cross-attention</div>
              <div className="modality">Adaptive Fusion</div>
            </div>
          </div>
        </ArchitectureDiagram>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Model Evolution</SectionTitle>
        <ModelTimeline>
          <h4>Key Multimodal Models Timeline</h4>
          <div className="timeline">
            <div className="model">
              <div className="model-year">2021</div>
              <div className="model-name">CLIP</div>
              <div className="model-desc">
                Contrastive pre-training on 400M image-text pairs, enabling zero-shot image classification.
              </div>
            </div>
            <div className="model">
              <div className="model-year">2021</div>
              <div className="model-name">DALL-E</div>
              <div className="model-desc">
                Text-to-image generation using autoregressive transformers and VQ-VAE.
              </div>
            </div>
            <div className="model">
              <div className="model-year">2022</div>
              <div className="model-name">Flamingo</div>
              <div className="model-desc">
                Few-shot learning across vision and language tasks with frozen language model.
              </div>
            </div>
            <div className="model">
              <div className="model-year">2022</div>
              <div className="model-name">DALL-E 2</div>
              <div className="model-desc">
                High-quality text-to-image using CLIP embeddings and diffusion models.
              </div>
            </div>
            <div className="model">
              <div className="model-year">2023</div>
              <div className="model-name">GPT-4V</div>
              <div className="model-desc">
                Multimodal ChatGPT that can understand and reason about images.
              </div>
            </div>
            <div className="model">
              <div className="model-year">2024</div>
              <div className="model-name">Gemini Ultra</div>
              <div className="model-desc">
                Native multimodal model trained on text, images, audio, and video.
              </div>
            </div>
          </div>
        </ModelTimeline>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Applications & Tasks</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Image Captioning:</strong> Generate natural language descriptions of visual content</li>
          <li><CheckCircle size={16} /><strong>Visual Question Answering:</strong> Answer questions about image content using natural language</li>
          <li><CheckCircle size={16} /><strong>Text-to-Image Generation:</strong> Create images from textual descriptions (DALL-E, Midjourney)</li>
          <li><CheckCircle size={16} /><strong>Cross-Modal Retrieval:</strong> Find images using text queries or text using image queries</li>
          <li><CheckCircle size={16} /><strong>Video Understanding:</strong> Analyze temporal visual content with language grounding</li>
          <li><CheckCircle size={16} /><strong>Audio-Visual Learning:</strong> Connect speech, sound, and visual information</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Technical Challenges</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Modality Gap:</strong> Different modalities cluster in separate embedding regions</li>
          <li><CheckCircle size={16} /><strong>Alignment Quality:</strong> Ensuring semantic correspondence across modalities</li>
          <li><CheckCircle size={16} /><strong>Computational Efficiency:</strong> Processing multiple high-dimensional modalities</li>
          <li><CheckCircle size={16} /><strong>Data Quality:</strong> Obtaining high-quality aligned multimodal datasets</li>
          <li><CheckCircle size={16} /><strong>Evaluation Metrics:</strong> Measuring cross-modal understanding and generation quality</li>
          <li><CheckCircle size={16} /><strong>Bias and Fairness:</strong> Ensuring equitable representation across modalities</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Interview Focus Areas</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Architecture Design:</strong> How to effectively combine different encoders and fusion strategies</li>
          <li><CheckCircle size={16} /><strong>Training Strategies:</strong> Curriculum learning, progressive training, and modality-specific losses</li>
          <li><CheckCircle size={16} /><strong>Evaluation Protocols:</strong> How to assess multimodal understanding beyond traditional metrics</li>
          <li><CheckCircle size={16} /><strong>Real-world Applications:</strong> Deploying multimodal systems for search, accessibility, creativity</li>
          <li><CheckCircle size={16} /><strong>Scaling Considerations:</strong> Challenges in scaling multimodal training to billions of parameters</li>
          <li><CheckCircle size={16} /><strong>Safety and Alignment:</strong> Ensuring multimodal models behave appropriately across all modalities</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Test Your Knowledge</SectionTitle>
        {!showQuiz ? (
          <Card>
            <h4><Book />Ready for the Quiz?</h4>
            <p>Test your understanding of multimodal AI with these challenging questions.</p>
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
            questions={multimodalQuestions}
          />
        )}
      </ContentSection>
    </motion.div>
  );
};

export default MultimodalPage;
