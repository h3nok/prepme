import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowLeft, Cpu, TrendingUp, Zap, CheckCircle, Book } from 'lucide-react';
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

const ScalingChart = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.lg} 0;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.md};
    text-align: center;
  }

  .model-evolution {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: ${props => props.theme.spacing.md};
    margin-top: ${props => props.theme.spacing.lg};
  }

  .model-card {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.md};
    border-radius: ${props => props.theme.radii.md};
    border: 1px solid ${props => props.theme.colors.border};
    text-align: center;

    .model-name {
      font-weight: 600;
      color: ${props => props.theme.colors.primary};
      font-size: 0.9rem;
    }

    .model-params {
      font-size: 0.8rem;
      color: ${props => props.theme.colors.textSecondary};
    }
  }
`;

const llmQuestions = [
  {
    id: "llm-1",
    question: "What is the key insight behind scaling laws in LLMs?",
    options: [
      "Larger models always perform better",
      "Performance scales predictably with compute, data, and parameters",
      "More data is always better than more parameters",
      "Scaling only works for certain architectures"
    ],
    correct: 1,
    explanation: "Scaling laws show that model performance follows predictable power-law relationships with compute budget, dataset size, and model parameters when not bottlenecked by other factors.",
    difficulty: "medium" as const
  },
  {
    id: "llm-2",
    question: "What is the primary goal of RLHF (Reinforcement Learning from Human Feedback)?",
    options: [
      "To make models faster at inference",
      "To align model outputs with human preferences and values",
      "To reduce model parameters",
      "To improve mathematical reasoning"
    ],
    correct: 1,
    explanation: "RLHF trains a reward model from human feedback and uses reinforcement learning to optimize the language model to generate outputs that score highly according to human preferences.",
    difficulty: "medium" as const
  },
  {
    id: "llm-3",
    question: "What are 'emergent abilities' in large language models?",
    options: [
      "Abilities that appear only after certain scale thresholds",
      "Abilities that are explicitly programmed",
      "Abilities that decrease with model size",
      "Abilities that only work in specific domains"
    ],
    correct: 0,
    explanation: "Emergent abilities are capabilities that are not present in smaller models but suddenly appear when models reach certain scale thresholds, such as few-shot reasoning or following complex instructions.",
    difficulty: "hard" as const
  }
];

const LLMsPage: React.FC = () => {
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
          🤖 Large <span>Language</span> Models
        </PageTitle>
        <PageDescription>
          Explore the frontiers of large language models. From scaling laws to emergent abilities, 
          understand how massive neural networks learn to understand and generate human language 
          with unprecedented sophistication.
        </PageDescription>
      </PageHeader>

      <ContentSection>
        <SectionTitle>Core Concepts</SectionTitle>
        <ConceptGrid>
          <Card>
            <h4><TrendingUp />Scaling Laws</h4>
            <p>
              Mathematical relationships governing how model performance improves with 
              increased compute, parameters, and data. Critical for resource allocation 
              and predicting future capabilities.
            </p>
          </Card>
          
          <Card variant="accent">
            <h4><Zap />Emergent Abilities</h4>
            <p>
              Capabilities that suddenly appear at certain scale thresholds, like 
              few-shot learning, chain-of-thought reasoning, and following complex instructions.
            </p>
          </Card>
          
          <Card variant="purple">
            <h4><Cpu />Training Dynamics</h4>
            <p>
              From pre-training on massive text corpora to fine-tuning with human feedback. 
              Understanding the multi-stage process that creates capable language models.
            </p>
          </Card>
        </ConceptGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Scaling Laws & Model Evolution</SectionTitle>
        
        <FormulaCard>
          <h4>Chinchilla Scaling Laws</h4>
          <p>Optimal relationship between model parameters (N) and training tokens (D):</p>
          <Math block>
            {"N_{\\text{optimal}} \\propto C^{0.5}, \\quad D_{\\text{optimal}} \\propto C^{0.5}"}
          </Math>
          <p>
            Where C is the compute budget. This suggests that for optimal performance, 
            model size and training data should scale equally with available compute.
          </p>
        </FormulaCard>

        <ScalingChart>
          <h4>LLM Evolution Timeline</h4>
          <div className="model-evolution">
            <div className="model-card">
              <div className="model-name">GPT-1</div>
              <div className="model-params">117M parameters</div>
            </div>
            <div className="model-card">
              <div className="model-name">BERT-Large</div>
              <div className="model-params">340M parameters</div>
            </div>
            <div className="model-card">
              <div className="model-name">GPT-2</div>
              <div className="model-params">1.5B parameters</div>
            </div>
            <div className="model-card">
              <div className="model-name">T5-11B</div>
              <div className="model-params">11B parameters</div>
            </div>
            <div className="model-card">
              <div className="model-name">GPT-3</div>
              <div className="model-params">175B parameters</div>
            </div>
            <div className="model-card">
              <div className="model-name">PaLM</div>
              <div className="model-params">540B parameters</div>
            </div>
            <div className="model-card">
              <div className="model-name">GPT-4</div>
              <div className="model-params">~1.7T parameters</div>
            </div>
          </div>
        </ScalingChart>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Training Methodology</SectionTitle>
        
        <FormulaCard>
          <h4>Language Modeling Objective</h4>
          <p>The core pre-training objective maximizes the likelihood of the next token:</p>
          <Math block>
            {"L_{\\text{LM}} = -\\sum_{t=1}^{T} \\log P(x_t | x_{<t}; \\theta)"}
          </Math>
          <p>
            This simple objective, when applied at scale, leads to the emergence of 
            complex linguistic and reasoning capabilities.
          </p>
        </FormulaCard>

        <FormulaCard>
          <h4>RLHF Reward Optimization</h4>
          <p>Reinforcement learning from human feedback optimizes for human preferences:</p>
          <Math block>
            {"\\max_{\\pi} \\mathbb{E}_{x \\sim D, y \\sim \\pi(\\cdot|x)} [R(x,y)] - \\beta \\cdot \\text{KL}[\\pi(\\cdot|x) || \\pi_{\\text{ref}}(\\cdot|x)]"}
          </Math>
          <p>
            Where R(x,y) is the learned reward model and the KL term prevents the model 
            from deviating too far from the reference policy.
          </p>
        </FormulaCard>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Advanced Techniques</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Instruction Tuning:</strong> Fine-tuning on diverse instruction-following tasks to improve zero-shot generalization</li>
          <li><CheckCircle size={16} /><strong>Chain-of-Thought:</strong> Prompting models to show intermediate reasoning steps improves complex reasoning</li>
          <li><CheckCircle size={16} /><strong>Constitutional AI:</strong> Training models to critique and revise their own outputs according to principles</li>
          <li><CheckCircle size={16} /><strong>In-Context Learning:</strong> Few-shot learning through examples in the prompt without parameter updates</li>
          <li><CheckCircle size={16} /><strong>Tool Use:</strong> Enabling models to interact with external tools and APIs to extend capabilities</li>
          <li><CheckCircle size={16} /><strong>Mixture of Experts:</strong> Scaling model capacity efficiently by using only relevant sub-networks</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Interview Focus Areas</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Scaling Strategies:</strong> How to optimally allocate compute between model size, data, and training time</li>
          <li><CheckCircle size={16} /><strong>Alignment Challenges:</strong> Technical approaches to ensuring AI systems behave as intended</li>
          <li><CheckCircle size={16} /><strong>Evaluation Metrics:</strong> How to measure and compare LLM capabilities across different tasks</li>
          <li><CheckCircle size={16} /><strong>Compute Optimization:</strong> Techniques for efficient training and inference at scale</li>
          <li><CheckCircle size={16} /><strong>Data Quality:</strong> Impact of training data composition on model capabilities and biases</li>
          <li><CheckCircle size={16} /><strong>Safety Considerations:</strong> Technical approaches to mitigating potential risks from powerful AI systems</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Test Your Knowledge</SectionTitle>
        {!showQuiz ? (
          <Card>
            <h4><Book />Ready for the Quiz?</h4>
            <p>Test your understanding of large language models with these challenging questions.</p>
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
            questions={llmQuestions}
          />
        )}
      </ContentSection>
    </motion.div>
  );
};

export default LLMsPage;
