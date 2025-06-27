import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowLeft, Target, Brain, Zap, Award, RefreshCw } from 'lucide-react';
import { Link } from 'react-router-dom';

import Card from '../components/Card';
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

const QuizGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const QuizCard = styled(Card)`
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.shadows.lg};
    border-color: ${props => props.theme.colors.primary};
  }
  
  .quiz-header {
    display: flex;
    align-items: center;
    gap: ${props => props.theme.spacing.sm};
    margin-bottom: ${props => props.theme.spacing.md};
    
    .quiz-icon {
      width: 40px;
      height: 40px;
      background: ${props => props.theme.colors.primary}20;
      border-radius: ${props => props.theme.radii.md};
      display: flex;
      align-items: center;
      justify-content: center;
      
      svg {
        color: ${props => props.theme.colors.primary};
      }
    }
    
    .quiz-info {
      .quiz-title {
        font-weight: 600;
        color: ${props => props.theme.colors.primary};
        font-size: 1.1rem;
        margin: 0;
      }
      
      .quiz-meta {
        font-size: 0.9rem;
        color: ${props => props.theme.colors.textSecondary};
        display: flex;
        gap: ${props => props.theme.spacing.md};
      }
    }
  }
  
  .difficulty-badge {
    display: inline-block;
    padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
    border-radius: ${props => props.theme.radii.sm};
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: ${props => props.theme.spacing.sm};
    
    &.easy {
      background: ${props => props.theme.colors.success}20;
      color: ${props => props.theme.colors.success};
    }
    
    &.medium {
      background: ${props => props.theme.colors.warning}20;
      color: ${props => props.theme.colors.warning};
    }
    
    &.hard {
      background: ${props => props.theme.colors.error}20;
      color: ${props => props.theme.colors.error};
    }
  }
`;

const QuizSelector = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
  
  button {
    padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
    border: 1px solid ${props => props.theme.colors.border};
    background: ${props => props.theme.colors.surface};
    color: ${props => props.theme.colors.text};
    border-radius: ${props => props.theme.radii.md};
    cursor: pointer;
    transition: all 0.2s ease;
    
    &:hover {
      border-color: ${props => props.theme.colors.primary};
      background: ${props => props.theme.colors.primary}10;
    }
    
    &.active {
      background: ${props => props.theme.colors.primary};
      color: white;
      border-color: ${props => props.theme.colors.primary};
    }
  }
`;

// Combined quiz questions from all topics
const allQuestions = [
  // Transformer questions
  {
    id: "transformer-1",
    question: "What is the main innovation of the Transformer architecture?",
    options: [
      "Convolutional layers for sequence processing",
      "Self-attention mechanism for parallel processing",
      "Recurrent connections for memory",
      "Pooling layers for dimensionality reduction"
    ],
    correct: 1,
    explanation: "The Transformer's key innovation is the self-attention mechanism that allows parallel processing of sequences, eliminating the sequential bottleneck of RNNs.",
    difficulty: "medium" as const,
    topic: "Transformers"
  },
  {
    id: "transformer-2",
    question: "In the attention formula, what does the √d_k normalization factor prevent?",
    options: [
      "Overfitting during training",
      "Gradient vanishing problems",
      "Softmax saturation in high dimensions",
      "Memory overflow issues"
    ],
    correct: 2,
    explanation: "The √d_k factor prevents the dot products from becoming too large in high dimensions, which would cause the softmax to saturate and produce very small gradients.",
    difficulty: "hard" as const,
    topic: "Transformers"
  },
  
  // LLM questions
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
    difficulty: "medium" as const,
    topic: "LLMs"
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
    difficulty: "medium" as const,
    topic: "LLMs"
  },
  
  // Diffusion questions
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
    difficulty: "medium" as const,
    topic: "Diffusion"
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
    difficulty: "hard" as const,
    topic: "Diffusion"
  },
  
  // Multimodal questions
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
    difficulty: "medium" as const,
    topic: "Multimodal"
  },
  
  // AWS questions
  {
    id: "aws-1",
    question: "Which AWS service is best for training large language models at scale?",
    options: [
      "EC2 with custom setup",
      "SageMaker Training Jobs with distributed training",
      "Lambda functions",
      "ECS containers"
    ],
    correct: 1,
    explanation: "SageMaker Training Jobs provide managed infrastructure for distributed training with automatic scaling, model parallelism, and optimized ML instances like p4d.24xlarge.",
    difficulty: "medium" as const,
    topic: "AWS"
  },
  {
    id: "aws-2",
    question: "What is the key advantage of using Amazon Bedrock for LLM applications?",
    options: [
      "Cheaper than training custom models",
      "Access to foundation models via API without managing infrastructure",
      "Better performance than custom models",
      "Automatic model fine-tuning"
    ],
    correct: 1,
    explanation: "Bedrock provides serverless access to foundation models from companies like Anthropic, Cohere, and Stability AI without needing to provision or manage infrastructure.",
    difficulty: "easy" as const,
    topic: "AWS"
  }
];

interface QuizPageProps {}

const QuizPage: React.FC<QuizPageProps> = () => {
  const [selectedTopic, setSelectedTopic] = useState<string>('all');
  const [activeQuiz, setActiveQuiz] = useState<string | null>(null);

  const topics = ['all', 'Transformers', 'LLMs', 'Diffusion', 'Multimodal', 'AWS'];
  
  const getFilteredQuestions = () => {
    if (selectedTopic === 'all') return allQuestions;
    return allQuestions.filter(q => q.topic === selectedTopic);
  };

  const getTopicStats = (topic: string) => {
    const topicQuestions = topic === 'all' 
      ? allQuestions 
      : allQuestions.filter(q => q.topic === topic);
    
    const difficulties = topicQuestions.reduce((acc, q) => {
      acc[q.difficulty] = (acc[q.difficulty] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return {
      total: topicQuestions.length,
      difficulties
    };
  };

  if (activeQuiz) {
    const quizQuestions = getFilteredQuestions();
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
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <PageTitle>
              🎯 <span>{selectedTopic}</span> Quiz
            </PageTitle>
            <button
              onClick={() => setActiveQuiz(null)}
              style={{
                background: 'transparent',
                border: `1px solid #ff6b35`,
                color: '#ff6b35',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.9rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              <RefreshCw size={16} />
              Change Quiz
            </button>
          </div>
        </PageHeader>
        
        <Quiz questions={quizQuestions} title={`${selectedTopic} Quiz`} />
      </motion.div>
    );
  }

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
          🎯 Practice <span>Quizzes</span>
        </PageTitle>
        <PageDescription>
          Test your knowledge across all AI topics. Choose from topic-specific quizzes 
          or take the comprehensive assessment covering transformers, LLMs, diffusion 
          models, multimodal AI, and AWS deployment.
        </PageDescription>
      </PageHeader>

      <ContentSection>
        <SectionTitle>Select Quiz Topic</SectionTitle>
        <QuizSelector>
          {topics.map(topic => (
            <button
              key={topic}
              className={selectedTopic === topic ? 'active' : ''}
              onClick={() => setSelectedTopic(topic)}
            >
              {topic === 'all' ? 'All Topics' : topic}
            </button>
          ))}
        </QuizSelector>
      </ContentSection>

      <ContentSection>
        <QuizGrid>
          <div onClick={() => setActiveQuiz(selectedTopic)}>
            <QuizCard>
              <div className="quiz-header">
                <div className="quiz-icon">
                  <Target size={20} />
                </div>
                <div className="quiz-info">
                  <h3 className="quiz-title">
                    {selectedTopic === 'all' ? 'Comprehensive Assessment' : `${selectedTopic} Quiz`}
                  </h3>
                  <div className="quiz-meta">
                    <span>{getTopicStats(selectedTopic).total} questions</span>
                    <span>Mixed difficulty</span>
                  </div>
                </div>
              </div>
              <p>
                {selectedTopic === 'all' 
                  ? 'Test your knowledge across all AI topics with our comprehensive quiz covering transformers, LLMs, diffusion models, multimodal AI, and AWS deployment strategies.'
                  : `Focus on ${selectedTopic.toLowerCase()} with targeted questions covering key concepts, mathematical foundations, and practical applications.`
                }
              </p>
              <div className="difficulty-stats">
                {Object.entries(getTopicStats(selectedTopic).difficulties).map(([diff, count]) => (
                  <span key={diff} className={`difficulty-badge ${diff}`}>
                    {count} {diff}
                  </span>
                ))}
              </div>
            </QuizCard>
          </div>

          <Card>
            <div className="quiz-header">
              <div className="quiz-icon">
                <Brain size={20} />
              </div>
              <div className="quiz-info">
                <h3 className="quiz-title">Quick Review</h3>
                <div className="quiz-meta">
                  <span>Key concepts</span>
                  <span>5 min</span>
                </div>
              </div>
            </div>
            <p>
              Quick review of essential concepts and formulas. Perfect for last-minute 
              preparation before interviews or as a warm-up exercise.
            </p>
            <span className="difficulty-badge easy">Easy</span>
          </Card>

          <Card>
            <div className="quiz-header">
              <div className="quiz-icon">
                <Zap size={20} />
              </div>
              <div className="quiz-info">
                <h3 className="quiz-title">Lightning Round</h3>
                <div className="quiz-meta">
                  <span>Rapid fire</span>
                  <span>3 min</span>
                </div>
              </div>
            </div>
            <p>
              Fast-paced quiz with quick questions to test your instant recall of 
              key facts, definitions, and concepts across all topics.
            </p>
            <span className="difficulty-badge medium">Medium</span>
          </Card>

          <Card>
            <div className="quiz-header">
              <div className="quiz-icon">
                <Award size={20} />
              </div>
              <div className="quiz-info">
                <h3 className="quiz-title">Expert Challenge</h3>
                <div className="quiz-meta">
                  <span>Advanced</span>
                  <span>15 min</span>
                </div>
              </div>
            </div>
            <p>
              Advanced questions covering cutting-edge research, implementation details, 
              and complex scenarios you might encounter in senior-level interviews.
            </p>
            <span className="difficulty-badge hard">Hard</span>
          </Card>
        </QuizGrid>
      </ContentSection>
    </motion.div>
  );
};

export default QuizPage;
