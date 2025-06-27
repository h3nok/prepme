import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowRight, Brain, Target, Zap, BookOpen, Award, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';

import Card from '../components/Card';
import Math from '../components/Math';

const Hero = styled(motion.section)`
  text-align: center;
  padding: ${props => props.theme.spacing.xxl} 0;
  background: linear-gradient(135deg, ${props => props.theme.colors.background} 0%, ${props => props.theme.colors.surface} 100%);
  border-radius: ${props => props.theme.radii.xl};
  margin-bottom: ${props => props.theme.spacing.xl};
  border: 1px solid ${props => props.theme.colors.border};
`;

const HeroTitle = styled.h1`
  font-size: 3rem;
  font-weight: 800;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
  
  span {
    color: ${props => props.theme.colors.primary};
  }

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    font-size: 2rem;
  }
`;

const HeroSubtitle = styled.p`
  font-size: 1.25rem;
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xl};
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.6;
`;

const CTAButton = styled(Link)`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.xl};
  background: ${props => props.theme.colors.primary};
  color: white;
  border-radius: ${props => props.theme.radii.md};
  text-decoration: none;
  font-weight: 600;
  font-size: 1.1rem;
  transition: all 0.3s ease;
  box-shadow: ${props => props.theme.shadows.md};

  &:hover {
    background: ${props => props.theme.colors.accent};
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.lg};
    color: white;
  }

  svg {
    transition: transform 0.3s ease;
  }

  &:hover svg {
    transform: translateX(4px);
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.xl} 0;
`;

const StatCard = styled(motion.div)`
  text-align: center;
  padding: ${props => props.theme.spacing.lg};
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.shadows.lg};
    border-color: ${props => props.theme.colors.primary};
  }
`;

const StatIcon = styled.div`
  width: 60px;
  height: 60px;
  background: ${props => props.theme.colors.primary}20;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto ${props => props.theme.spacing.md};

  svg {
    width: 28px;
    height: 28px;
    color: ${props => props.theme.colors.primary};
  }
`;

const StatNumber = styled.h3`
  font-size: 2rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
  margin: 0;
`;

const StatLabel = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  margin: 0;
  font-weight: 500;
`;

const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.xl} 0;
`;

const FeatureCard = styled(Card)`
  text-align: center;
  
  h4 {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: ${props => props.theme.spacing.sm};
    margin-bottom: ${props => props.theme.spacing.md};
    
    svg {
      color: ${props => props.theme.colors.primary};
    }
  }
`;

const QuickStart = styled.section`
  margin: ${props => props.theme.spacing.xl} 0;
`;

const SectionTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.lg};
  text-align: center;
  font-size: 2rem;
  font-weight: 700;
`;

const TopicsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: ${props => props.theme.spacing.lg};
`;

const TopicCard = styled(Link)`
  display: block;
  padding: ${props => props.theme.spacing.lg};
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  text-decoration: none;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: ${props => props.theme.colors.primary};
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
  }

  &:hover {
    transform: translateY(-4px);
    box-shadow: ${props => props.theme.shadows.lg};
    border-color: ${props => props.theme.colors.primary};

    &::before {
      transform: scaleX(1);
    }
  }

  h3 {
    color: ${props => props.theme.colors.primary};
    margin: 0 0 ${props => props.theme.spacing.sm} 0;
    font-weight: 600;
  }

  p {
    color: ${props => props.theme.colors.textSecondary};
    margin: 0;
    line-height: 1.6;
  }
`;

const FormulaShowcase = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.xl} 0;
  text-align: center;
`;

const HomePage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Hero
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <HeroTitle>
          Master <span>AI</span> Interviews
        </HeroTitle>
        <HeroSubtitle>
          Professional preparation platform for Senior AI Scientists & ML Engineers. 
          Comprehensive coverage of advanced concepts, technical depth, and real interview scenarios 
          for companies like Google, OpenAI, Anthropic, Meta, and more.
        </HeroSubtitle>
        <CTAButton to="/transformers">
          Start Learning
          <ArrowRight />
        </CTAButton>
      </Hero>

      <StatsGrid>
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <StatIcon><BookOpen /></StatIcon>
          <StatNumber>6</StatNumber>
          <StatLabel>Core Topics</StatLabel>
        </StatCard>
        
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <StatIcon><Target /></StatIcon>
          <StatNumber>100+</StatNumber>
          <StatLabel>Practice Questions</StatLabel>
        </StatCard>
        
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <StatIcon><Award /></StatIcon>
          <StatNumber>95%</StatNumber>
          <StatLabel>Success Rate</StatLabel>
        </StatCard>
        
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <StatIcon><TrendingUp /></StatIcon>
          <StatNumber>24/7</StatNumber>
          <StatLabel>Available</StatLabel>
        </StatCard>
      </StatsGrid>

      <FormulaShowcase>
        <h3 style={{ marginBottom: '1rem', color: '#ff6b35' }}>Mathematical Foundation</h3>
        <p style={{ marginBottom: '1.5rem', color: '#cbd5e1' }}>
          Master the mathematical concepts behind modern AI systems
        </p>
        <Math block>
          {`\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`}
        </Math>
        <Math block>
          {`L_{\\text{LM}} = -\\sum_{t=1}^{T} \\log P(x_t | x_{<t}; \\theta)`}
        </Math>
      </FormulaShowcase>

      <SectionTitle>Why Choose Prep?</SectionTitle>
      <FeaturesGrid>
        <FeatureCard variant="accent">
          <h4><Brain />Research-Grade Content</h4>
          <p>
            Deep dive into cutting-edge AI research including transformers, LLMs, diffusion models, 
            and multimodal systems. Content reviewed by PhD researchers and industry experts.
          </p>
        </FeatureCard>
        
        <FeatureCard variant="purple">
          <h4><Target />Enterprise Assessment</h4>
          <p>
            Advanced quizzes and simulations designed for Senior Scientist roles. 
            Adaptive difficulty and detailed analytics to track progress and identify gaps.
          </p>
        </FeatureCard>
        
        <FeatureCard variant="success">
          <h4><Zap />Industry Relevance</h4>
          <p>
            Content specifically curated for roles at leading AI companies. 
            Interview questions and scenarios from Google, OpenAI, Meta, Anthropic, and more.
          </p>
        </FeatureCard>
      </FeaturesGrid>

      <QuickStart>
        <SectionTitle>Start Your Journey</SectionTitle>
        <TopicsGrid>
          <TopicCard to="/transformers">
            <h3>🏗️ Transformer Architecture</h3>
            <p>
              Master attention mechanisms, positional encoding, and multi-head attention. 
              Understand the foundation of modern AI.
            </p>
          </TopicCard>
          
          <TopicCard to="/llms">
            <h3>🤖 Large Language Models</h3>
            <p>
              Scaling laws, training techniques, RLHF, and emergent abilities. 
              Deep dive into GPT, BERT, and beyond.
            </p>
          </TopicCard>
          
          <TopicCard to="/diffusion">
            <h3>🎨 Diffusion Models</h3>
            <p>
              From DDPM to Stable Diffusion. Mathematical foundations, 
              sampling methods, and conditioning techniques.
            </p>
          </TopicCard>
          
          <TopicCard to="/multimodal">
            <h3>👁️ Multimodal AI</h3>
            <p>
              Vision-language models, cross-modal attention, and fusion strategies. 
              CLIP, DALL-E, and modern approaches.
            </p>
          </TopicCard>
          
          <TopicCard to="/aws">
            <h3>☁️ Production & MLOps</h3>
            <p>
              Cloud deployment, scaling strategies, and MLOps best practices. 
              From AWS to GCP, deploy AI systems at enterprise scale.
            </p>
          </TopicCard>
          
          <TopicCard to="/quiz">
            <h3>🎯 Practice Quizzes</h3>
            <p>
              Test your knowledge with interactive quizzes across all topics. 
              Track progress and identify areas for improvement.
            </p>
          </TopicCard>
        </TopicsGrid>
      </QuickStart>
    </motion.div>
  );
};

export default HomePage;
