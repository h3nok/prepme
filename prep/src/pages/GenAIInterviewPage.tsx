import React, { useState, useContext } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, BookOpen, Zap, CheckCircle, Clock, Target,
  ArrowRight, ArrowLeft, Star, Award, TrendingUp, AlertCircle
} from 'lucide-react';

import { genAIInterviewModule } from '../data/GenAIInterviewModule';
import { genAIInterviewQuiz } from '../data/GenAIInterviewQuiz';
import LearningInterface from '../components/LearningInterface';
import Quiz from '../components/Quiz';
import { useTheme } from '../context/ThemeContext';

const PageContainer = styled.div<{ isDark: boolean }>`
  min-height: 100vh;
  background: ${props => props.isDark 
    ? 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%)'
    : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%)'
  };
  padding: 2rem;
  transition: all 0.3s ease;
`;

const HeroSection = styled(motion.div)<{ isDark: boolean }>`
  text-align: center;
  margin-bottom: 3rem;
  padding: 3rem;
  background: ${props => props.isDark 
    ? 'rgba(139, 92, 246, 0.1)' 
    : 'rgba(139, 92, 246, 0.05)'
  };
  border-radius: 20px;
  border: 1px solid ${props => props.isDark ? 'rgba(139, 92, 246, 0.3)' : 'rgba(139, 92, 246, 0.2)'};
  backdrop-filter: blur(10px);
`;

const HeroTitle = styled.h1<{ isDark: boolean }>`
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #c084fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const HeroSubtitle = styled.p<{ isDark: boolean }>`
  font-size: 1.3rem;
  color: ${props => props.isDark ? '#e2e8f0' : '#475569'};
  max-width: 600px;
  margin: 0 auto 2rem;
  line-height: 1.6;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
`;

const StatCard = styled(motion.div)<{ isDark: boolean }>`
  background: ${props => props.isDark 
    ? 'rgba(30, 41, 59, 0.8)' 
    : 'rgba(255, 255, 255, 0.9)'
  };
  border-radius: 15px;
  padding: 1.5rem;
  text-align: center;
  border: 1px solid ${props => props.isDark ? 'rgba(71, 85, 105, 0.3)' : 'rgba(203, 213, 225, 0.5)'};
  backdrop-filter: blur(10px);
`;

const StatNumber = styled.div<{ isDark: boolean }>`
  font-size: 2.5rem;
  font-weight: 800;
  color: #8b5cf6;
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.div<{ isDark: boolean }>`
  font-size: 0.9rem;
  color: ${props => props.isDark ? '#94a3b8' : '#64748b'};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const ContentSection = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-bottom: 3rem;

  @media (max-width: 968px) {
    grid-template-columns: 1fr;
  }
`;

const FeatureCard = styled(motion.div)<{ isDark: boolean }>`
  background: ${props => props.isDark 
    ? 'rgba(30, 41, 59, 0.8)' 
    : 'rgba(255, 255, 255, 0.9)'
  };
  border-radius: 20px;
  padding: 2rem;
  border: 1px solid ${props => props.isDark ? 'rgba(71, 85, 105, 0.3)' : 'rgba(203, 213, 225, 0.5)'};
  backdrop-filter: blur(10px);
`;

const FeatureTitle = styled.h3<{ isDark: boolean }>`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => props.isDark ? '#f1f5f9' : '#1e293b'};
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const FeatureDescription = styled.p<{ isDark: boolean }>`
  color: ${props => props.isDark ? '#cbd5e1' : '#475569'};
  line-height: 1.6;
  margin-bottom: 1.5rem;
`;

const FeatureList = styled.ul<{ isDark: boolean }>`
  list-style: none;
  padding: 0;
  
  li {
    color: ${props => props.isDark ? '#e2e8f0' : '#334155'};
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    &:before {
      content: '✦';
      color: #8b5cf6;
      font-weight: bold;
    }
  }
`;

const ActionButton = styled(motion.button)<{ isDark: boolean }>`
  background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
  color: white;
  border: none;
  border-radius: 50px;
  padding: 1rem 2rem;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(139, 92, 246, 0.3);
  }
`;

const PreparationTips = styled(motion.div)<{ isDark: boolean }>`
  background: ${props => props.isDark 
    ? 'rgba(236, 72, 153, 0.1)' 
    : 'rgba(236, 72, 153, 0.05)'
  };
  border-radius: 15px;
  padding: 2rem;
  margin-bottom: 2rem;
  border: 1px solid ${props => props.isDark ? 'rgba(236, 72, 153, 0.3)' : 'rgba(236, 72, 153, 0.2)'};
`;

const TipTitle = styled.h3<{ isDark: boolean }>`
  color: #ec4899;
  font-size: 1.3rem;
  font-weight: 700;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const TipList = styled.div<{ isDark: boolean }>`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
`;

const TipItem = styled.div<{ isDark: boolean }>`
  color: ${props => props.isDark ? '#e2e8f0' : '#334155'};
  padding: 1rem;
  background: ${props => props.isDark ? 'rgba(30, 41, 59, 0.5)' : 'rgba(255, 255, 255, 0.5)'};
  border-radius: 10px;
  border-left: 4px solid #ec4899;
`;

export const GenAIInterviewPage: React.FC = () => {
  const [currentView, setCurrentView] = useState<'overview' | 'learning' | 'quiz'>('overview');
  const { isDark } = useTheme();

  const handleStartLearning = () => {
    setCurrentView('learning');
  };

  const handleStartQuiz = () => {
    setCurrentView('quiz');
  };

  const handleBackToOverview = () => {
    setCurrentView('overview');
  };

  if (currentView === 'learning') {
    // For now, redirect to a simple view. The LearningInterface can be enhanced later
    // to accept module props or we can create a dedicated GenAI learning component
    return (
      <PageContainer isDark={isDark}>
        <div style={{ textAlign: 'center', padding: '3rem' }}>
          <h2 style={{ color: isDark ? '#f1f5f9' : '#1e293b', marginBottom: '2rem' }}>
            GenAI Interview Learning Module
          </h2>
          <p style={{ color: isDark ? '#cbd5e1' : '#475569', marginBottom: '2rem' }}>
            Comprehensive learning content coming soon! This will include interactive slideshows,
            visualizations, and hands-on practice for all GenAI interview topics.
          </p>
          <ActionButton isDark={isDark} onClick={handleBackToOverview}>
            <ArrowLeft size={20} /> Back to Overview
          </ActionButton>
        </div>
      </PageContainer>
    );
  }

  if (currentView === 'quiz') {
    return (
      <PageContainer isDark={isDark}>
        <div style={{ textAlign: 'center', padding: '3rem' }}>
          <h2 style={{ color: isDark ? '#f1f5f9' : '#1e293b', marginBottom: '2rem' }}>
            GenAI Interview Quiz
          </h2>
          <p style={{ color: isDark ? '#cbd5e1' : '#475569', marginBottom: '2rem' }}>
            Interactive quiz with 25+ interview-style questions coming soon!
            Practice with questions that mirror real GenAI interviews.
          </p>
          <ActionButton isDark={isDark} onClick={handleBackToOverview}>
            <ArrowLeft size={20} /> Back to Overview
          </ActionButton>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer isDark={isDark}>
      <HeroSection
        isDark={isDark}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <HeroTitle isDark={isDark}>
          GenAI Interview Mastery
        </HeroTitle>
        <HeroSubtitle isDark={isDark}>
          Comprehensive preparation for Generative AI interviews. Master the concepts, architectures, 
          and techniques that interviewers ask about most frequently. Get ready to ace your GenAI interview!
        </HeroSubtitle>
        
        <StatsGrid>
          <StatCard isDark={isDark}
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <StatNumber isDark={isDark}>20+</StatNumber>
            <StatLabel isDark={isDark}>Hours of Content</StatLabel>
          </StatCard>
          <StatCard isDark={isDark}
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <StatNumber isDark={isDark}>25+</StatNumber>
            <StatLabel isDark={isDark}>Interview Questions</StatLabel>
          </StatCard>
          <StatCard isDark={isDark}
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <StatNumber isDark={isDark}>50+</StatNumber>
            <StatLabel isDark={isDark}>Key Concepts</StatLabel>
          </StatCard>
          <StatCard isDark={isDark}
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <StatNumber isDark={isDark}>100+</StatNumber>
            <StatLabel isDark={isDark}>Practice Questions</StatLabel>
          </StatCard>
        </StatsGrid>
      </HeroSection>

      <PreparationTips
        isDark={isDark}
        initial={{ opacity: 0, x: -30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <TipTitle isDark={isDark}>
          <Target size={24} />
          Critical Interview Success Tips
        </TipTitle>
        <TipList isDark={isDark}>
          <TipItem isDark={isDark}>
            <strong>Know the Math:</strong> Be ready to explain attention mechanisms, scaling laws, and loss functions with mathematical precision.
          </TipItem>
          <TipItem isDark={isDark}>
            <strong>Understand Trade-offs:</strong> Always discuss computational complexity, memory requirements, and quality trade-offs.
          </TipItem>
          <TipItem isDark={isDark}>
            <strong>Recent Developments:</strong> Stay current with latest papers on RLHF, diffusion models, and architectural innovations.
          </TipItem>
          <TipItem isDark={isDark}>
            <strong>Practical Experience:</strong> Be prepared to discuss implementation challenges and production considerations.
          </TipItem>
        </TipList>
      </PreparationTips>

      <ContentSection>
        <FeatureCard
          isDark={isDark}
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          whileHover={{ scale: 1.02 }}
        >
          <FeatureTitle isDark={isDark}>
            <Brain size={24} />
            Deep Technical Content
          </FeatureTitle>
          <FeatureDescription isDark={isDark}>
            Master every aspect of modern GenAI systems with interview-focused explanations, 
            mathematical details, and practical insights that interviewers love to explore.
          </FeatureDescription>
          <FeatureList isDark={isDark}>
            <li>Transformer architecture deep dive</li>
            <li>LLM training and scaling laws</li>
            <li>RLHF and alignment techniques</li>
            <li>Diffusion models and sampling</li>
            <li>Production deployment strategies</li>
          </FeatureList>
          <ActionButton isDark={isDark} onClick={handleStartLearning}>
            Start Learning <ArrowRight size={20} />
          </ActionButton>
        </FeatureCard>

        <FeatureCard
          isDark={isDark}
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          whileHover={{ scale: 1.02 }}
        >
          <FeatureTitle isDark={isDark}>
            <Award size={24} />
            Interview-Style Questions
          </FeatureTitle>
          <FeatureDescription isDark={isDark}>
            Practice with questions that mirror real GenAI interviews. Each question includes 
            detailed explanations and tips on what interviewers are looking for.
          </FeatureDescription>
          <FeatureList isDark={isDark}>
            <li>High-frequency interview topics</li>
            <li>Difficulty-graded questions</li>
            <li>Detailed answer explanations</li>
            <li>Common mistake warnings</li>
            <li>Follow-up question preparation</li>
          </FeatureList>
          <ActionButton isDark={isDark} onClick={handleStartQuiz}>
            Take Practice Quiz <CheckCircle size={20} />
          </ActionButton>
        </FeatureCard>
      </ContentSection>
    </PageContainer>
  );
};
