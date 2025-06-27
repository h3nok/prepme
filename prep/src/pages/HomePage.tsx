import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowRight, Brain, Target, Zap, Building2, Users, Globe, Star } from 'lucide-react';
import { Link } from 'react-router-dom';

// Glassmorphic Container Base
const GlassContainer = styled(motion.div)`
  background: rgba(30, 41, 59, 0.4);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  box-shadow: 
    0 8px 32px rgba(0, 0, 0, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
`;

// Hero Section with Glassmorphic Design
const Hero = styled(GlassContainer)`
  text-align: center;
  padding: 4rem 2rem;
  margin-bottom: 3rem;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255, 107, 53, 0.1) 0%, transparent 70%);
    animation: float 20s ease-in-out infinite;
  }

  @keyframes float {
    0%, 100% { transform: translate(-50%, -50%) rotate(0deg); }
    50% { transform: translate(-50%, -50%) rotate(180deg); }
  }
`;

const HeroContent = styled.div`
  position: relative;
  z-index: 2;
`;

const HeroTitle = styled(motion.h1)`
  font-size: clamp(2.5rem, 6vw, 4rem);
  font-weight: 800;
  background: linear-gradient(135deg, #f8fafc 0%, #ff6b35 50%, #f8fafc 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1.5rem;
  line-height: 1.1;
`;

const HeroSubtitle = styled(motion.p)`
  font-size: 1.25rem;
  color: rgba(248, 250, 252, 0.8);
  margin-bottom: 2rem;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.6;
`;

const CTAButton = styled(motion(Link))`
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
  color: white;
  border-radius: 50px;
  text-decoration: none;
  font-weight: 600;
  font-size: 1.1rem;
  box-shadow: 
    0 8px 32px rgba(255, 107, 53, 0.3),
    0 4px 16px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.1);

  &:hover {
    transform: translateY(-4px);
    box-shadow: 
      0 12px 40px rgba(255, 107, 53, 0.4),
      0 8px 24px rgba(0, 0, 0, 0.3);
    color: white;
  }

  svg {
    transition: transform 0.3s ease;
  }

  &:hover svg {
    transform: translateX(4px);
  }
`;

// Stats Grid with Glassmorphic Cards
const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
`;

const StatCard = styled(GlassContainer)`
  text-align: center;
  padding: 2rem 1.5rem;
  transition: all 0.4s ease;

  &:hover {
    transform: translateY(-8px);
    box-shadow: 
      0 20px 50px rgba(0, 0, 0, 0.4),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
  }
`;

const StatIcon = styled.div`
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.2) 0%, rgba(255, 107, 53, 0.1) 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
  border: 1px solid rgba(255, 107, 53, 0.3);

  svg {
    width: 32px;
    height: 32px;
    color: #ff6b35;
  }
`;

const StatNumber = styled.h3`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
`;

const StatLabel = styled.p`
  color: rgba(248, 250, 252, 0.7);
  margin: 0;
  font-weight: 500;
  font-size: 1rem;
`;

// Companies Section
const CompaniesSection = styled.section`
  margin: 4rem 0;
  text-align: center;
`;

const SectionTitle = styled(motion.h2)`
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #f8fafc 0%, #ff6b35 50%, #f8fafc 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
`;

const SectionSubtitle = styled(motion.p)`
  color: rgba(248, 250, 252, 0.7);
  max-width: 800px;
  margin: 0 auto 3rem;
  font-size: 1.2rem;
  line-height: 1.6;
`;

const CompaniesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
`;

const CompanyCard = styled(GlassContainer)<{ accentColor?: string }>`
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.4s ease;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, ${props => props.accentColor || '#ff6b35'} 0%, transparent 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  &:hover {
    transform: translateY(-12px);
    box-shadow: 
      0 25px 60px rgba(0, 0, 0, 0.5),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
    
    &::before {
      opacity: 1;
    }
  }
`;

const CompanyLogo = styled.div<{ accentColor?: string }>`
  width: 100px;
  height: 100px;
  margin: 0 auto 1.5rem;
  background: linear-gradient(135deg, ${props => props.accentColor || '#ff6b35'}20 0%, ${props => props.accentColor || '#ff6b35'}10 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
  font-weight: bold;
  color: ${props => props.accentColor || '#ff6b35'};
  border: 2px solid ${props => props.accentColor || '#ff6b35'}40;
  box-shadow: 0 8px 32px ${props => props.accentColor || '#ff6b35'}20;
`;

const CompanyName = styled.h3`
  font-size: 1.5rem;
  font-weight: 700;
  color: #f8fafc;
  margin: 0 0 1rem;
`;

const CompanyFocus = styled.p`
  color: rgba(248, 250, 252, 0.7);
  font-size: 1rem;
  line-height: 1.6;
  margin: 0 0 1.5rem;
`;

const CompanyTopics = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
`;

const TopicTag = styled.span<{ accentColor?: string }>`
  background: ${props => props.accentColor || '#ff6b35'}20;
  color: ${props => props.accentColor || '#ff6b35'};
  padding: 0.375rem 0.75rem;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 500;
  border: 1px solid ${props => props.accentColor || '#ff6b35'}30;
  backdrop-filter: blur(10px);
`;

// Features Section
const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  margin: 4rem 0;
`;

const FeatureCard = styled(GlassContainer)`
  padding: 2.5rem;
  text-align: center;
  transition: all 0.4s ease;

  &:hover {
    transform: translateY(-8px);
    box-shadow: 
      0 20px 50px rgba(0, 0, 0, 0.4),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
  }

  h4 {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    font-size: 1.5rem;
    color: #f8fafc;
    
    svg {
      color: #ff6b35;
      width: 28px;
      height: 28px;
    }
  }

  p {
    color: rgba(248, 250, 252, 0.8);
    line-height: 1.7;
    font-size: 1.1rem;
  }
`;

// Quick Start Section
const QuickStartGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
`;

const QuickStartCard = styled(motion(Link))`
  display: block;
  padding: 2rem;
  background: rgba(30, 41, 59, 0.3);
  backdrop-filter: blur(15px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  text-decoration: none;
  transition: all 0.4s ease;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff6b35 0%, #ff8c42 100%);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
  }

  &:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    border-color: rgba(255, 107, 53, 0.3);

    &::before {
      transform: scaleX(1);
    }
  }

  h3 {
    color: #ff6b35;
    margin: 0 0 1rem 0;
    font-weight: 600;
    font-size: 1.3rem;
  }

  p {
    color: rgba(248, 250, 252, 0.8);
    margin: 0;
    line-height: 1.6;
  }
`;

const HomePage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      style={{ 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)',
        padding: '2rem 1rem'
      }}
    >
      <Hero
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        <HeroContent>
          <HeroTitle
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Master AI Interviews at Top Companies
          </HeroTitle>
          <HeroSubtitle
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            Enterprise-ready preparation platform for Senior AI Scientists & ML Engineers. 
            Comprehensive coverage tailored for Google, Meta, OpenAI, Anthropic, Amazon, and other leading AI companies.
          </HeroSubtitle>
          <CTAButton
            to="/transformers"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Start Learning
            <ArrowRight />
          </CTAButton>
        </HeroContent>
      </Hero>

      <StatsGrid>
        <StatCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <StatIcon><Building2 /></StatIcon>
          <StatNumber>20+</StatNumber>
          <StatLabel>AI Companies</StatLabel>
        </StatCard>
        
        <StatCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <StatIcon><Brain /></StatIcon>
          <StatNumber>500+</StatNumber>
          <StatLabel>Interview Questions</StatLabel>
        </StatCard>
        
        <StatCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <StatIcon><Users /></StatIcon>
          <StatNumber>10K+</StatNumber>
          <StatLabel>Engineers Prepared</StatLabel>
        </StatCard>
        
        <StatCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <StatIcon><Globe /></StatIcon>
          <StatNumber>24/7</StatNumber>
          <StatLabel>Global Access</StatLabel>
        </StatCard>
      </StatsGrid>

      <CompaniesSection>
        <SectionTitle
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <Building2 />
          Interview Prep for Top AI Companies
        </SectionTitle>
        <SectionSubtitle
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          Tailored preparation for the specific focus areas and interview styles of leading AI companies
        </SectionSubtitle>
        
        <CompaniesGrid>
          <CompanyCard
            accentColor="#4285f4"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <CompanyLogo accentColor="#4285f4">G</CompanyLogo>
            <CompanyName>Google / DeepMind</CompanyName>
            <CompanyFocus>
              Research excellence, scalable systems, and foundational AI research. 
              Strong focus on theoretical understanding and practical implementation.
            </CompanyFocus>
            <CompanyTopics>
              <TopicTag accentColor="#4285f4">Transformers</TopicTag>
              <TopicTag accentColor="#4285f4">Search/Ranking</TopicTag>
              <TopicTag accentColor="#4285f4">Distributed ML</TopicTag>
              <TopicTag accentColor="#4285f4">Reinforcement Learning</TopicTag>
            </CompanyTopics>
          </CompanyCard>

          <CompanyCard
            accentColor="#ff6b35"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.9 }}
          >
            <CompanyLogo accentColor="#ff6b35">A</CompanyLogo>
            <CompanyName>Amazon</CompanyName>
            <CompanyFocus>
              Customer-obsessed AI solutions, cloud-scale deployment, and practical business impact. 
              Emphasis on system design and operational excellence.
            </CompanyFocus>
            <CompanyTopics>
              <TopicTag accentColor="#ff6b35">MLOps</TopicTag>
              <TopicTag accentColor="#ff6b35">Recommendation</TopicTag>
              <TopicTag accentColor="#ff6b35">Alexa/NLP</TopicTag>
              <TopicTag accentColor="#ff6b35">AWS Services</TopicTag>
            </CompanyTopics>
          </CompanyCard>

          <CompanyCard
            accentColor="#1877f2"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.0 }}
          >
            <CompanyLogo accentColor="#1877f2">M</CompanyLogo>
            <CompanyName>Meta</CompanyName>
            <CompanyFocus>
              Social-scale AI, computer vision, and immersive experiences. 
              Focus on real-time systems and billion-user applications.
            </CompanyFocus>
            <CompanyTopics>
              <TopicTag accentColor="#1877f2">Computer Vision</TopicTag>
              <TopicTag accentColor="#1877f2">Feed Ranking</TopicTag>
              <TopicTag accentColor="#1877f2">Multimodal</TopicTag>
              <TopicTag accentColor="#1877f2">AR/VR AI</TopicTag>
            </CompanyTopics>
          </CompanyCard>

          <CompanyCard
            accentColor="#00d4aa"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.1 }}
          >
            <CompanyLogo accentColor="#00d4aa">O</CompanyLogo>
            <CompanyName>OpenAI</CompanyName>
            <CompanyFocus>
              AGI research, large language models, and safety alignment. 
              Cutting-edge research with focus on capabilities and alignment.
            </CompanyFocus>
            <CompanyTopics>
              <TopicTag accentColor="#00d4aa">LLMs</TopicTag>
              <TopicTag accentColor="#00d4aa">RLHF</TopicTag>
              <TopicTag accentColor="#00d4aa">Safety</TopicTag>
              <TopicTag accentColor="#00d4aa">Scaling Laws</TopicTag>
            </CompanyTopics>
          </CompanyCard>

          <CompanyCard
            accentColor="#9b59b6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.2 }}
          >
            <CompanyLogo accentColor="#9b59b6">A</CompanyLogo>
            <CompanyName>Anthropic</CompanyName>
            <CompanyFocus>
              AI safety, constitutional AI, and responsible scaling. 
              Deep focus on alignment, interpretability, and safe AI development.
            </CompanyFocus>
            <CompanyTopics>
              <TopicTag accentColor="#9b59b6">Constitutional AI</TopicTag>
              <TopicTag accentColor="#9b59b6">Safety Research</TopicTag>
              <TopicTag accentColor="#9b59b6">Interpretability</TopicTag>
              <TopicTag accentColor="#9b59b6">Alignment</TopicTag>
            </CompanyTopics>
          </CompanyCard>

          <CompanyCard
            accentColor="#1db954"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.3 }}
          >
            <CompanyLogo accentColor="#1db954">+</CompanyLogo>
            <CompanyName>Leading AI Companies</CompanyName>
            <CompanyFocus>
              Microsoft, Apple, NVIDIA, Tesla, ByteDance, Stability AI, and emerging AI startups. 
              Diverse opportunities across industries and applications.
            </CompanyFocus>
            <CompanyTopics>
              <TopicTag accentColor="#1db954">Autonomous Systems</TopicTag>
              <TopicTag accentColor="#1db954">Edge AI</TopicTag>
              <TopicTag accentColor="#1db954">Generative AI</TopicTag>
              <TopicTag accentColor="#1db954">Robotics</TopicTag>
            </CompanyTopics>
          </CompanyCard>
        </CompaniesGrid>
      </CompaniesSection>

      <SectionTitle
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1.4 }}
      >
        Why Choose PrepMe?
      </SectionTitle>
      
      <FeaturesGrid>
        <FeatureCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.5 }}
        >
          <h4><Brain />Research-Grade Content</h4>
          <p>
            Deep dive into cutting-edge AI research including transformers, LLMs, diffusion models, 
            and multimodal systems. Content reviewed by PhD researchers and industry experts.
          </p>
        </FeatureCard>
        
        <FeatureCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.6 }}
        >
          <h4><Target />Enterprise Assessment</h4>
          <p>
            Advanced quizzes and simulations designed for Senior Scientist roles. 
            Adaptive difficulty and detailed analytics to track progress and identify gaps.
          </p>
        </FeatureCard>
        
        <FeatureCard
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.7 }}
        >
          <h4><Zap />Multi-Company Focus</h4>
          <p>
            Content tailored for specific companies' interview styles and focus areas. 
            From Google's theoretical depth to Meta's scale challenges to OpenAI's cutting-edge research.
          </p>
        </FeatureCard>
      </FeaturesGrid>

      <CompaniesSection>
        <SectionTitle
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.8 }}
        >
          <Star />
          Start Your Journey
        </SectionTitle>
        
        <QuickStartGrid>
          <QuickStartCard
            to="/transformers"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.9 }}
          >
            <h3>🏗️ Transformer Architecture</h3>
            <p>
              Master attention mechanisms, positional encoding, and multi-head attention. 
              Understand the foundation of modern AI.
            </p>
          </QuickStartCard>
          
          <QuickStartCard
            to="/llms"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 2.0 }}
          >
            <h3>🤖 Large Language Models</h3>
            <p>
              Scaling laws, training techniques, RLHF, and emergent abilities. 
              Deep dive into GPT, BERT, and beyond.
            </p>
          </QuickStartCard>
          
          <QuickStartCard
            to="/diffusion"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 2.1 }}
          >
            <h3>🎨 Diffusion Models</h3>
            <p>
              From DDPM to Stable Diffusion. Mathematical foundations, 
              sampling methods, and conditioning techniques.
            </p>
          </QuickStartCard>
          
          <QuickStartCard
            to="/multimodal"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 2.2 }}
          >
            <h3>👁️ Multimodal AI</h3>
            <p>
              Vision-language models, cross-modal attention, and fusion strategies. 
              CLIP, DALL-E, and modern approaches.
            </p>
          </QuickStartCard>
          
          <QuickStartCard
            to="/aws"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 2.3 }}
          >
            <h3>☁️ Production & MLOps</h3>
            <p>
              Cloud deployment, scaling strategies, and MLOps best practices. 
              From AWS to GCP, deploy AI systems at enterprise scale.
            </p>
          </QuickStartCard>
          
          <QuickStartCard
            to="/quiz"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 2.4 }}
          >
            <h3>🎯 Practice Quizzes</h3>
            <p>
              Test your knowledge with interactive quizzes across all topics. 
              Track progress and identify areas for improvement.
            </p>
          </QuickStartCard>
        </QuickStartGrid>
      </CompaniesSection>
    </motion.div>
  );
};

export default HomePage;
