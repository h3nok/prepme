import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowRight, Zap, Brain, Target } from 'lucide-react';
import { Link } from 'react-router-dom';

const LandingContainer = styled.div`
  min-height: 100vh;
  background: ${props => props.theme.colors.background === '#ffffff' 
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)'};
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  position: relative;
  overflow: hidden;

  /* Animated background */
  &::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: ${props => props.theme.colors.background === '#ffffff' 
      ? 'conic-gradient(from 0deg, rgba(102, 126, 234, 0.1) 0%, rgba(255, 107, 53, 0.15) 33%, rgba(118, 75, 162, 0.1) 66%, rgba(102, 126, 234, 0.1) 100%)'
      : 'conic-gradient(from 0deg, rgba(255, 107, 53, 0.15) 0%, rgba(59, 130, 246, 0.1) 33%, rgba(16, 185, 129, 0.1) 66%, rgba(255, 107, 53, 0.1) 100%)'};
    animation: rotate 30s linear infinite;
    opacity: 0.3;
  }

  @keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const LandingContent = styled(motion.div)`
  text-align: center;
  max-width: 800px;
  z-index: 2;
  position: relative;
`;

const Logo = styled(motion.div)`
  margin-bottom: 2rem;
  
  h1 {
    font-size: clamp(3.5rem, 8vw, 6rem);
    font-weight: 900;
    background: linear-gradient(135deg, #ff6b35 0%, #f093fb 50%, #f5576c 100%);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
    text-shadow: 0 10px 30px rgba(255, 107, 53, 0.3);
  }
`;

const Tagline = styled(motion.p)`
  font-size: 1.5rem;
  color: ${props => props.theme.colors.background === '#ffffff' ? '#ffffff' : '#cbd5e1'};
  margin-bottom: 3rem;
  font-weight: 500;
  line-height: 1.6;
  text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
`;

const FeatureGrid = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
`;

const FeatureCard = styled(motion.div)`
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 20px;
  text-align: center;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    background: rgba(255, 255, 255, 0.15);
  }

  svg {
    width: 40px;
    height: 40px;
    color: #ff6b35;
    margin-bottom: 1rem;
  }

  h3 {
    color: ${props => props.theme.colors.background === '#ffffff' ? '#ffffff' : '#f8fafc'};
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
  }

  p {
    color: ${props => props.theme.colors.background === '#ffffff' ? 'rgba(255, 255, 255, 0.8)' : '#cbd5e1'};
    font-size: 0.9rem;
    line-height: 1.5;
  }
`;

const CTAButton = styled(motion(Link))`
  display: inline-flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem 3rem;
  background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 50%, #ff6b35 100%);
  color: white;
  border-radius: 60px;
  text-decoration: none;
  font-weight: 700;
  font-size: 1.2rem;
  position: relative;
  overflow: hidden;
  box-shadow: 
    0 10px 30px rgba(255, 107, 53, 0.4),
    0 5px 15px rgba(0, 0, 0, 0.2);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  border: 2px solid transparent;

  /* Animated background */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.6s;
  }

  &:hover {
    transform: translateY(-8px) scale(1.05);
    box-shadow: 
      0 25px 50px rgba(255, 107, 53, 0.6),
      0 15px 30px rgba(0, 0, 0, 0.3);
    color: white;
    
    &::before {
      left: 100%;
    }

    svg {
      transform: translateX(8px);
    }
  }

  &:active {
    transform: translateY(-4px) scale(1.02);
  }

  svg {
    transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
  }
`;

const LandingPage: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: "AI Interview Prep",
      description: "Advanced preparation for senior AI roles and research positions"
    },
    {
      icon: Target,
      title: "Expert Content",
      description: "Curated by industry professionals and research scientists"
    },
    {
      icon: Zap,
      title: "Interactive Learning",
      description: "Dynamic visualizations and hands-on practice problems"
    }
  ];

  return (
    <LandingContainer>
      <LandingContent
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Logo
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h1>PrepMe</h1>
        </Logo>

        <Tagline
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          Professional AI Interview Preparation Platform
          <br />
          for Senior Scientists & Advanced AI Roles
        </Tagline>

        <FeatureGrid
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.8 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <feature.icon />
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </FeatureCard>
          ))}
        </FeatureGrid>

        <CTAButton
          to="/home"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.2 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
        >
          Enter PrepMe Platform
          <ArrowRight />
        </CTAButton>
      </LandingContent>
    </LandingContainer>
  );
};

export default LandingPage;
