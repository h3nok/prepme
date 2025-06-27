import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BookOpen, Brain, Calculator, ChevronRight, Clock, 
  Trophy, Zap, Target, Sparkles, Play, BarChart3,
  Layers, Beaker, Lightbulb, Code, GitBranch
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { LearningModule, UserProgress } from '../types/LearningModule';
import { fundamentalsModule } from '../data/FundamentalsModule';

// Glassmorphic styling
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

const DashboardContainer = styled.div`
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
`;

const DashboardHeader = styled(GlassContainer)`
  padding: 2rem;
  margin-bottom: 2rem;
  text-align: center;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, #f8fafc 0%, #ff6b35 50%, #7c3aed 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
`;

const Subtitle = styled.p`
  color: rgba(248, 250, 252, 0.8);
  font-size: 1.1rem;
  max-width: 600px;
  margin: 0 auto;
`;

// Hexagonal Grid System
const ModuleGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
`;

const HexModule = styled.div`
  position: relative;
  aspect-ratio: 1;
  max-width: 320px;
  margin: 0 auto;
`;

const HexagonShape = styled(motion.div)<{ $color: string; $completed: boolean }>`
  width: 100%;
  height: 100%;
  background: ${props => props.$completed 
    ? `linear-gradient(135deg, ${props.$color}20, ${props.$color}40)`
    : `linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(30, 41, 59, 0.8))`
  };
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 2px solid ${props => props.$completed ? props.$color : 'rgba(255, 255, 255, 0.1)'};
  border-radius: 24px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, ${props => props.$color}10, transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  &:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 
      0 20px 40px rgba(0, 0, 0, 0.4),
      0 0 0 1px ${props => props.$color}40;
    
    &::before {
      opacity: 1;
    }
  }
`;

const ModuleContent = styled.div`
  position: absolute;
  inset: 2rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  z-index: 2;
`;

const ModuleHeader = styled.div`
  text-align: center;
  margin-bottom: 1rem;
`;

const ModuleIcon = styled.div<{ $color: string }>`
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, ${props => props.$color}, ${props => props.$color}80);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
  box-shadow: 0 8px 24px ${props => props.$color}30;

  svg {
    width: 30px;
    height: 30px;
    color: white;
  }
`;

const ModuleTitle = styled.h3`
  color: #f8fafc;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  line-height: 1.3;
`;

const ModuleDescription = styled.p`
  color: rgba(248, 250, 252, 0.7);
  font-size: 0.9rem;
  line-height: 1.4;
  margin-bottom: 1rem;
`;

const ModuleStats = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const StatItem = styled.div`
  text-align: center;
`;

const StatValue = styled.div`
  color: #f8fafc;
  font-weight: 700;
  font-size: 1.1rem;
`;

const StatLabel = styled.div`
  color: rgba(248, 250, 252, 0.6);
  font-size: 0.8rem;
  margin-top: 0.25rem;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 1rem;
`;

const ProgressFill = styled(motion.div)<{ $color: string }>`
  height: 100%;
  background: linear-gradient(90deg, ${props => props.$color}, ${props => props.$color}80);
  border-radius: 4px;
`;

const StartButton = styled(motion.button)<{ $color: string }>`
  width: 100%;
  padding: 0.75rem;
  background: linear-gradient(135deg, ${props => props.$color}, ${props => props.$color}80);
  color: white;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px ${props => props.$color}40;
  }
`;

// Quick Stats Section
const QuickStats = styled(GlassContainer)`
  padding: 2rem;
  margin-bottom: 2rem;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
`;

const StatCard = styled(motion.div)<{ $color: string }>`
  text-align: center;
  padding: 1.5rem;
  background: linear-gradient(135deg, ${props => props.$color}20, ${props => props.$color}10);
  border: 1px solid ${props => props.$color}30;
  border-radius: 16px;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px ${props => props.$color}20;
  }
`;

const StatIcon = styled.div<{ $color: string }>`
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: linear-gradient(135deg, ${props => props.$color}, ${props => props.$color}80);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;

  svg {
    width: 24px;
    height: 24px;
    color: white;
  }
`;

// Learning Path Visualization
const LearningPath = styled(GlassContainer)`
  padding: 2rem;
  margin-bottom: 2rem;
`;

const PathTitle = styled.h2`
  color: #f8fafc;
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  text-align: center;
`;

const PathFlow = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
`;

const PathNode = styled(motion.div)<{ $active: boolean; $completed: boolean; $color: string }>`
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: ${props => 
    props.$completed 
      ? `linear-gradient(135deg, ${props.$color}, ${props.$color}80)`
      : props.$active
        ? `linear-gradient(135deg, ${props.$color}40, ${props.$color}20)`
        : 'rgba(30, 41, 59, 0.6)'
  };
  border: 2px solid ${props => 
    props.$completed || props.$active ? props.$color : 'rgba(255, 255, 255, 0.1)'
  };
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;

  &:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 32px ${props => props.$color}30;
  }
`;

const PathNodeTitle = styled.div`
  color: #f8fafc;
  font-weight: 600;
  font-size: 0.9rem;
  text-align: center;
  margin-top: 0.5rem;
`;

const PathArrow = styled(ChevronRight)`
  width: 24px;
  height: 24px;
  color: rgba(248, 250, 252, 0.5);
  margin: 0 0.5rem;
`;

const modules: LearningModule[] = [fundamentalsModule];

const pathSteps = [
  { id: 'fundamentals', title: 'Fundamentals', color: '#ff6b35', completed: false, active: true },
  { id: 'transformers', title: 'Transformers', color: '#7c3aed', completed: false, active: false },
  { id: 'llms', title: 'LLMs', color: '#059669', completed: false, active: false },
  { id: 'multimodal', title: 'Multimodal', color: '#dc2626', completed: false, active: false },
  { id: 'systems', title: 'ML Systems', color: '#0891b2', completed: false, active: false }
];

const LearningDashboard: React.FC = () => {
  const [userProgress, setUserProgress] = useState<UserProgress[]>([]);
  const [stats, setStats] = useState({
    totalHours: 0,
    completedModules: 0,
    currentStreak: 0,
    totalConcepts: 0
  });

  useEffect(() => {
    // Calculate stats
    const totalConcepts = modules.reduce((sum, module) => sum + module.concepts.length, 0);
    const totalHours = modules.reduce((sum, module) => sum + module.estimatedHours, 0);
    
    setStats({
      totalHours,
      completedModules: 0,
      currentStreak: 3,
      totalConcepts
    });
  }, []);

  const getModuleProgress = (moduleId: string): number => {
    // For demo, return sample progress
    if (moduleId === 'fundamentals') return 0;
    return 0;
  };

  return (
    <DashboardContainer>
      <DashboardHeader
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Title>Your Learning Journey</Title>
        <Subtitle>
          Master machine learning fundamentals through interactive visualizations, 
          progressive disclosure, and hands-on practice
        </Subtitle>
      </DashboardHeader>

      <QuickStats
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <StatsGrid>
          <StatCard $color="#ff6b35" whileHover={{ y: -4 }}>
            <StatIcon $color="#ff6b35">
              <Clock />
            </StatIcon>
            <StatValue>{stats.totalHours}h</StatValue>
            <StatLabel>Total Content</StatLabel>
          </StatCard>
          
          <StatCard $color="#7c3aed" whileHover={{ y: -4 }}>
            <StatIcon $color="#7c3aed">
              <Trophy />
            </StatIcon>
            <StatValue>{stats.completedModules}</StatValue>
            <StatLabel>Completed Modules</StatLabel>
          </StatCard>
          
          <StatCard $color="#059669" whileHover={{ y: -4 }}>
            <StatIcon $color="#059669">
              <Zap />
            </StatIcon>
            <StatValue>{stats.currentStreak}</StatValue>
            <StatLabel>Day Streak</StatLabel>
          </StatCard>
          
          <StatCard $color="#dc2626" whileHover={{ y: -4 }}>
            <StatIcon $color="#dc2626">
              <Target />
            </StatIcon>
            <StatValue>{stats.totalConcepts}</StatValue>
            <StatLabel>Total Concepts</StatLabel>
          </StatCard>
        </StatsGrid>
      </QuickStats>

      <LearningPath
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <PathTitle>Learning Path</PathTitle>
        <PathFlow>
          {pathSteps.map((step, index) => (
            <React.Fragment key={step.id}>
              <PathNode
                $active={step.active}
                $completed={step.completed}
                $color={step.color}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Brain size={24} />
                <PathNodeTitle>{step.title}</PathNodeTitle>
              </PathNode>
              {index < pathSteps.length - 1 && <PathArrow />}
            </React.Fragment>
          ))}
        </PathFlow>
      </LearningPath>

      <ModuleGrid>
        {modules.map((module, index) => {
          const progress = getModuleProgress(module.id);
          const isCompleted = progress >= 100;
          
          return (
            <motion.div
              key={module.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
            >
              <HexModule>
                <HexagonShape
                  $color={module.color}
                  $completed={isCompleted}
                  whileHover={{ y: -8, scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <ModuleContent>
                    <ModuleHeader>
                      <ModuleIcon $color={module.color}>
                        <Brain />
                      </ModuleIcon>
                      <ModuleTitle>{module.title}</ModuleTitle>
                      <ModuleDescription>{module.description}</ModuleDescription>
                    </ModuleHeader>

                    <div>
                      <ModuleStats>
                        <StatItem>
                          <StatValue>{module.concepts.length}</StatValue>
                          <StatLabel>Concepts</StatLabel>
                        </StatItem>
                        <StatItem>
                          <StatValue>{module.estimatedHours}h</StatValue>
                          <StatLabel>Duration</StatLabel>
                        </StatItem>
                        <StatItem>
                          <StatValue>{module.difficulty}</StatValue>
                          <StatLabel>Level</StatLabel>
                        </StatItem>
                      </ModuleStats>

                      <ProgressBar>
                        <ProgressFill
                          $color={module.color}
                          initial={{ width: 0 }}
                          animate={{ width: `${progress}%` }}
                          transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                        />
                      </ProgressBar>

                      <Link to={`/learning/${module.id}`} style={{ textDecoration: 'none' }}>
                        <StartButton
                          $color={module.color}
                          whileHover={{ y: -2 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <Play size={16} />
                          {progress > 0 ? 'Continue' : 'Start Learning'}
                        </StartButton>
                      </Link>
                    </div>
                  </ModuleContent>
                </HexagonShape>
              </HexModule>
            </motion.div>
          );
        })}
      </ModuleGrid>
    </DashboardContainer>
  );
};

export default LearningDashboard;
