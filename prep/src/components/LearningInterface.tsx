import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronLeft, ChevronRight, ChevronDown, ChevronUp,
  BookOpen, Calculator, Lightbulb, Play, Pause,
  RotateCcw, Settings, Bookmark, Menu, X,
  ArrowLeft, ArrowRight, Eye, EyeOff
} from 'lucide-react';
import { LearningModule, Concept, Slide } from '../types/LearningModule';
import { fundamentalsModule } from '../data/FundamentalsModule';
import { transformerModule } from '../data/TransformerModule';
import { useTheme } from '../context/ThemeContext';

// Import visualization components
import AttentionVisualization from './visualizations/AttentionVisualization';
import MultiHeadAttention from './visualizations/MultiHeadAttention';
import SinusoidalEncoding from './visualizations/SinusoidalEncoding';
import ScalingLaws from './visualizations/ScalingLaws';
import NextTokenPrediction from './visualizations/NextTokenPrediction';
import DiffusionProcess from './visualizations/DiffusionProcess';

// Main Layout
const LearningContainer = styled.div`
  display: flex;
  height: 100vh;
  background: ${props => props.theme.colors.background};
  overflow: hidden;
`;

// Left Panel - Navigation
const NavigationPanel = styled(motion.div)<{ $isCollapsed: boolean }>`
  width: ${props => props.$isCollapsed ? '60px' : '300px'};
  background: rgba(30, 41, 59, 0.8);
  backdrop-filter: blur(20px);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  z-index: 100;
`;

const NavHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const NavTitle = styled.h2<{ $show: boolean }>`
  color: #f8fafc;
  font-size: 1.2rem;
  font-weight: 700;
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
`;

const CollapseButton = styled.button`
  background: none;
  border: none;
  color: rgba(248, 250, 252, 0.7);
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 8px;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #f8fafc;
  }
`;

const ConceptList = styled.div`
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
`;

const ConceptItem = styled(motion.div)<{ $active: boolean; $completed: boolean }>`
  padding: 1rem;
  margin-bottom: 0.5rem;
  border-radius: 12px;
  cursor: pointer;
  background: ${props => 
    props.$active 
      ? 'linear-gradient(135deg, rgba(255, 107, 53, 0.2), rgba(255, 107, 53, 0.1))'
      : props.$completed
        ? 'rgba(16, 185, 129, 0.1)'
        : 'rgba(255, 255, 255, 0.05)'
  };
  border: 1px solid ${props => 
    props.$active 
      ? 'rgba(255, 107, 53, 0.3)'
      : props.$completed
        ? 'rgba(16, 185, 129, 0.3)'
        : 'rgba(255, 255, 255, 0.1)'
  };
  transition: all 0.3s ease;

  &:hover {
    background: rgba(255, 107, 53, 0.1);
    border-color: rgba(255, 107, 53, 0.2);
  }
`;

const ConceptTitle = styled.h4<{ $show: boolean }>`
  color: #f8fafc;
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
`;

const ConceptProgress = styled.div<{ $show: boolean }>`
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $progress: number }>`
  width: ${props => props.$progress}%;
  height: 100%;
  background: linear-gradient(90deg, #ff6b35, #ff8c42);
  transition: width 0.3s ease;
`;

// Center Panel - Content
const ContentPanel = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: ${props => props.theme.colors.background};
  position: relative;
`;

const ContentHeader = styled.div`
  padding: 1.5rem 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(30, 41, 59, 0.6);
  backdrop-filter: blur(20px);
`;

const ContentTitle = styled.h1`
  color: #f8fafc;
  font-size: 1.8rem;
  font-weight: 700;
`;

const SlideControls = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const ControlButton = styled.button<{ $active?: boolean }>`
  background: ${props => props.$active ? '#ff6b35' : 'rgba(255, 255, 255, 0.1)'};
  border: 1px solid ${props => props.$active ? '#ff6b35' : 'rgba(255, 255, 255, 0.2)'};
  color: #f8fafc;
  padding: 0.75rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;

  &:hover {
    background: ${props => props.$active ? '#ff8c42' : 'rgba(255, 255, 255, 0.2)'};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const SlideContent = styled.div`
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
  max-width: 800px;
  margin: 0 auto;
`;

const ContentTier = styled(motion.div)<{ $visible: boolean }>`
  margin-bottom: 2rem;
  opacity: ${props => props.$visible ? 1 : 0.3};
  transition: opacity 0.3s ease;
`;

const TierHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
  cursor: pointer;
`;

const TierLabel = styled.span<{ $color: string }>`
  background: linear-gradient(135deg, ${props => props.$color}, ${props => props.$color}80);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
`;

const TierText = styled.div`
  color: #f8fafc;
  font-size: 1.1rem;
  line-height: 1.7;
  margin-left: 1rem;
`;

const MathContainer = styled.div`
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1.5rem 0;
  text-align: center;
`;

const MathFormula = styled.div`
  color: #f8fafc;
  font-size: 1.2rem;
  margin-bottom: 0.5rem;
`;

const MathExplanation = styled.p`
  color: rgba(248, 250, 252, 0.7);
  font-size: 0.9rem;
  font-style: italic;
`;

// Right Panel - Tools
const ToolsPanel = styled(motion.div)<{ $isCollapsed: boolean }>`
  width: ${props => props.$isCollapsed ? '60px' : '350px'};
  background: rgba(30, 41, 59, 0.8);
  backdrop-filter: blur(20px);
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  z-index: 100;
`;

const ToolsHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const ToolsContent = styled.div`
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
`;

const VisualizationContainer = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const ToolSection = styled.div`
  margin-bottom: 2rem;
`;

const ToolSectionTitle = styled.h3<{ $show: boolean }>`
  color: #f8fafc;
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
`;

const Navigation = styled.div`
  padding: 1.5rem 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(30, 41, 59, 0.6);
  backdrop-filter: blur(20px);
`;

const NavButton = styled.button`
  background: linear-gradient(135deg, #ff6b35, #ff8c42);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(255, 107, 53, 0.3);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const SlideIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: rgba(248, 250, 252, 0.7);
  font-size: 0.9rem;
`;

const modules: Record<string, LearningModule> = {
  fundamentals: fundamentalsModule,
  transformer: transformerModule
};

const LearningInterface: React.FC = () => {
  const { moduleId } = useParams<{ moduleId: string }>();
  const navigate = useNavigate();
  
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [toolsCollapsed, setToolsCollapsed] = useState(false);
  const [currentConceptIndex, setCurrentConceptIndex] = useState(0);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [visibleTiers, setVisibleTiers] = useState({ tier1: true, tier2: false, tier3: false });
  const [autoPlay, setAutoPlay] = useState(false);

  const module = moduleId ? modules[moduleId] : null;

  useEffect(() => {
    if (!module) {
      navigate('/learning');
      return;
    }
  }, [module, navigate]);

  useEffect(() => {
    if (!autoPlay) return;
    
    const timer = setInterval(() => {
      nextSlide();
    }, 10000); // 10 seconds per slide

    return () => clearInterval(timer);
  }, [autoPlay, currentConceptIndex, currentSlideIndex]);

  if (!module) return null;

  const currentConcept = module.concepts[currentConceptIndex];
  const currentSlide = currentConcept?.slides[currentSlideIndex];
  const totalSlides = currentConcept?.slides.length || 0;

  const nextSlide = () => {
    if (currentSlideIndex < totalSlides - 1) {
      setCurrentSlideIndex(prev => prev + 1);
    } else if (currentConceptIndex < module.concepts.length - 1) {
      setCurrentConceptIndex(prev => prev + 1);
      setCurrentSlideIndex(0);
    }
  };

  const prevSlide = () => {
    if (currentSlideIndex > 0) {
      setCurrentSlideIndex(prev => prev - 1);
    } else if (currentConceptIndex > 0) {
      setCurrentConceptIndex(prev => prev - 1);
      const prevConcept = module.concepts[currentConceptIndex - 1];
      setCurrentSlideIndex(prevConcept.slides.length - 1);
    }
  };

  const toggleTier = (tier: keyof typeof visibleTiers) => {
    setVisibleTiers(prev => ({ ...prev, [tier]: !prev[tier] }));
  };

  // Visualization renderer
  const renderVisualization = (visualization: any) => {
    switch (visualization.component) {
      case 'AttentionVisualization':
        return <AttentionVisualization data={visualization.data} controls={visualization.controls} />;
      case 'MultiHeadAttention':
        return <MultiHeadAttention data={visualization.data} controls={visualization.controls} />;
      case 'SinusoidalEncoding':
        return <SinusoidalEncoding data={visualization.data} controls={visualization.controls} />;
      case 'ScalingLaws':
        return <ScalingLaws data={visualization.data} controls={visualization.controls} />;
      case 'NextTokenPrediction':
        return <NextTokenPrediction data={visualization.data} controls={visualization.controls} />;
      case 'DiffusionProcess':
        return <DiffusionProcess data={visualization.data} controls={visualization.controls} />;
      default:
        return <div>Visualization not found: {visualization.component}</div>;
    }
  };

  return (
    <LearningContainer>
      {/* Navigation Panel */}
      <NavigationPanel $isCollapsed={navCollapsed}>
        <NavHeader>
          <NavTitle $show={!navCollapsed}>{module.title}</NavTitle>
          <CollapseButton onClick={() => setNavCollapsed(!navCollapsed)}>
            {navCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </CollapseButton>
        </NavHeader>
        
        <ConceptList>
          {module.concepts.map((concept, index) => (
            <ConceptItem
              key={concept.id}
              $active={index === currentConceptIndex}
              $completed={index < currentConceptIndex}
              onClick={() => {
                setCurrentConceptIndex(index);
                setCurrentSlideIndex(0);
              }}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
            >
              <ConceptTitle $show={!navCollapsed}>{concept.title}</ConceptTitle>
              <ConceptProgress $show={!navCollapsed}>
                <ProgressBar>
                  <ProgressFill $progress={index === currentConceptIndex ? (currentSlideIndex + 1) / concept.slides.length * 100 : index < currentConceptIndex ? 100 : 0} />
                </ProgressBar>
              </ConceptProgress>
            </ConceptItem>
          ))}
        </ConceptList>
      </NavigationPanel>

      {/* Content Panel */}
      <ContentPanel>
        <ContentHeader>
          <ContentTitle>{currentSlide?.title}</ContentTitle>
          <SlideControls>
            <ControlButton onClick={() => setAutoPlay(!autoPlay)} $active={autoPlay}>
              {autoPlay ? <Pause size={16} /> : <Play size={16} />}
              {autoPlay ? 'Pause' : 'Auto Play'}
            </ControlButton>
            <ControlButton onClick={() => setToolsCollapsed(!toolsCollapsed)}>
              <Settings size={16} />
              Tools
            </ControlButton>
          </SlideControls>
        </ContentHeader>

        <SlideContent>
          <AnimatePresence mode="wait">
            {currentSlide && (
              <motion.div
                key={`${currentConceptIndex}-${currentSlideIndex}`}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.5 }}
              >
                {/* Tier 1 - Core Concept */}
                <ContentTier $visible={visibleTiers.tier1}>
                  <TierHeader onClick={() => toggleTier('tier1')}>
                    <TierLabel $color="#ff6b35">Core Concept</TierLabel>
                    {visibleTiers.tier1 ? <EyeOff size={16} /> : <Eye size={16} />}
                  </TierHeader>
                  <TierText>{currentSlide.content.tier1}</TierText>
                </ContentTier>

                {/* Tier 2 - Mechanism */}
                {currentSlide.content.tier2 && (
                  <ContentTier $visible={visibleTiers.tier2}>
                    <TierHeader onClick={() => toggleTier('tier2')}>
                      <TierLabel $color="#7c3aed">How it Works</TierLabel>
                      {visibleTiers.tier2 ? <EyeOff size={16} /> : <Eye size={16} />}
                    </TierHeader>
                    {visibleTiers.tier2 && <TierText>{currentSlide.content.tier2}</TierText>}
                  </ContentTier>
                )}

                {/* Tier 3 - Technical Details */}
                {currentSlide.content.tier3 && (
                  <ContentTier $visible={visibleTiers.tier3}>
                    <TierHeader onClick={() => toggleTier('tier3')}>
                      <TierLabel $color="#059669">Technical Details</TierLabel>
                      {visibleTiers.tier3 ? <EyeOff size={16} /> : <Eye size={16} />}
                    </TierHeader>
                    {visibleTiers.tier3 && <TierText>{currentSlide.content.tier3}</TierText>}
                  </ContentTier>
                )}

                {/* Math Notations */}
                {currentSlide.mathNotations?.map(math => (
                  <MathContainer key={math.id}>
                    <MathFormula>{math.latex}</MathFormula>
                    <MathExplanation>{math.explanation}</MathExplanation>
                  </MathContainer>
                ))}

                {/* Visualizations */}
                {currentSlide.visualizations?.map(viz => (
                  <div key={viz.id}>
                    {renderVisualization(viz)}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </SlideContent>

        <Navigation>
          <NavButton onClick={prevSlide} disabled={currentConceptIndex === 0 && currentSlideIndex === 0}>
            <ArrowLeft size={16} />
            Previous
          </NavButton>
          
          <SlideIndicator>
            Slide {currentSlideIndex + 1} of {totalSlides} | Concept {currentConceptIndex + 1} of {module.concepts.length}
          </SlideIndicator>
          
          <NavButton 
            onClick={nextSlide} 
            disabled={currentConceptIndex === module.concepts.length - 1 && currentSlideIndex === totalSlides - 1}
          >
            Next
            <ArrowRight size={16} />
          </NavButton>
        </Navigation>
      </ContentPanel>

      {/* Tools Panel */}
      <ToolsPanel $isCollapsed={toolsCollapsed}>
        <ToolsHeader>
          <NavTitle $show={!toolsCollapsed}>Learning Tools</NavTitle>
          <CollapseButton onClick={() => setToolsCollapsed(!toolsCollapsed)}>
            {toolsCollapsed ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </CollapseButton>
        </ToolsHeader>
        
        <ToolsContent>
          <ToolSection>
            <ToolSectionTitle $show={!toolsCollapsed}>Visualizations</ToolSectionTitle>
            {currentSlide?.visualizations?.map(viz => (
              <VisualizationContainer key={viz.id}>
                <div style={{ color: '#f8fafc', textAlign: 'center' }}>
                  {viz.type === 'interactive' ? '🎛️' : '📊'} {viz.component}
                </div>
              </VisualizationContainer>
            ))}
          </ToolSection>

          <ToolSection>
            <ToolSectionTitle $show={!toolsCollapsed}>Quick Actions</ToolSectionTitle>
            <ControlButton onClick={() => setVisibleTiers({ tier1: true, tier2: true, tier3: true })}>
              <Eye size={16} />
              {!toolsCollapsed && 'Show All'}
            </ControlButton>
            <ControlButton onClick={() => setVisibleTiers({ tier1: true, tier2: false, tier3: false })}>
              <EyeOff size={16} />
              {!toolsCollapsed && 'Hide Details'}
            </ControlButton>
          </ToolSection>
        </ToolsContent>
      </ToolsPanel>
    </LearningContainer>
  );
};

export default LearningInterface;
