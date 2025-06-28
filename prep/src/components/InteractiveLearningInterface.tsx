import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { 
  ChevronLeft, 
  ChevronRight, 
  Play, 
  Pause, 
  RotateCcw, 
  Settings, 
  BookOpen,
  Code,
  Brain,
  Lightbulb,
  Target,
  CheckCircle,
  Circle,
  ArrowLeft,
  ArrowRight,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Menu,
  X,
  Clock,
  Award,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  Calculator,
  Beaker,
  MessageSquare,
  Star
} from 'lucide-react';

// Import types
import { LearningModule, Concept, Slide } from '../types/LearningModule';

// Import modules
import { llmsModule } from '../data/LLMsModule';
import { transformerModule } from '../data/TransformerModule';
import { diffusionModule } from '../data/DiffusionModule';
import { fundamentalsModule } from '../data/FundamentalsModule';

// Import visualizations
import AttentionVisualization from './visualizations/AttentionVisualization';
import ScalingLaws from './visualizations/ScalingLaws';
import MultiHeadAttention from './visualizations/MultiHeadAttention';

// Main container
const LearningContainer = styled.div`
  display: flex;
  height: 100vh;
  background: ${props => props.theme.colors.background};
  position: relative;
`;

// Left navigation panel
const NavigationPanel = styled(motion.div)<{ $isCollapsed: boolean }>`
  width: ${props => props.$isCollapsed ? '60px' : '320px'};
  height: 100vh;
  background: linear-gradient(135deg, 
    ${props => props.theme.colors.surface}f0, 
    ${props => props.theme.colors.background}f0
  );
  backdrop-filter: blur(20px);
  border-right: 1px solid ${props => props.theme.colors.border};
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  z-index: 100;
  box-shadow: ${props => props.theme.shadows.lg};
`;

const NavHeader = styled.div`
  padding: ${props => props.theme.spacing.lg};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const NavTitle = styled.h2<{ $show: boolean }>`
  color: ${props => props.theme.colors.text};
  font-size: 1.2rem;
  font-weight: 700;
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
`;

const CollapseButton = styled.button`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  padding: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.md};
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
    transform: scale(1.05);
  }
`;

const ConceptList = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.lg};
  overflow-y: auto;
`;

const ConceptItem = styled(motion.div)<{ $active: boolean; $completed: boolean }>`
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.lg};
  cursor: pointer;
  background: ${props => 
    props.$active 
      ? `linear-gradient(135deg, ${props.theme.colors.primary}20, ${props.theme.colors.primary}10)`
      : props.$completed
        ? `${props.theme.colors.success}10`
        : props.theme.colors.surface
  };
  border: 1px solid ${props => 
    props.$active 
      ? props.theme.colors.primary
      : props.$completed
        ? props.theme.colors.success
        : props.theme.colors.border
  };
  transition: all 0.3s ease;

  &:hover {
    background: ${props => props.theme.colors.primary}10;
    border-color: ${props => props.theme.colors.primary};
    transform: translateX(5px);
  }
`;

const ConceptTitle = styled.h4<{ $show: boolean }>`
  color: ${props => props.theme.colors.text};
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.xs};
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
  background: ${props => props.theme.colors.border};
  border-radius: 2px;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $progress: number }>`
  width: ${props => props.$progress}%;
  height: 100%;
  background: linear-gradient(90deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.accent});
  transition: width 0.3s ease;
`;

// Main content area
const ContentPanel = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: ${props => props.theme.colors.background};
  position: relative;
`;

const ContentHeader = styled.div`
  padding: ${props => props.theme.spacing.lg} ${props => props.theme.spacing.xl};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, 
    ${props => props.theme.colors.surface}80, 
    ${props => props.theme.colors.background}80
  );
  backdrop-filter: blur(20px);
`;

const ContentTitle = styled.h1`
  color: ${props => props.theme.colors.text};
  font-size: 1.8rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const SlideControls = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const ControlButton = styled.button<{ $variant?: 'primary' | 'secondary' }>`
  background: ${props => 
    props.$variant === 'primary' 
      ? props.theme.colors.primary 
      : props.theme.colors.surface
  };
  border: 1px solid ${props => 
    props.$variant === 'primary' 
      ? props.theme.colors.primary 
      : props.theme.colors.border
  };
  color: ${props => 
    props.$variant === 'primary' 
      ? 'white' 
      : props.theme.colors.text
  };
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.radii.md};
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  font-weight: 500;

  &:hover {
    background: ${props => 
      props.$variant === 'primary' 
        ? props.theme.colors.accent 
        : props.theme.colors.primary
    };
    color: white;
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.md};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

const SlideProgress = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
`;

// Slide content area
const SlideContent = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.xl};
  overflow-y: auto;
  max-width: 1000px;
  margin: 0 auto;
  width: 100%;
`;

const SlideContainer = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  padding: ${props => props.theme.spacing.xl};
  margin-bottom: ${props => props.theme.spacing.lg};
  border: 1px solid ${props => props.theme.colors.border};
  box-shadow: ${props => props.theme.shadows.lg};
`;

const SlideTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: ${props => props.theme.spacing.lg};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const TierSection = styled(motion.div)<{ $expanded: boolean }>`
  margin-bottom: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.lg};
  overflow: hidden;
  border: 1px solid ${props => props.theme.colors.border};
`;

const TierHeader = styled.div<{ $color: string }>`
  background: linear-gradient(135deg, ${props => props.$color}20, ${props => props.$color}10);
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.lg};
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  transition: all 0.3s ease;

  &:hover {
    background: linear-gradient(135deg, ${props => props.$color}30, ${props => props.$color}15);
  }
`;

const TierLabel = styled.div<{ $color: string }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
  
  span {
    background: ${props => props.$color};
    color: white;
    padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.md};
    border-radius: ${props => props.theme.radii.lg};
    font-size: 0.85rem;
    font-weight: 600;
  }
  
  h3 {
    color: ${props => props.theme.colors.text};
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0;
  }
`;

const TierContent = styled(motion.div)`
  padding: ${props => props.theme.spacing.lg};
  background: ${props => props.theme.colors.background};
`;

const TierText = styled.div`
  color: ${props => props.theme.colors.text};
  font-size: 1.1rem;
  line-height: 1.7;
  
  h3 {
    color: ${props => props.theme.colors.text};
    font-size: 1.3rem;
    font-weight: 600;
    margin: ${props => props.theme.spacing.lg} 0 ${props => props.theme.spacing.md} 0;
    display: flex;
    align-items: center;
    gap: ${props => props.theme.spacing.sm};
  }
  
  h4 {
    color: ${props => props.theme.colors.primary};
    font-size: 1.1rem;
    font-weight: 600;
    margin: ${props => props.theme.spacing.md} 0 ${props => props.theme.spacing.sm} 0;
  }
  
  p {
    margin-bottom: ${props => props.theme.spacing.md};
    line-height: 1.8;
  }
  
  ul, ol {
    margin: ${props => props.theme.spacing.md} 0;
    padding-left: ${props => props.theme.spacing.xl};
  }
  
  li {
    margin-bottom: ${props => props.theme.spacing.sm};
    line-height: 1.6;
  }
  
  strong {
    color: ${props => props.theme.colors.primary};
    font-weight: 600;
  }
  
  em {
    color: ${props => props.theme.colors.accent};
    font-style: italic;
  }
  
  .code-block {
    background: ${props => props.theme.colors.surface};
    border: 1px solid ${props => props.theme.colors.border};
    border-radius: ${props => props.theme.radii.lg};
    padding: ${props => props.theme.spacing.lg};
    margin: ${props => props.theme.spacing.lg} 0;
    overflow-x: auto;
    
    h4 {
      margin-top: 0;
      color: ${props => props.theme.colors.text};
    }
    
    pre {
      background: ${props => props.theme.colors.background};
      border-radius: ${props => props.theme.radii.md};
      padding: ${props => props.theme.spacing.md};
      overflow-x: auto;
      font-family: ${props => props.theme.fonts.mono};
      font-size: 0.9rem;
      line-height: 1.5;
      margin: 0;
    }
    
    code {
      font-family: ${props => props.theme.fonts.mono};
      font-size: 0.9rem;
    }
  }
`;

// Code block styling
const CodeBlock = styled.div`
  margin: ${props => props.theme.spacing.lg} 0;
  border-radius: ${props => props.theme.radii.lg};
  overflow: hidden;
  border: 1px solid ${props => props.theme.colors.border};
`;

const CodeHeader = styled.div`
  background: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: flex;
  align-items: center;
  justify-content: between;
`;

const CodeTitle = styled.span`
  color: ${props => props.theme.colors.text};
  font-size: 0.9rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
`;

// Interactive elements
const InteractiveSection = styled.div`
  margin: ${props => props.theme.spacing.xl} 0;
  padding: ${props => props.theme.spacing.lg};
  background: linear-gradient(135deg, 
    ${props => props.theme.colors.primary}10, 
    ${props => props.theme.colors.accent}10
  );
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.primary}30;
`;

const InteractiveTitle = styled.h3`
  color: ${props => props.theme.colors.text};
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.md};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const InteractiveGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-top: ${props => props.theme.spacing.lg};
`;

const InteractiveCard = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.lg};
  border: 1px solid ${props => props.theme.colors.border};
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-5px);
    box-shadow: ${props => props.theme.shadows.xl};
  }
`;

const CardTitle = styled.h4`
  color: ${props => props.theme.colors.text};
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.sm};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
`;

const CardDescription = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.95rem;
  line-height: 1.6;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const ActionButton = styled.button`
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.radii.md};
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};

  &:hover {
    background: ${props => props.theme.colors.accent};
    transform: translateY(-2px);
  }
`;

// Right tools panel
const ToolsPanel = styled(motion.div)<{ $isCollapsed: boolean }>`
  width: ${props => props.$isCollapsed ? '60px' : '280px'};
  height: 100vh;
  background: linear-gradient(135deg, 
    ${props => props.theme.colors.surface}f0, 
    ${props => props.theme.colors.background}f0
  );
  backdrop-filter: blur(20px);
  border-left: 1px solid ${props => props.theme.colors.border};
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  z-index: 100;
  box-shadow: ${props => props.theme.shadows.lg};
`;

const ToolsHeader = styled.div`
  padding: ${props => props.theme.spacing.lg};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const ToolsList = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.lg};
  overflow-y: auto;
`;

const ToolItem = styled.div`
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.sm};
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    background: ${props => props.theme.colors.primary}10;
    border-color: ${props => props.theme.colors.primary};
    transform: translateX(-5px);
  }
`;

const ToolTitle = styled.h4<{ $show: boolean }>`
  color: ${props => props.theme.colors.text};
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.xs};
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
`;

const ToolDescription = styled.p<{ $show: boolean }>`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.8rem;
  line-height: 1.4;
  opacity: ${props => props.$show ? 1 : 0};
  transition: opacity 0.3s ease;
`;

// Component
const InteractiveLearningInterface: React.FC = () => {
  const { moduleId } = useParams<{ moduleId: string }>();
  const navigate = useNavigate();
  
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [currentConceptIndex, setCurrentConceptIndex] = useState(0);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [expandedTiers, setExpandedTiers] = useState<{ [key: string]: boolean }>({
    tier1: true,
    tier2: false,
    tier3: false
  });
  const [isPlaying, setIsPlaying] = useState(false);

  // Get the current module
  const modules = {
    'llms': llmsModule,
    'transformers': transformerModule,
    'diffusion': diffusionModule,
    'fundamentals': fundamentalsModule
  };
  
  const currentModule = modules[moduleId as keyof typeof modules];
  const currentConcept = currentModule?.concepts[currentConceptIndex];
  const currentSlide = currentConcept?.slides[currentSlideIndex];

  const nextSlide = () => {
    if (currentConcept && currentSlideIndex < currentConcept.slides.length - 1) {
      setCurrentSlideIndex(prev => prev + 1);
    } else if (currentModule && currentConceptIndex < currentModule.concepts.length - 1) {
      setCurrentConceptIndex(prev => prev + 1);
      setCurrentSlideIndex(0);
    }
  };

  const prevSlide = () => {
    if (currentSlideIndex > 0) {
      setCurrentSlideIndex(prev => prev - 1);
    } else if (currentConceptIndex > 0) {
      setCurrentConceptIndex(prev => prev - 1);
      setCurrentSlideIndex(0);
    }
  };

  // Auto-play functionality
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isPlaying && currentModule) {
      interval = setInterval(() => {
        nextSlide();
      }, 10000); // 10 seconds per slide
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, currentConceptIndex, currentSlideIndex, currentModule]);
  if (!currentModule) {
    return (
      <LearningContainer>
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <h2>Module not found</h2>
          <button onClick={() => navigate('/')}>Go Home</button>
        </div>
      </LearningContainer>
    );
  }

  if (!currentConcept || !currentSlide) {
    return (
      <LearningContainer>
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <h2>No content available</h2>
          <button onClick={() => navigate('/')}>Go Home</button>
        </div>
      </LearningContainer>
    );
  }

  const toggleTier = (tier: string) => {
    setExpandedTiers(prev => ({
      ...prev,
      [tier]: !prev[tier]
    }));
  };

  const totalSlides = currentModule.concepts.reduce((acc, concept) => acc + concept.slides.length, 0);
  const currentSlideNumber = currentModule.concepts
    .slice(0, currentConceptIndex)
    .reduce((acc, concept) => acc + concept.slides.length, 0) + currentSlideIndex + 1;

  const renderCodeBlock = (code: string, language: string, title?: string) => (
    <CodeBlock>
      <CodeHeader>
        <CodeTitle>
          <Code size={16} />
          {title || `${language.toUpperCase()} Code`}
        </CodeTitle>
      </CodeHeader>
      <SyntaxHighlighter
        language={language}
        style={tomorrow}
        customStyle={{
          margin: 0,
          background: 'transparent',
          fontSize: '0.9rem',
          lineHeight: '1.5',
          padding: '1rem'
        }}
        showLineNumbers={true}
        wrapLines={true}
      >
        {code}
      </SyntaxHighlighter>
    </CodeBlock>
  );

  const renderEnhancedCodeExample = () => {
    if (currentSlide.id === 'attention-coding-challenge') {
      return (
        <>
          {renderCodeBlock(`import numpy as np

def attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k) 
        V: Value matrix (seq_len, d_v)
        mask: Optional attention mask
    
    Returns:
        output: Attention output (seq_len, d_v)
        weights: Attention weights (seq_len, seq_len)
    """
    # Step 1: Compute attention scores (Q @ K^T)
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose())
    
    # Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
    scores = scores / (d_k ** 0.5)
    
    # Step 3: Apply mask if provided (for padding or causal attention)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Step 4: Apply softmax to get attention weights
    # Subtract max for numerical stability
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Step 5: Apply attention weights to values
    output = np.matmul(weights, V)
    
    return output, weights

# Example usage
def demo_attention():
    """Demonstrate attention with a simple example."""
    # Create example embeddings for sentence: "The cat sat"
    seq_len, d_model = 3, 4
    
    # Simple embeddings (in practice, these come from an embedding layer)
    embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4],  # "The"
        [0.5, 0.6, 0.7, 0.8],  # "cat" 
        [0.9, 1.0, 1.1, 1.2]   # "sat"
    ])
    
    # For simplicity, use the same embeddings as Q, K, V
    # (in practice, these would be different linear projections)
    Q = K = V = embeddings
    
    # Compute attention
    output, weights = attention(Q, K, V)
    
    print("Input tokens: ['The', 'cat', 'sat']")
    print("\\nAttention weights:")
    print("(rows=queries, cols=keys)")
    print(weights.round(3))
    
    print("\\nContextualized representations:")
    print(output.round(3))
    
    return output, weights

# Run the demo
if __name__ == "__main__":
    demo_attention()`, 'python', 'Complete Attention Implementation')}
          
          {renderCodeBlock(`# Advanced Multi-Head Attention Implementation
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Multi-head attention implementation.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, X, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            X: Input tensor (seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Multi-head attention output
            attention_weights: All attention weights
        """
        seq_len, d_model = X.shape
        
        # Linear projections
        Q = X @ self.W_Q  # (seq_len, d_model)
        K = X @ self.W_K  # (seq_len, d_model)
        V = X @ self.W_V  # (seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        # Apply attention for each head
        attention_outputs = []
        attention_weights = []
        
        for head in range(self.num_heads):
            output, weights = self.scaled_dot_product_attention(
                Q[head], K[head], V[head], mask
            )
            attention_outputs.append(output)
            attention_weights.append(weights)
        
        # Concatenate heads
        concat_output = np.concatenate(attention_outputs, axis=-1)
        
        # Final linear transformation
        final_output = concat_output @ self.W_O
        
        return final_output, np.array(attention_weights)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Single-head attention computation."""
        scores = Q @ K.transpose()
        scores = scores / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -np.inf, scores)
        
        # Softmax
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores_shifted)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        
        output = weights @ V
        return output, weights

# Example usage
def demo_multihead_attention():
    """Demonstrate multi-head attention."""
    d_model, num_heads, seq_len = 8, 2, 4
    
    # Create example input
    X = np.random.randn(seq_len, d_model)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attention_weights = mha.forward(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of heads: {num_heads}")
    
    return output, attention_weights

# Run demo
demo_multihead_attention()`, 'python', 'Multi-Head Attention Implementation')}
        </>
      );
    }

    // Default attention implementation for other slides
    if (currentSlide.id === 'attention-mechanism') {
      return renderCodeBlock(`import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        return torch.matmul(attention_weights, V)
            `, 'python', 'PyTorch Multi-Head Attention Implementation');
    }

    return null;
  };

  const renderInteractiveElements = () => {
    if (!currentSlide?.interactiveElements?.length) return null;

    return (
      <InteractiveSection>
        <InteractiveTitle>
          <Brain size={20} />
          Interactive Learning
        </InteractiveTitle>
        <InteractiveGrid>
          {currentSlide.interactiveElements.map((element, index) => (
            <InteractiveCard key={index}>
              <CardTitle>
                {element.type === 'calculator' && <Calculator size={16} />}
                {element.type === 'simulator' && <Beaker size={16} />}
                {element.type === 'playground' && <Code size={16} />}
                {element.type === 'quiz' && <MessageSquare size={16} />}
                {element.type === 'calculator' && 'Calculator'}
                {element.type === 'simulator' && 'Simulator'}
                {element.type === 'playground' && 'Code Playground'}
                {element.type === 'quiz' && 'Knowledge Check'}
              </CardTitle>
              <CardDescription>
                {element.type === 'calculator' && 'Interactive calculator for computational concepts'}
                {element.type === 'simulator' && 'Hands-on simulation environment'}
                {element.type === 'playground' && 'Code editor with live execution'}
                {element.type === 'quiz' && 'Quick quiz to test your understanding'}
              </CardDescription>
              <ActionButton>
                <Play size={14} />
                Launch {element.type}
              </ActionButton>
            </InteractiveCard>
          ))}
        </InteractiveGrid>
      </InteractiveSection>
    );
  };

  const renderVisualizations = () => {
    if (!currentSlide?.visualizations?.length) return null;

    return (
      <InteractiveSection>
        <InteractiveTitle>
          <Target size={20} />
          Visualizations
        </InteractiveTitle>
        <InteractiveGrid>
          {currentSlide.visualizations.map((viz, index) => (
            <InteractiveCard key={index}>
              <CardTitle>
                <Eye size={16} />
                {viz.component} Visualization
              </CardTitle>
              <CardDescription>
                Interactive {viz.type} visualization to help understand the concept
              </CardDescription>
              <div style={{ marginTop: '1rem', minHeight: '200px' }}>
                {viz.component === 'AttentionVisualization' && (
                  <AttentionVisualization 
                    data={{
                      sentence: "The quick brown fox jumps over the lazy dog",
                      focusWord: "fox",
                      showWeights: true
                    }}
                  />
                )}
                {viz.component === 'ScalingLaws' && (
                  <ScalingLaws 
                    data={{
                      models: ["GPT-1", "GPT-2", "GPT-3", "GPT-4"],
                      showOptimal: true
                    }}
                  />
                )}
                {viz.component === 'MultiHeadAttention' && (
                  <MultiHeadAttention 
                    data={{
                      sentence: "Attention is all you need",
                      numHeads: 8,
                      showSpecializations: true
                    }}
                  />
                )}
              </div>
            </InteractiveCard>
          ))}
        </InteractiveGrid>
      </InteractiveSection>
    );
  };

  return (
    <LearningContainer>
      {/* Left Navigation Panel */}
      <NavigationPanel $isCollapsed={leftCollapsed}>
        <NavHeader>
          <NavTitle $show={!leftCollapsed}>{currentModule.title}</NavTitle>
          <CollapseButton onClick={() => setLeftCollapsed(!leftCollapsed)}>
            {leftCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </CollapseButton>
        </NavHeader>
        
        <ConceptList>
          {currentModule.concepts.map((concept, index) => (
            <ConceptItem
              key={concept.id}
              $active={index === currentConceptIndex}
              $completed={index < currentConceptIndex}
              onClick={() => {
                setCurrentConceptIndex(index);
                setCurrentSlideIndex(0);
              }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <ConceptTitle $show={!leftCollapsed}>
                {index < currentConceptIndex ? (
                  <CheckCircle size={16} style={{ color: '#10b981' }} />
                ) : (
                  <Circle size={16} />
                )}
                {concept.title}
              </ConceptTitle>
              <ConceptProgress $show={!leftCollapsed}>
                <ProgressBar>
                  <ProgressFill 
                    $progress={index === currentConceptIndex ? 
                      ((currentSlideIndex + 1) / concept.slides.length) * 100 : 
                      index < currentConceptIndex ? 100 : 0
                    } 
                  />
                </ProgressBar>
              </ConceptProgress>
            </ConceptItem>
          ))}
        </ConceptList>
      </NavigationPanel>

      {/* Main Content Panel */}
      <ContentPanel>
        <ContentHeader>
          <ContentTitle>
            <BookOpen size={24} />
            {currentSlide.title}
          </ContentTitle>
          
          <SlideControls>
            <SlideProgress>
              <Clock size={16} />
              {currentSlideNumber} / {totalSlides}
            </SlideProgress>
            
            <ControlButton onClick={() => setIsPlaying(!isPlaying)} $variant="secondary">
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              {isPlaying ? 'Pause' : 'Play'}
            </ControlButton>
            
            <ControlButton onClick={prevSlide} disabled={currentConceptIndex === 0 && currentSlideIndex === 0}>
              <ArrowLeft size={16} />
              Previous
            </ControlButton>
            
            <ControlButton 
              onClick={nextSlide} 
              disabled={currentConceptIndex === currentModule.concepts.length - 1 && 
                       currentSlideIndex === currentConcept.slides.length - 1}
              $variant="primary"
            >
              Next
              <ArrowRight size={16} />
            </ControlButton>
          </SlideControls>
        </ContentHeader>

        <SlideContent>
          <SlideContainer
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <SlideTitle>
              <Lightbulb size={28} />
              {currentSlide.title}
            </SlideTitle>

            {/* Tier 1 - Core Concept */}
            <TierSection $expanded={expandedTiers.tier1}>
              <TierHeader $color="#ff6b35" onClick={() => toggleTier('tier1')}>
                <TierLabel $color="#ff6b35">
                  <span>Core Concept</span>
                  <h3>Essential Understanding</h3>
                </TierLabel>
                {expandedTiers.tier1 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
              </TierHeader>
              <AnimatePresence>
                {expandedTiers.tier1 && (
                  <TierContent
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <TierText dangerouslySetInnerHTML={{ __html: currentSlide.content.tier1 }} />
                  </TierContent>
                )}
              </AnimatePresence>
            </TierSection>

            {/* Tier 2 - Mechanism */}
            {currentSlide.content.tier2 && (
              <TierSection $expanded={expandedTiers.tier2}>
                <TierHeader $color="#059669" onClick={() => toggleTier('tier2')}>
                  <TierLabel $color="#059669">
                    <span>Deep Dive</span>
                    <h3>How It Works</h3>
                  </TierLabel>
                  {expandedTiers.tier2 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </TierHeader>
                <AnimatePresence>
                  {expandedTiers.tier2 && (
                    <TierContent
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <TierText dangerouslySetInnerHTML={{ __html: currentSlide.content.tier2 }} />
                    </TierContent>
                  )}
                </AnimatePresence>
              </TierSection>
            )}

            {/* Tier 3 - Technical Details */}
            {currentSlide.content.tier3 && (
              <TierSection $expanded={expandedTiers.tier3}>
                <TierHeader $color="#7c3aed" onClick={() => toggleTier('tier3')}>
                  <TierLabel $color="#7c3aed">
                    <span>Expert Level</span>
                    <h3>Technical Implementation</h3>
                  </TierLabel>
                  {expandedTiers.tier3 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </TierHeader>
                <AnimatePresence>
                  {expandedTiers.tier3 && (
                    <TierContent
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <TierText dangerouslySetInnerHTML={{ __html: currentSlide.content.tier3 }} />
                    </TierContent>
                  )}
                </AnimatePresence>
              </TierSection>
            )}

            {/* Code Examples */}
            {renderEnhancedCodeExample()}

            {/* Interactive Elements */}
            {renderInteractiveElements()}

            {/* Visualizations */}
            {renderVisualizations()}

            {/* Key Points */}
            {currentSlide.keyPoints && (
              <InteractiveSection>
                <InteractiveTitle>
                  <Star size={20} />
                  Key Takeaways
                </InteractiveTitle>
                <ul style={{ fontSize: '1.1rem', lineHeight: '1.8' }}>
                  {currentSlide.keyPoints.map((point, index) => (
                    <li key={index} style={{ marginBottom: '0.5rem' }}>
                      {point}
                    </li>
                  ))}
                </ul>
              </InteractiveSection>
            )}
          </SlideContainer>
        </SlideContent>
      </ContentPanel>

      {/* Right Tools Panel */}
      <ToolsPanel $isCollapsed={rightCollapsed}>
        <ToolsHeader>
          <NavTitle $show={!rightCollapsed}>Learning Tools</NavTitle>
          <CollapseButton onClick={() => setRightCollapsed(!rightCollapsed)}>
            {rightCollapsed ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </CollapseButton>
        </ToolsHeader>
        
        <ToolsList>
          <ToolItem>
            <ToolTitle $show={!rightCollapsed}>
              <Calculator size={16} />
              Math Calculator
            </ToolTitle>
            <ToolDescription $show={!rightCollapsed}>
              Compute attention scores and model parameters
            </ToolDescription>
          </ToolItem>
          
          <ToolItem>
            <ToolTitle $show={!rightCollapsed}>
              <BookOpen size={16} />
              Glossary
            </ToolTitle>
            <ToolDescription $show={!rightCollapsed}>
              Quick reference for technical terms
            </ToolDescription>
          </ToolItem>
          
          <ToolItem>
            <ToolTitle $show={!rightCollapsed}>
              <Target size={16} />
              Practice Quiz
            </ToolTitle>
            <ToolDescription $show={!rightCollapsed}>
              Test your understanding with interactive questions
            </ToolDescription>
          </ToolItem>
          
          <ToolItem>
            <ToolTitle $show={!rightCollapsed}>
              <TrendingUp size={16} />
              Progress Tracker
            </ToolTitle>
            <ToolDescription $show={!rightCollapsed}>
              Monitor your learning journey
            </ToolDescription>
          </ToolItem>
        </ToolsList>
      </ToolsPanel>
    </LearningContainer>
  );
};

export default InteractiveLearningInterface;
