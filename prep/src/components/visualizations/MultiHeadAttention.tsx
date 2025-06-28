import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

const Container = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.md} 0;
`;

const HeadGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const HeadCard = styled(motion.div)<{ $isActive: boolean }>`
  background: ${props => props.$isActive 
    ? `rgba(124, 58, 237, 0.1)` 
    : props.theme.colors.background
  };
  border: 2px solid ${props => props.$isActive 
    ? props.theme.colors.primary 
    : props.theme.colors.border
  };
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.1);
  }
`;

const HeadTitle = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.sm} 0;
  font-size: 0.9rem;
  font-weight: 600;
`;

const HeadDescription = styled.p`
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
  margin: 0 0 ${props => props.theme.spacing.sm} 0;
  line-height: 1.4;
`;

const AttentionMatrix = styled.div`
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 2px;
  margin-top: ${props => props.theme.spacing.sm};
`;

const MatrixCell = styled.div<{ $weight: number }>`
  width: 20px;
  height: 20px;
  background: ${props => `rgba(124, 58, 237, ${props.$weight})`};
  border-radius: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6rem;
  color: ${props => props.$weight > 0.5 ? 'white' : 'inherit'};
  font-weight: 600;
`;

const Controls = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.lg};
  flex-wrap: wrap;
  justify-content: center;
`;

const Control = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.xs};
`;

const Label = styled.label`
  font-size: 0.9rem;
  font-weight: 500;
  color: ${props => props.theme.colors.textSecondary};
`;

const Slider = styled.input`
  width: 120px;
`;

const Toggle = styled.label`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  cursor: pointer;
  font-size: 0.9rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const Checkbox = styled.input`
  width: 16px;
  height: 16px;
`;

const SentenceDisplay = styled.div`
  text-align: center;
  margin: ${props => props.theme.spacing.lg} 0;
  padding: ${props => props.theme.spacing.md};
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.radii.md};
  border: 1px solid ${props => props.theme.colors.border};
`;

const Explanation = styled.div`
  background: ${props => props.theme.colors.surface};
  border-left: 4px solid ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
`;

const MultiHeadAttention: React.FC<{
  data: {
    sentence: string;
    numHeads: number;
    showSpecializations: boolean;
  };
  controls?: Array<{
    id: string;
    type: string;
    label: string;
    range?: [number, number];
    defaultValue?: any;
  }>;
}> = ({ data, controls }) => {
  const [numHeads, setNumHeads] = useState(data.numHeads);
  const [showSpecializations, setShowSpecializations] = useState(data.showSpecializations);
  const [activeHead, setActiveHead] = useState(0);
  
  const tokens = data.sentence.split(' ');
  
  // Define different attention head specializations
  const headSpecializations = [
    {
      name: 'Syntactic',
      description: 'Focuses on grammatical relationships',
      pattern: [0.9, 0.1, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.1, 0.1]
    },
    {
      name: 'Semantic',
      description: 'Captures meaning and word associations',
      pattern: [0.3, 0.8, 0.4, 0.2, 0.1, 0.1, 0.2, 0.4, 0.8, 0.3]
    },
    {
      name: 'Positional',
      description: 'Attends to nearby words',
      pattern: [0.1, 0.8, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.8, 0.1]
    },
    {
      name: 'Long-range',
      description: 'Connects distant words',
      pattern: [0.7, 0.2, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.2, 0.7]
    },
    {
      name: 'Subject-verb',
      description: 'Links subjects to their verbs',
      pattern: [0.2, 0.9, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.9, 0.2]
    },
    {
      name: 'Object-focus',
      description: 'Highlights objects and complements',
      pattern: [0.1, 0.3, 0.2, 0.8, 0.1, 0.1, 0.8, 0.2, 0.3, 0.1]
    },
    {
      name: 'Article-noun',
      description: 'Connects articles to nouns',
      pattern: [0.9, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.9]
    },
    {
      name: 'Preposition',
      description: 'Focuses on prepositional relationships',
      pattern: [0.1, 0.2, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.2, 0.1]
    }
  ];

  const getAttentionWeights = (headIndex: number): number[] => {
    if (headIndex < headSpecializations.length) {
      return headSpecializations[headIndex].pattern;
    }
    // Generate random pattern for additional heads
    return Array.from({ length: tokens.length }, () => Math.random() * 0.8 + 0.1);
  };

  return (
    <Container>
      <Controls>
        {controls?.map(control => {
          if (control.type === 'slider') {
            return (
              <Control key={control.id}>
                <Label>{control.label}</Label>
                <Slider
                  type="range"
                  min={control.range?.[0]}
                  max={control.range?.[1]}
                  value={numHeads}
                  onChange={(e) => setNumHeads(parseInt(e.target.value))}
                />
                <span style={{ fontSize: '0.8rem', color: '#666' }}>{numHeads}</span>
              </Control>
            );
          }
          if (control.type === 'toggle') {
            return (
              <Toggle key={control.id}>
                <Checkbox
                  type="checkbox"
                  checked={showSpecializations}
                  onChange={(e) => setShowSpecializations(e.target.checked)}
                />
                {control.label}
              </Toggle>
            );
          }
          return null;
        })}
      </Controls>

      <SentenceDisplay>
        <h4>Input Sentence</h4>
        <p style={{ fontSize: '1.1rem', fontWeight: '500' }}>{data.sentence}</p>
      </SentenceDisplay>

      <HeadGrid>
        <AnimatePresence>
          {Array.from({ length: numHeads }, (_, index) => {
            const head = headSpecializations[index] || {
              name: `Head ${index + 1}`,
              description: 'Learned attention pattern',
              pattern: getAttentionWeights(index)
            };
            const isActive = index === activeHead;
            
            return (
              <HeadCard
                key={index}
                $isActive={isActive}
                onClick={() => setActiveHead(index)}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <HeadTitle>{head.name}</HeadTitle>
                {showSpecializations && (
                  <HeadDescription>{head.description}</HeadDescription>
                )}
                <AttentionMatrix>
                  {head.pattern.slice(0, 36).map((weight, cellIndex) => (
                    <MatrixCell key={cellIndex} $weight={weight}>
                      {(weight * 100).toFixed(0)}
                    </MatrixCell>
                  ))}
                </AttentionMatrix>
              </HeadCard>
            );
          })}
        </AnimatePresence>
      </HeadGrid>

      <Explanation>
        <h4>Multi-Head Attention Specialization</h4>
        <p>
          Each attention head learns to specialize in different types of relationships. 
          Click on different heads to see their attention patterns. The matrix shows 
          attention weights between tokens (darker = stronger attention).
        </p>
        <p>
          <strong>Key Insight:</strong> By having multiple heads, the model can simultaneously 
          capture syntactic, semantic, positional, and other types of relationships, 
          creating richer representations than a single attention mechanism.
        </p>
        {activeHead < headSpecializations.length && (
          <p>
            <strong>Active Head:</strong> {headSpecializations[activeHead].name} - 
            {headSpecializations[activeHead].description}
          </p>
        )}
      </Explanation>
    </Container>
  );
};

export default MultiHeadAttention; 