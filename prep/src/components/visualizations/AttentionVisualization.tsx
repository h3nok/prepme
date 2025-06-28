import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

const Container = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.md} 0;
`;

const SentenceContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
  justify-content: center;
`;

const Token = styled(motion.div)<{ $isFocused: boolean; $attentionWeight: number }>`
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => 
    props.$isFocused 
      ? `rgba(124, 58, 237, ${0.2 + props.$attentionWeight * 0.6})`
      : `rgba(124, 58, 237, ${props.$attentionWeight * 0.3})`
  };
  border: 2px solid ${props => 
    props.$isFocused 
      ? props.theme.colors.primary 
      : `rgba(124, 58, 237, ${props.$attentionWeight * 0.5})`
  };
  border-radius: ${props => props.theme.radii.md};
  font-weight: ${props => props.$isFocused ? '600' : '400'};
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
  }
`;

const AttentionWeight = styled.div`
  position: absolute;
  top: -8px;
  right: -8px;
  background: ${props => props.theme.colors.primary};
  color: white;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
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

const Select = styled.select`
  padding: ${props => props.theme.spacing.sm};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text};
  font-size: 0.9rem;
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

const Explanation = styled.div`
  background: ${props => props.theme.colors.surface};
  border-left: 4px solid ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
`;

const AttentionVisualization: React.FC<{
  data: {
    sentence: string;
    focusWord: string;
    showWeights: boolean;
  };
  controls?: Array<{
    id: string;
    type: string;
    label: string;
    options?: string[];
    defaultValue?: any;
  }>;
}> = ({ data, controls }) => {
  const [focusWord, setFocusWord] = useState(data.focusWord);
  const [showWeights, setShowWeights] = useState(data.showWeights);
  
  const tokens = data.sentence.split(' ');
  
  // Simulate attention weights based on word relationships
  const getAttentionWeight = (token: string, focus: string): number => {
    if (token === focus) return 1.0;
    
    // Simple heuristics for demonstration
    const relationships: Record<string, Record<string, number>> = {
      'The': { 'cat': 0.8, 'mat': 0.6, 'sat': 0.3, 'on': 0.4, 'the': 0.9 },
      'cat': { 'The': 0.8, 'sat': 0.9, 'on': 0.7, 'the': 0.6, 'mat': 0.5 },
      'sat': { 'cat': 0.9, 'on': 0.8, 'the': 0.6, 'mat': 0.7, 'The': 0.3 },
      'on': { 'sat': 0.8, 'the': 0.9, 'mat': 0.8, 'cat': 0.7, 'The': 0.4 },
      'the': { 'on': 0.9, 'mat': 0.8, 'sat': 0.6, 'cat': 0.6, 'The': 0.9 },
      'mat': { 'the': 0.8, 'on': 0.8, 'sat': 0.7, 'cat': 0.5, 'The': 0.6 }
    };
    
    return relationships[focus]?.[token] || 0.1;
  };

  const handleTokenClick = (token: string) => {
    setFocusWord(token);
  };

  return (
    <Container>
      <Controls>
        {controls?.map(control => {
          if (control.type === 'dropdown') {
            return (
              <Control key={control.id}>
                <Label>{control.label}</Label>
                <Select
                  value={focusWord}
                  onChange={(e) => setFocusWord(e.target.value)}
                >
                  {control.options?.map(option => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </Select>
              </Control>
            );
          }
          if (control.type === 'toggle') {
            return (
              <Toggle key={control.id}>
                <Checkbox
                  type="checkbox"
                  checked={showWeights}
                  onChange={(e) => setShowWeights(e.target.checked)}
                />
                {control.label}
              </Toggle>
            );
          }
          return null;
        })}
      </Controls>

      <SentenceContainer>
        <AnimatePresence>
          {tokens.map((token, index) => {
            const attentionWeight = getAttentionWeight(token, focusWord);
            const isFocused = token === focusWord;
            
            return (
              <Token
                key={`${token}-${index}`}
                $isFocused={isFocused}
                $attentionWeight={attentionWeight}
                onClick={() => handleTokenClick(token)}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {token}
                {showWeights && (
                  <AttentionWeight>
                    {(attentionWeight * 100).toFixed(0)}
                  </AttentionWeight>
                )}
              </Token>
            );
          })}
        </AnimatePresence>
      </SentenceContainer>

      <Explanation>
        <h4>How Attention Works</h4>
        <p>
          When the model processes the word <strong>"{focusWord}"</strong>, it computes attention weights 
          to determine how much to "pay attention" to each other word. Higher weights (darker colors) 
          indicate stronger attention. Click on different words to see how attention patterns change!
        </p>
        <p>
          <strong>Key Insight:</strong> Attention allows the model to create context-aware representations 
          by dynamically focusing on the most relevant parts of the input sequence.
        </p>
      </Explanation>
    </Container>
  );
};

export default AttentionVisualization; 