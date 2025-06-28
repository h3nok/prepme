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

const Input = styled.input`
  padding: ${props => props.theme.spacing.sm};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text};
  font-size: 0.9rem;
  min-width: 200px;
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

const PromptContainer = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const PromptText = styled.div`
  font-size: 1.1rem;
  font-weight: 500;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const PromptTitle = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
`;

const PredictionContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const PredictionCard = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
`;

const PredictionTitle = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  text-align: center;
`;

const TokenList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.sm};
`;

const TokenItem = styled(motion.div)<{ $rank: number }>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.sm};
  background: ${props => {
    const intensity = Math.max(0.1, 1 - props.$rank * 0.15);
    return `rgba(5, 150, 105, ${intensity})`;
  }};
  border-radius: ${props => props.theme.radii.sm};
  border-left: 4px solid ${props => props.theme.colors.primary};
`;

const TokenText = styled.span`
  font-weight: 500;
  color: ${props => props.theme.colors.text};
`;

const TokenProbability = styled.span`
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 600;
`;

const ProbabilityBar = styled.div`
  width: 100%;
  height: 20px;
  background: ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.sm};
  overflow: hidden;
  margin-top: ${props => props.theme.spacing.sm};
`;

const ProbabilityFill = styled(motion.div)<{ $probability: number }>`
  height: 100%;
  background: linear-gradient(90deg, #059669, #10b981);
  width: ${props => props.$probability * 100}%;
`;

const Explanation = styled.div`
  background: ${props => props.theme.colors.surface};
  border-left: 4px solid ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
`;

const NextTokenPrediction: React.FC<{
  data: {
    prompt: string;
    showProbabilities: boolean;
    showTopK: number;
  };
  controls?: Array<{
    id: string;
    type: string;
    label: string;
    defaultValue?: any;
  }>;
}> = ({ data, controls }) => {
  const [prompt, setPrompt] = useState(data.prompt);
  const [showProbabilities, setShowProbabilities] = useState(data.showProbabilities);
  const [topK, setTopK] = useState(data.showTopK);

  // Simulate token predictions based on the prompt
  const getPredictions = (input: string): Array<{ token: string; probability: number }> => {
    const predictions = [
      { token: 'is', probability: 0.35 },
      { token: 'will', probability: 0.25 },
      { token: 'can', probability: 0.15 },
      { token: 'may', probability: 0.10 },
      { token: 'should', probability: 0.08 },
      { token: 'could', probability: 0.04 },
      { token: 'might', probability: 0.02 },
      { token: 'would', probability: 0.01 }
    ];

    // Adjust predictions based on input context
    if (input.toLowerCase().includes('future')) {
      predictions[0] = { token: 'will', probability: 0.40 };
      predictions[1] = { token: 'is', probability: 0.30 };
    }
    if (input.toLowerCase().includes('artificial intelligence')) {
      predictions[0] = { token: 'will', probability: 0.45 };
      predictions[1] = { token: 'is', probability: 0.25 };
    }

    return predictions.slice(0, topK);
  };

  const predictions = getPredictions(prompt);

  return (
    <Container>
      <Controls>
        {controls?.map(control => {
          if (control.type === 'input') {
            return (
              <Control key={control.id}>
                <Label>{control.label}</Label>
                <Input
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt..."
                />
              </Control>
            );
          }
          if (control.type === 'slider') {
            return (
              <Control key={control.id}>
                <Label>{control.label}</Label>
                <Slider
                  type="range"
                  min={1}
                  max={10}
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                />
                <span style={{ fontSize: '0.8rem', color: '#666' }}>{topK}</span>
              </Control>
            );
          }
          if (control.type === 'toggle') {
            return (
              <Toggle key={control.id}>
                <Checkbox
                  type="checkbox"
                  checked={showProbabilities}
                  onChange={(e) => setShowProbabilities(e.target.checked)}
                />
                {control.label}
              </Toggle>
            );
          }
          return null;
        })}
      </Controls>

      <PromptContainer>
        <PromptTitle>Input Prompt</PromptTitle>
        <PromptText>{prompt}</PromptText>
      </PromptContainer>

      <PredictionContainer>
        <PredictionCard>
          <PredictionTitle>Top {topK} Predictions</PredictionTitle>
          <TokenList>
            <AnimatePresence>
              {predictions.map((pred, index) => (
                <TokenItem
                  key={pred.token}
                  $rank={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <TokenText>{pred.token}</TokenText>
                  {showProbabilities && (
                    <TokenProbability>{(pred.probability * 100).toFixed(1)}%</TokenProbability>
                  )}
                </TokenItem>
              ))}
            </AnimatePresence>
          </TokenList>
        </PredictionCard>

        <PredictionCard>
          <PredictionTitle>Probability Distribution</PredictionTitle>
          <TokenList>
            {predictions.map((pred, index) => (
              <div key={pred.token}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ fontSize: '0.9rem' }}>{pred.token}</span>
                  <span style={{ fontSize: '0.8rem', color: '#666' }}>
                    {(pred.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <ProbabilityBar>
                  <ProbabilityFill
                    $probability={pred.probability}
                    initial={{ width: 0 }}
                    animate={{ width: `${pred.probability * 100}%` }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  />
                </ProbabilityBar>
              </div>
            ))}
          </TokenList>
        </PredictionCard>
      </PredictionContainer>

      <Explanation>
        <h4>Next Token Prediction</h4>
        <p>
          Next token prediction is the core pre-training objective for language models. 
          Given a sequence of tokens, the model predicts the most likely next token 
          based on learned patterns from training data.
        </p>
        <p>
          <strong>How it works:</strong>
        </p>
        <ul>
          <li><strong>Context Understanding:</strong> The model analyzes the input prompt to understand context</li>
          <li><strong>Pattern Recognition:</strong> It identifies learned patterns from training data</li>
          <li><strong>Probability Distribution:</strong> Outputs a probability distribution over all possible next tokens</li>
          <li><strong>Selection:</strong> The highest probability token is typically selected (or sampled with temperature)</li>
        </ul>
        <p>
          <strong>Training Objective:</strong> L = -Σ log P(token_i | token_&lt;i)
        </p>
        <p>
          This simple objective teaches the model syntax, semantics, reasoning, and world knowledge 
          through exposure to massive amounts of text data.
        </p>
      </Explanation>
    </Container>
  );
};

export default NextTokenPrediction; 