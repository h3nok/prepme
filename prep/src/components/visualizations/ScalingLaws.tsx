import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

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

const ChartContainer = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const ChartTitle = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  text-align: center;
`;

const Canvas = styled.canvas`
  width: 100%;
  height: 300px;
  border-radius: ${props => props.theme.radii.sm};
`;

const ModelComparison = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.md};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const ModelCard = styled(motion.div)<{ $isOptimal: boolean }>`
  background: ${props => props.$isOptimal 
    ? `rgba(5, 150, 105, 0.1)` 
    : props.theme.colors.background
  };
  border: 2px solid ${props => props.$isOptimal 
    ? props.theme.colors.primary 
    : props.theme.colors.border
  };
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
  text-align: center;
`;

const ModelName = styled.h5`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.sm} 0;
  font-size: 1rem;
`;

const ModelStats = styled.div`
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.4;
`;

const Explanation = styled.div`
  background: ${props => props.theme.colors.surface};
  border-left: 4px solid ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
`;

const ScalingLaws: React.FC<{
  data: {
    models: string[];
    showOptimal: boolean;
  };
  controls?: Array<{
    id: string;
    type: string;
    label: string;
    defaultValue?: any;
  }>;
}> = ({ data, controls }) => {
  const [showOptimal, setShowOptimal] = useState(data.showOptimal);
  const [logScale, setLogScale] = useState(true);

  // Model data with parameters, tokens, and performance
  const modelData = {
    'GPT-3': { params: 175, tokens: 300, performance: 0.85, optimal: false },
    'Chinchilla': { params: 70, tokens: 1400, performance: 0.88, optimal: true },
    'LLaMA': { params: 65, tokens: 1500, performance: 0.87, optimal: true },
    'PaLM': { params: 540, tokens: 780, performance: 0.89, optimal: false }
  };

  // Draw scaling laws chart
  useEffect(() => {
    const canvas = document.getElementById('scalingChart') as HTMLCanvasElement;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const padding = 40;

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Vertical grid lines (log scale)
    const logMin = Math.log10(10);
    const logMax = Math.log10(1000);
    for (let i = 0; i <= 10; i++) {
      const x = padding + (width - 2 * padding) * (i / 10);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (height - 2 * padding) * (i / 5);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    
    // X-axis labels (parameters in billions)
    for (let i = 0; i <= 5; i++) {
      const x = padding + (width - 2 * padding) * (i / 5);
      const value = Math.pow(10, logMin + (logMax - logMin) * (i / 5));
      ctx.fillText(`${value.toFixed(0)}B`, x, height - padding + 20);
    }

    // Y-axis labels (performance)
    for (let i = 0; i <= 5; i++) {
      const y = height - padding - (height - 2 * padding) * (i / 5);
      const value = 0.8 + 0.1 * (i / 5);
      ctx.fillText(value.toFixed(2), padding - 10, y + 4);
    }

    // Draw model points
    Object.entries(modelData).forEach(([name, data]) => {
      const x = padding + (width - 2 * padding) * (Math.log10(data.params) - logMin) / (logMax - logMin);
      const y = height - padding - (height - 2 * padding) * (data.performance - 0.8) / 0.1;
      
      // Draw point
      ctx.fillStyle = data.optimal ? '#059669' : '#7c3aed';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Draw label
      ctx.fillStyle = '#374151';
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(name, x + 8, y + 3);
    });

    // Draw optimal scaling line
    if (showOptimal) {
      ctx.strokeStyle = '#059669';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      
      for (let i = 0; i <= 100; i++) {
        const params = Math.pow(10, logMin + (logMax - logMin) * (i / 100));
        const performance = 0.8 + 0.1 * Math.log10(params / 10) / Math.log10(1000 / 10);
        const x = padding + (width - 2 * padding) * (Math.log10(params) - logMin) / (logMax - logMin);
        const y = height - padding - (height - 2 * padding) * (performance - 0.8) / 0.1;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

  }, [showOptimal, logScale]);

  return (
    <Container>
      <Controls>
        {controls?.map(control => {
          if (control.type === 'toggle') {
            if (control.id === 'show-optimal') {
              return (
                <Toggle key={control.id}>
                  <Checkbox
                    type="checkbox"
                    checked={showOptimal}
                    onChange={(e) => setShowOptimal(e.target.checked)}
                  />
                  {control.label}
                </Toggle>
              );
            }
            if (control.id === 'log-scale') {
              return (
                <Toggle key={control.id}>
                  <Checkbox
                    type="checkbox"
                    checked={logScale}
                    onChange={(e) => setLogScale(e.target.checked)}
                  />
                  {control.label}
                </Toggle>
              );
            }
          }
          return null;
        })}
      </Controls>

      <ChartContainer>
        <ChartTitle>Scaling Laws: Model Size vs Performance</ChartTitle>
        <Canvas id="scalingChart" />
        <p style={{ fontSize: '0.8rem', textAlign: 'center', marginTop: '0.5rem' }}>
          Parameters (Billions) vs Performance Score
        </p>
      </ChartContainer>

      <ModelComparison>
        {Object.entries(modelData).map(([name, data]) => (
          <ModelCard
            key={name}
            $isOptimal={data.optimal}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ModelName>{name}</ModelName>
            <ModelStats>
              <div><strong>Parameters:</strong> {data.params}B</div>
              <div><strong>Training Tokens:</strong> {data.tokens}B</div>
              <div><strong>Performance:</strong> {data.performance}</div>
              {data.optimal && <div style={{ color: '#059669', fontWeight: 'bold' }}>✓ Optimal Scaling</div>}
            </ModelStats>
          </ModelCard>
        ))}
      </ModelComparison>

      <Explanation>
        <h4>Scaling Laws in Language Models</h4>
        <p>
          Scaling laws describe how model performance improves with size, data, and compute. 
          The Chinchilla scaling law shows that optimal model size scales with data size, 
          leading to more efficient training.
        </p>
        <p>
          <strong>Key Insights:</strong>
        </p>
        <ul>
          <li><strong>Optimal Scaling:</strong> Model size should scale with data size (N ∝ D^0.74)</li>
          <li><strong>Sample Efficiency:</strong> Larger models are more sample-efficient</li>
          <li><strong>Compute Budget:</strong> Optimal allocation between model size and training data</li>
          <li><strong>Diminishing Returns:</strong> Performance scales logarithmically with model size</li>
        </ul>
        <p>
          <strong>Formula:</strong> L(N, D) = E + A/N^α + B/D^β where L is loss, N is model size, and D is data size
        </p>
      </Explanation>
    </Container>
  );
};

export default ScalingLaws; 