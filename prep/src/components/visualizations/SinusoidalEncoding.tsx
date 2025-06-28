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

const VisualizationContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ChartContainer = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
`;

const ChartTitle = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  text-align: center;
`;

const Canvas = styled.canvas`
  width: 100%;
  height: 200px;
  border-radius: ${props => props.theme.radii.sm};
`;

const EncodingGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
  gap: 2px;
  margin-top: ${props => props.theme.spacing.md};
`;

const EncodingCell = styled.div<{ $value: number }>`
  aspect-ratio: 1;
  background: ${props => {
    const absValue = Math.abs(props.$value);
    const intensity = Math.min(absValue * 2, 1);
    const hue = props.$value > 0 ? 240 : 0; // Blue for positive, red for negative
    return `hsla(${hue}, 70%, 50%, ${intensity})`;
  }};
  border-radius: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6rem;
  font-weight: 600;
  color: ${props => Math.abs(props.$value) > 0.5 ? 'white' : 'inherit'};
`;

const Explanation = styled.div`
  background: ${props => props.theme.colors.surface};
  border-left: 4px solid ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
`;

const SinusoidalEncoding: React.FC<{
  data: {
    sequenceLength: number;
    embeddingDim: number;
    showFrequencies: boolean;
  };
  controls?: Array<{
    id: string;
    type: string;
    label: string;
    range?: [number, number];
    defaultValue?: any;
  }>;
}> = ({ data, controls }) => {
  const [sequenceLength, setSequenceLength] = useState(data.sequenceLength);
  const [embeddingDim, setEmbeddingDim] = useState(data.embeddingDim);
  const [showFrequencies, setShowFrequencies] = useState(data.showFrequencies);
  const [selectedPosition, setSelectedPosition] = useState(0);

  // Generate sinusoidal positional encodings
  const generatePositionalEncoding = (pos: number, d_model: number): number[] => {
    const encoding = [];
    for (let i = 0; i < d_model; i++) {
      if (i % 2 === 0) {
        // Even dimensions: sin
        encoding.push(Math.sin(pos / Math.pow(10000, 2 * i / d_model)));
      } else {
        // Odd dimensions: cos
        encoding.push(Math.cos(pos / Math.pow(10000, 2 * (i - 1) / d_model)));
      }
    }
    return encoding;
  };

  // Draw sine wave chart
  useEffect(() => {
    const canvas = document.getElementById('sineChart') as HTMLCanvasElement;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = (canvas.offsetWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.offsetHeight);
      ctx.stroke();
    }
    for (let i = 0; i <= 4; i++) {
      const y = (canvas.offsetHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.offsetWidth, y);
      ctx.stroke();
    }

    // Draw sine waves for different frequencies
    const colors = ['#7c3aed', '#059669', '#dc2626', '#0891b2'];
    for (let dim = 0; dim < Math.min(4, embeddingDim); dim++) {
      ctx.strokeStyle = colors[dim];
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let pos = 0; pos < sequenceLength; pos++) {
        const x = (canvas.offsetWidth / sequenceLength) * pos;
        const value = dim % 2 === 0 
          ? Math.sin(pos / Math.pow(10000, 2 * dim / embeddingDim))
          : Math.cos(pos / Math.pow(10000, 2 * (dim - 1) / embeddingDim));
        const y = (canvas.offsetHeight / 2) * (1 - value);

        if (pos === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Highlight selected position
    if (selectedPosition >= 0 && selectedPosition < sequenceLength) {
      const x = (canvas.offsetWidth / sequenceLength) * selectedPosition;
      ctx.fillStyle = '#7c3aed';
      ctx.beginPath();
      ctx.arc(x, canvas.offsetHeight / 2, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [sequenceLength, embeddingDim, selectedPosition]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const pos = Math.floor((x / canvas.offsetWidth) * sequenceLength);
    setSelectedPosition(Math.max(0, Math.min(pos, sequenceLength - 1)));
  };

  const selectedEncoding = generatePositionalEncoding(selectedPosition, embeddingDim);

  return (
    <Container>
      <Controls>
        {controls?.map(control => {
          if (control.type === 'slider') {
            const value = control.id === 'sequence-length' ? sequenceLength : embeddingDim;
            const setValue = control.id === 'sequence-length' ? setSequenceLength : setEmbeddingDim;
            
            return (
              <Control key={control.id}>
                <Label>{control.label}</Label>
                <Slider
                  type="range"
                  min={control.range?.[0]}
                  max={control.range?.[1]}
                  value={value}
                  onChange={(e) => setValue(parseInt(e.target.value))}
                />
                <span style={{ fontSize: '0.8rem', color: '#666' }}>{value}</span>
              </Control>
            );
          }
          if (control.type === 'toggle') {
            return (
              <Toggle key={control.id}>
                <Checkbox
                  type="checkbox"
                  checked={showFrequencies}
                  onChange={(e) => setShowFrequencies(e.target.checked)}
                />
                {control.label}
              </Toggle>
            );
          }
          return null;
        })}
      </Controls>

      <VisualizationContainer>
        <ChartContainer>
          <ChartTitle>Sinusoidal Functions</ChartTitle>
          <Canvas
            id="sineChart"
            onClick={handleCanvasClick}
            style={{ cursor: 'pointer' }}
          />
          <p style={{ fontSize: '0.8rem', textAlign: 'center', marginTop: '0.5rem' }}>
            Click on the chart to select a position
          </p>
        </ChartContainer>

        <ChartContainer>
          <ChartTitle>Position {selectedPosition} Encoding</ChartTitle>
          <EncodingGrid>
            {selectedEncoding.map((value, index) => (
              <EncodingCell key={index} $value={value}>
                {value.toFixed(2)}
              </EncodingCell>
            ))}
          </EncodingGrid>
          <p style={{ fontSize: '0.8rem', textAlign: 'center', marginTop: '0.5rem' }}>
            Blue = positive, Red = negative, Intensity = magnitude
          </p>
        </ChartContainer>
      </VisualizationContainer>

      <Explanation>
        <h4>Sinusoidal Positional Encoding</h4>
        <p>
          The sinusoidal encoding uses different frequencies for different dimensions. 
          Even dimensions use sine functions, odd dimensions use cosine functions. 
          This creates unique position representations that can generalize to longer sequences.
        </p>
        <p>
          <strong>Key Properties:</strong>
        </p>
        <ul>
          <li><strong>Deterministic:</strong> Same position always gets same encoding</li>
          <li><strong>Extrapolation:</strong> Can handle sequences longer than training</li>
          <li><strong>Relative positions:</strong> Can compute relative distances as linear combinations</li>
          <li><strong>Unique:</strong> Each position has a distinct encoding pattern</li>
        </ul>
        <p>
          <strong>Formula:</strong> PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), 
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        </p>
      </Explanation>
    </Container>
  );
};

export default SinusoidalEncoding; 