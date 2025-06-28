import React, { useState, useEffect, useRef } from 'react';
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

const ProcessContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ProcessCard = styled.div`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
`;

const ProcessTitle = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  text-align: center;
`;

const Canvas = styled.canvas`
  width: 100%;
  height: 200px;
  border-radius: ${props => props.theme.radii.sm};
  background: ${props => props.theme.colors.background};
`;

const Timeline = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: ${props => props.theme.spacing.md} 0;
  padding: ${props => props.theme.spacing.sm};
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.sm};
`;

const TimelineStep = styled.div<{ $active: boolean; $completed: boolean }>`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: ${props => {
    if (props.$active) return props.theme.colors.primary;
    if (props.$completed) return '#10b981';
    return props.theme.colors.border;
  }};
  border: 2px solid ${props => props.theme.colors.background};
  transition: all 0.3s ease;
`;

const StepInfo = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: ${props => props.theme.spacing.sm};
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const Explanation = styled.div`
  background: ${props => props.theme.colors.surface};
  border-left: 4px solid ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
`;

const DiffusionProcess: React.FC<{
  data: {
    steps: number;
    showForward: boolean;
    showReverse: boolean;
  };
  controls?: Array<{
    id: string;
    type: string;
    label: string;
    defaultValue?: any;
  }>;
}> = ({ data, controls }) => {
  const [steps, setSteps] = useState(data.steps);
  const [showForward, setShowForward] = useState(data.showForward);
  const [showReverse, setShowReverse] = useState(data.showReverse);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const forwardCanvasRef = useRef<HTMLCanvasElement>(null);
  const reverseCanvasRef = useRef<HTMLCanvasElement>(null);

  // Generate noise schedule (linear)
  const generateNoiseSchedule = (numSteps: number) => {
    const schedule = [];
    for (let i = 0; i < numSteps; i++) {
      schedule.push(0.0001 + (0.02 - 0.0001) * (i / (numSteps - 1)));
    }
    return schedule;
  };

  // Draw forward process
  useEffect(() => {
    const canvas = forwardCanvasRef.current;
    if (!canvas || !showForward) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const padding = 20;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding + (width - 2 * padding) * (i / 10);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Draw noise schedule
    const noiseSchedule = generateNoiseSchedule(steps);
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let i = 0; i < steps; i++) {
      const x = padding + (width - 2 * padding) * (i / (steps - 1));
      const y = height - padding - (height - 2 * padding) * (noiseSchedule[i] / 0.02);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw current step indicator
    if (currentStep < steps) {
      const x = padding + (width - 2 * padding) * (currentStep / (steps - 1));
      const y = height - padding - (height - 2 * padding) * (noiseSchedule[currentStep] / 0.02);
      
      ctx.fillStyle = '#dc2626';
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Timestep', width / 2, height - 5);
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Noise Level (β)', 0, 0);
    ctx.restore();

  }, [steps, showForward, currentStep]);

  // Draw reverse process
  useEffect(() => {
    const canvas = reverseCanvasRef.current;
    if (!canvas || !showReverse) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const padding = 20;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding + (width - 2 * padding) * (i / 10);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Draw reverse process (denoising)
    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let i = 0; i < steps; i++) {
      const x = padding + (width - 2 * padding) * (i / (steps - 1));
      const y = padding + (height - 2 * padding) * (i / (steps - 1)); // Denoising goes from noise to clean
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw current step indicator
    if (currentStep < steps) {
      const x = padding + (width - 2 * padding) * (currentStep / (steps - 1));
      const y = padding + (height - 2 * padding) * (currentStep / (steps - 1));
      
      ctx.fillStyle = '#059669';
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Timestep', width / 2, height - 5);
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Image Quality', 0, 0);
    ctx.restore();

  }, [steps, showReverse, currentStep]);

  // Animation loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps - 1) {
          setIsPlaying(false);
          return 0;
        }
        return prev + 1;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, steps]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
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
                  min={10}
                  max={1000}
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value))}
                />
                <span style={{ fontSize: '0.8rem', color: '#666' }}>{steps}</span>
              </Control>
            );
          }
          if (control.type === 'toggle') {
            if (control.id === 'show-forward') {
              return (
                <Toggle key={control.id}>
                  <Checkbox
                    type="checkbox"
                    checked={showForward}
                    onChange={(e) => setShowForward(e.target.checked)}
                  />
                  {control.label}
                </Toggle>
              );
            }
            if (control.id === 'show-reverse') {
              return (
                <Toggle key={control.id}>
                  <Checkbox
                    type="checkbox"
                    checked={showReverse}
                    onChange={(e) => setShowReverse(e.target.checked)}
                  />
                  {control.label}
                </Toggle>
              );
            }
          }
          return null;
        })}
        
        <Control>
          <Label>Animation</Label>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handlePlayPause}
              style={{
                padding: '4px 8px',
                background: isPlaying ? '#dc2626' : '#059669',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
            >
              {isPlaying ? '⏸️ Pause' : '▶️ Play'}
            </button>
            <button
              onClick={handleReset}
              style={{
                padding: '4px 8px',
                background: '#6b7280',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
            >
              🔄 Reset
            </button>
          </div>
        </Control>
      </Controls>

      <ProcessContainer>
        {showForward && (
          <ProcessCard>
            <ProcessTitle>Forward Process (Noising)</ProcessTitle>
            <Canvas ref={forwardCanvasRef} />
            <StepInfo>
              <span>Step: {currentStep + 1}/{steps}</span>
              <span>β: {currentStep < steps ? generateNoiseSchedule(steps)[currentStep].toFixed(4) : '0.0000'}</span>
            </StepInfo>
          </ProcessCard>
        )}
        
        {showReverse && (
          <ProcessCard>
            <ProcessTitle>Reverse Process (Denoising)</ProcessTitle>
            <Canvas ref={reverseCanvasRef} />
            <StepInfo>
              <span>Step: {steps - currentStep}/{steps}</span>
              <span>Quality: {((steps - currentStep) / steps * 100).toFixed(1)}%</span>
            </StepInfo>
          </ProcessCard>
        )}
      </ProcessContainer>

      <Timeline>
        {Array.from({ length: Math.min(10, steps) }, (_, i) => {
          const stepIndex = Math.floor(i * (steps - 1) / 9);
          return (
            <TimelineStep
              key={i}
              $active={currentStep === stepIndex}
              $completed={currentStep > stepIndex}
            />
          );
        })}
      </Timeline>

      <Explanation>
        <h4>Diffusion Process Overview</h4>
        <p>
          Diffusion models work by learning to reverse a gradual noising process. 
          The forward process adds noise step by step until the image becomes pure noise, 
          while the reverse process learns to remove noise to recover the original image.
        </p>
        <p>
          <strong>Key Components:</strong>
        </p>
        <ul>
          <li><strong>Forward Process:</strong> q(x_t | x_t-1) = N(x_t; sqrt(1-β_t) x_t-1, β_t I)</li>
          <li><strong>Noise Schedule:</strong> β_t controls how much noise is added at each step</li>
          <li><strong>Reverse Process:</strong> p_θ(x_t-1 | x_t) learns to predict and remove noise</li>
          <li><strong>Training:</strong> Model learns to predict the noise ε that was added</li>
        </ul>
        <p>
          <strong>Advantages:</strong> More stable than GANs, high-quality generation, 
          diverse outputs, and principled training objective.
        </p>
      </Explanation>
    </Container>
  );
};

export default DiffusionProcess; 