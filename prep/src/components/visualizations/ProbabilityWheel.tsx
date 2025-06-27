import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const Container = styled.div`
  width: 100%;
  height: 300px;
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const SVGContainer = styled.svg`
  width: 200px;
  height: 200px;
  overflow: visible;
`;

const Sector = styled(motion.path)<{ $color: string }>`
  fill: ${props => props.$color};
  stroke: #1e293b;
  stroke-width: 2;
  cursor: pointer;
`;

const Controls = styled.div`
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const Slider = styled.input`
  width: 150px;
  appearance: none;
  height: 4px;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.2);
  outline: none;

  &::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ff6b35;
    cursor: pointer;
  }

  &::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ff6b35;
    cursor: pointer;
    border: none;
  }
`;

const Label = styled.label`
  color: #f8fafc;
  font-size: 0.9rem;
  font-weight: 500;
`;

const ProbabilityDisplay = styled.div`
  margin-top: 1rem;
  text-align: center;
  color: #f8fafc;
`;

interface ProbabilityWheelProps {
  events?: string[];
  probabilities?: number[];
}

const ProbabilityWheel: React.FC<ProbabilityWheelProps> = ({
  events = ['Heads', 'Tails'],
  probabilities = [0.5, 0.5]
}) => {
  const [bias, setBias] = useState(0.5);
  const [isSpinning, setIsSpinning] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const headsProbability = bias;
  const tailsProbability = 1 - bias;

  const createPath = (centerX: number, centerY: number, radius: number, startAngle: number, endAngle: number) => {
    const start = polarToCartesian(centerX, centerY, radius, endAngle);
    const end = polarToCartesian(centerX, centerY, radius, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";

    return [
      "M", centerX, centerY,
      "L", start.x, start.y,
      "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y,
      "Z"
    ].join(" ");
  };

  const polarToCartesian = (centerX: number, centerY: number, radius: number, angleInDegrees: number) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + (radius * Math.cos(angleInRadians)),
      y: centerY + (radius * Math.sin(angleInRadians))
    };
  };

  const spin = () => {
    setIsSpinning(true);
    setResult(null);
    
    setTimeout(() => {
      const random = Math.random();
      const outcome = random < headsProbability ? 'Heads' : 'Tails';
      setResult(outcome);
      setIsSpinning(false);
    }, 2000);
  };

  const headsAngle = headsProbability * 360;
  const tailsAngle = 360 - headsAngle;

  return (
    <Container>
      <SVGContainer>
        {/* Heads sector */}
        <Sector
          $color="rgba(255, 107, 53, 0.8)"
          d={createPath(100, 100, 80, 0, headsAngle)}
          animate={isSpinning ? { rotate: 360 } : {}}
          transition={isSpinning ? { duration: 2, ease: "easeOut" } : {}}
          whileHover={{ scale: 1.05 }}
          onClick={spin}
        />
        
        {/* Tails sector */}
        <Sector
          $color="rgba(124, 58, 237, 0.8)"
          d={createPath(100, 100, 80, headsAngle, 360)}
          animate={isSpinning ? { rotate: 360 } : {}}
          transition={isSpinning ? { duration: 2, ease: "easeOut" } : {}}
          whileHover={{ scale: 1.05 }}
          onClick={spin}
        />
        
        {/* Center circle */}
        <circle cx="100" cy="100" r="10" fill="#1e293b" />
        
        {/* Labels */}
        <text x="130" y="60" fill="#f8fafc" fontSize="12" fontWeight="600">
          Heads
        </text>
        <text x="130" y="150" fill="#f8fafc" fontSize="12" fontWeight="600">
          Tails
        </text>
      </SVGContainer>

      <Controls>
        <Label>Coin Bias:</Label>
        <Slider
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={bias}
          onChange={(e) => setBias(parseFloat(e.target.value))}
        />
        <span style={{ color: '#f8fafc', fontSize: '0.9rem' }}>
          {(bias * 100).toFixed(0)}%
        </span>
      </Controls>

      <ProbabilityDisplay>
        <div>P(Heads) = {headsProbability.toFixed(2)}</div>
        <div>P(Tails) = {tailsProbability.toFixed(2)}</div>
        {result && (
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{ 
              marginTop: '1rem', 
              padding: '0.5rem 1rem',
              background: result === 'Heads' ? 'rgba(255, 107, 53, 0.3)' : 'rgba(124, 58, 237, 0.3)',
              borderRadius: '8px',
              fontWeight: 'bold'
            }}
          >
            Result: {result}!
          </motion.div>
        )}
      </ProbabilityDisplay>
    </Container>
  );
};

export default ProbabilityWheel;
