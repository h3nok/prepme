import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const Container = styled.div`
  width: 100%;
  height: 400px;
  background: rgba(30, 41, 59, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
`;

const ChartContainer = styled.div`
  flex: 1;
  position: relative;
  margin-bottom: 1rem;
`;

const SVGContainer = styled.svg`
  width: 100%;
  height: 100%;
`;

const Controls = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1rem;
`;

const ControlGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Slider = styled.input`
  width: 100%;
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
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Stats = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  text-align: center;
  color: #f8fafc;
  font-size: 0.9rem;
`;

interface NormalDistributionProps {
  mu?: number;
  sigma?: number;
}

const NormalDistribution: React.FC<NormalDistributionProps> = ({
  mu: initialMu = 0,
  sigma: initialSigma = 1
}) => {
  const [mu, setMu] = useState(initialMu);
  const [sigma, setSigma] = useState(initialSigma);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Normal distribution PDF
  const normalPDF = (x: number, mean: number, std: number): number => {
    const coefficient = 1 / (std * Math.sqrt(2 * Math.PI));
    const exponent = -0.5 * Math.pow((x - mean) / std, 2);
    return coefficient * Math.exp(exponent);
  };

  // Generate curve points
  const generateCurvePoints = (width: number, height: number) => {
    const points = [];
    const xMin = mu - 4 * sigma;
    const xMax = mu + 4 * sigma;
    const xRange = xMax - xMin;
    
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * xRange;
      const y = normalPDF(x, mu, sigma);
      const pixelX = (i / 200) * width;
      const pixelY = height - (y / normalPDF(mu, mu, sigma)) * height * 0.8;
      points.push(`${pixelX},${pixelY}`);
    }
    
    return points.join(' ');
  };

  // Generate area under curve for standard deviations
  const generateAreaPoints = (width: number, height: number, stdDevs: number) => {
    const points = [];
    const xMin = mu - stdDevs * sigma;
    const xMax = mu + stdDevs * sigma;
    const xRange = mu + 4 * sigma - (mu - 4 * sigma);
    const xOffset = mu - 4 * sigma;
    
    // Add bottom points
    const startX = ((xMin - xOffset) / xRange) * width;
    const endX = ((xMax - xOffset) / xRange) * width;
    
    points.push(`${startX},${height}`);
    
    for (let i = 0; i <= 100; i++) {
      const x = xMin + (i / 100) * (xMax - xMin);
      const y = normalPDF(x, mu, sigma);
      const pixelX = ((x - xOffset) / xRange) * width;
      const pixelY = height - (y / normalPDF(mu, mu, sigma)) * height * 0.8;
      points.push(`${pixelX},${pixelY}`);
    }
    
    points.push(`${endX},${height}`);
    
    return points.join(' ');
  };

  const chartWidth = 300;
  const chartHeight = 200;

  // Calculate probabilities
  const prob1Sigma = 0.6827;
  const prob2Sigma = 0.9545;
  const prob3Sigma = 0.9973;

  return (
    <Container>
      <ChartContainer>
        <SVGContainer viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
          {/* Background grid */}
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* 3 sigma area */}
          <polygon
            points={generateAreaPoints(chartWidth, chartHeight, 3)}
            fill="rgba(255, 107, 53, 0.1)"
            stroke="none"
          />
          
          {/* 2 sigma area */}
          <polygon
            points={generateAreaPoints(chartWidth, chartHeight, 2)}
            fill="rgba(255, 107, 53, 0.2)"
            stroke="none"
          />
          
          {/* 1 sigma area */}
          <polygon
            points={generateAreaPoints(chartWidth, chartHeight, 1)}
            fill="rgba(255, 107, 53, 0.4)"
            stroke="none"
          />
          
          {/* Main curve */}
          <polyline
            points={generateCurvePoints(chartWidth, chartHeight)}
            fill="none"
            stroke="#ff6b35"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          
          {/* Mean line */}
          <line
            x1={chartWidth / 2}
            y1="0"
            x2={chartWidth / 2}
            y2={chartHeight}
            stroke="#7c3aed"
            strokeWidth="2"
            strokeDasharray="5,5"
          />
          
          {/* Standard deviation lines */}
          {[-3, -2, -1, 1, 2, 3].map(std => {
            const x = chartWidth / 2 + (std / 4) * chartWidth / 2;
            return (
              <line
                key={std}
                x1={x}
                y1="0"
                x2={x}
                y2={chartHeight}
                stroke="rgba(248, 250, 252, 0.3)"
                strokeWidth="1"
                strokeDasharray="3,3"
              />
            );
          })}
          
          {/* Labels */}
          <text x={chartWidth / 2} y={chartHeight - 10} textAnchor="middle" fill="#f8fafc" fontSize="12">
            μ = {mu.toFixed(1)}
          </text>
        </SVGContainer>
      </ChartContainer>

      <Controls>
        <ControlGroup>
          <Label>
            Mean (μ)
            <span>{mu.toFixed(1)}</span>
          </Label>
          <Slider
            type="range"
            min="-3"
            max="3"
            step="0.1"
            value={mu}
            onChange={(e) => setMu(parseFloat(e.target.value))}
          />
        </ControlGroup>
        
        <ControlGroup>
          <Label>
            Standard Deviation (σ)
            <span>{sigma.toFixed(1)}</span>
          </Label>
          <Slider
            type="range"
            min="0.5"
            max="3"
            step="0.1"
            value={sigma}
            onChange={(e) => setSigma(parseFloat(e.target.value))}
          />
        </ControlGroup>
      </Controls>

      <Stats>
        <div>
          <div style={{ fontWeight: 'bold', color: '#ff6b35' }}>±1σ</div>
          <div>{(prob1Sigma * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div style={{ fontWeight: 'bold', color: '#ff6b35' }}>±2σ</div>
          <div>{(prob2Sigma * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div style={{ fontWeight: 'bold', color: '#ff6b35' }}>±3σ</div>
          <div>{(prob3Sigma * 100).toFixed(1)}%</div>
        </div>
      </Stats>
    </Container>
  );
};

export default NormalDistribution;
