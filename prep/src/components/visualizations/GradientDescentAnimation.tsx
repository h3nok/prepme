import React, { useState, useEffect } from 'react';
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
  grid-template-columns: 1fr 1fr 1fr;
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

const AnimationControls = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
  justify-content: center;
`;

const Button = styled.button<{ $active?: boolean }>`
  background: ${props => props.$active ? '#ff6b35' : 'rgba(255, 255, 255, 0.1)'};
  border: 1px solid ${props => props.$active ? '#ff6b35' : 'rgba(255, 255, 255, 0.2)'};
  color: #f8fafc;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9rem;

  &:hover {
    background: ${props => props.$active ? '#ff8c42' : 'rgba(255, 255, 255, 0.2)'};
  }
`;

const Stats = styled.div`
  text-align: center;
  color: #f8fafc;
  font-size: 0.9rem;
  margin-top: 1rem;
`;

interface GradientDescentAnimationProps {
  function?: 'quadratic' | 'complex';
}

const GradientDescentAnimation: React.FC<GradientDescentAnimationProps> = ({
  function: functionType = 'quadratic'
}) => {
  const [learningRate, setLearningRate] = useState(0.1);
  const [startingPoint, setStartingPoint] = useState(3);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentPoint, setCurrentPoint] = useState({ x: 3, y: 0 });
  const [path, setPath] = useState<Array<{ x: number; y: number }>>([]);
  const [step, setStep] = useState(0);

  const chartWidth = 400;
  const chartHeight = 300;
  const xMin = -5;
  const xMax = 5;
  const xRange = xMax - xMin;

  // Objective function (quadratic: f(x) = x^2)
  const objectiveFunction = (x: number): number => {
    if (functionType === 'quadratic') {
      return x * x;
    } else {
      // More complex function with local minima
      return x * x + 2 * Math.sin(3 * x) + 0.5 * Math.cos(5 * x);
    }
  };

  // Gradient (derivative) of the function
  const gradient = (x: number): number => {
    if (functionType === 'quadratic') {
      return 2 * x;
    } else {
      return 2 * x + 6 * Math.cos(3 * x) - 2.5 * Math.sin(5 * x);
    }
  };

  // Convert function coordinates to screen coordinates
  const toScreenCoords = (x: number, y: number) => {
    const screenX = ((x - xMin) / xRange) * chartWidth;
    const maxY = Math.max(...Array.from({ length: 100 }, (_, i) => {
      const fx = xMin + (i / 99) * xRange;
      return objectiveFunction(fx);
    }));
    const screenY = chartHeight - (y / maxY) * chartHeight * 0.8;
    return { x: screenX, y: screenY };
  };

  // Generate function curve points
  const generateFunctionCurve = () => {
    const points = [];
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * xRange;
      const y = objectiveFunction(x);
      const screen = toScreenCoords(x, y);
      points.push(`${screen.x},${screen.y}`);
    }
    return points.join(' ');
  };

  // Perform one gradient descent step
  const gradientDescentStep = (x: number, lr: number): number => {
    const grad = gradient(x);
    return x - lr * grad;
  };

  // Start animation
  const startAnimation = () => {
    setIsAnimating(true);
    setPath([]);
    setStep(0);
    
    let currentX = startingPoint;
    const newPath = [{ x: currentX, y: objectiveFunction(currentX) }];
    
    const animate = () => {
      const nextX = gradientDescentStep(currentX, learningRate);
      const nextY = objectiveFunction(nextX);
      
      newPath.push({ x: nextX, y: nextY });
      setPath([...newPath]);
      setCurrentPoint({ x: nextX, y: nextY });
      setStep(newPath.length - 1);
      
      currentX = nextX;
      
      // Continue if not converged and not too many steps
      if (Math.abs(gradient(nextX)) > 0.01 && newPath.length < 100) {
        setTimeout(animate, 500);
      } else {
        setIsAnimating(false);
      }
    };
    
    setTimeout(animate, 500);
  };

  // Reset animation
  const resetAnimation = () => {
    setIsAnimating(false);
    setPath([]);
    setCurrentPoint({ x: startingPoint, y: objectiveFunction(startingPoint) });
    setStep(0);
  };

  useEffect(() => {
    resetAnimation();
  }, [startingPoint, learningRate, functionType]);

  const currentScreen = toScreenCoords(currentPoint.x, currentPoint.y);

  return (
    <Container>
      <ChartContainer>
        <SVGContainer viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
          {/* Background grid */}
          <defs>
            <pattern id="gradient-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#gradient-grid)" />
          
          {/* Function curve */}
          <polyline
            points={generateFunctionCurve()}
            fill="none"
            stroke="#7c3aed"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          
          {/* Gradient descent path */}
          {path.length > 1 && (
            <polyline
              points={path.map(p => {
                const screen = toScreenCoords(p.x, p.y);
                return `${screen.x},${screen.y}`;
              }).join(' ')}
              fill="none"
              stroke="#ff6b35"
              strokeWidth="2"
              strokeDasharray="5,5"
              strokeLinecap="round"
            />
          )}
          
          {/* Path points */}
          {path.map((point, index) => {
            const screen = toScreenCoords(point.x, point.y);
            return (
              <motion.circle
                key={index}
                cx={screen.x}
                cy={screen.y}
                r="4"
                fill="#ff6b35"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.1 }}
              />
            );
          })}
          
          {/* Current point */}
          <motion.circle
            cx={currentScreen.x}
            cy={currentScreen.y}
            r="8"
            fill="#ff6b35"
            stroke="#fff"
            strokeWidth="2"
            animate={{ scale: isAnimating ? [1, 1.2, 1] : 1 }}
            transition={{ duration: 0.5, repeat: isAnimating ? Infinity : 0 }}
          />
          
          {/* Gradient arrow at current point */}
          {path.length > 0 && (
            <g>
              <motion.line
                x1={currentScreen.x}
                y1={currentScreen.y}
                x2={currentScreen.x + gradient(currentPoint.x) * -20}
                y2={currentScreen.y}
                stroke="#059669"
                strokeWidth="3"
                markerEnd="url(#arrowhead)"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.5 }}
              />
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                 refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#059669" />
                </marker>
              </defs>
            </g>
          )}
          
          {/* Axes */}
          <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
          <line x1={chartWidth/2} y1="0" x2={chartWidth/2} y2={chartHeight} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
          
          {/* Labels */}
          <text x={chartWidth/2} y={chartHeight - 5} textAnchor="middle" fill="#f8fafc" fontSize="12">
            x
          </text>
          <text x="10" y="15" fill="#f8fafc" fontSize="12">
            f(x)
          </text>
        </SVGContainer>
      </ChartContainer>

      <Controls>
        <ControlGroup>
          <Label>
            Learning Rate (α)
            <span>{learningRate.toFixed(2)}</span>
          </Label>
          <Slider
            type="range"
            min="0.01"
            max="0.5"
            step="0.01"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            disabled={isAnimating}
          />
        </ControlGroup>
        
        <ControlGroup>
          <Label>
            Starting Point
            <span>{startingPoint.toFixed(1)}</span>
          </Label>
          <Slider
            type="range"
            min="-4"
            max="4"
            step="0.1"
            value={startingPoint}
            onChange={(e) => setStartingPoint(parseFloat(e.target.value))}
            disabled={isAnimating}
          />
        </ControlGroup>

        <ControlGroup>
          <Label>Function Type</Label>
          <select 
            value={functionType} 
            onChange={(e) => setLearningRate(0.1)}
            style={{
              background: 'rgba(255, 255, 255, 0.1)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: '#f8fafc',
              padding: '0.5rem',
              borderRadius: '4px'
            }}
            disabled={isAnimating}
          >
            <option value="quadratic">Quadratic (x²)</option>
            <option value="complex">Complex</option>
          </select>
        </ControlGroup>
      </Controls>

      <AnimationControls>
        <Button onClick={startAnimation} disabled={isAnimating} $active={isAnimating}>
          {isAnimating ? 'Running...' : 'Start Descent'}
        </Button>
        <Button onClick={resetAnimation} disabled={isAnimating}>
          Reset
        </Button>
      </AnimationControls>

      <Stats>
        <div>Step: {step} | Current x: {currentPoint.x.toFixed(3)} | Current f(x): {currentPoint.y.toFixed(3)}</div>
        <div>Gradient: {gradient(currentPoint.x).toFixed(3)}</div>
      </Stats>
    </Container>
  );
};

export default GradientDescentAnimation;
