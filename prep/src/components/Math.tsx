import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { InlineMath, BlockMath } from 'react-katex';

interface MathComponentProps {
  children: string;
  block?: boolean;
  className?: string;
}

const MathContainer = styled(motion.div)<{ $block?: boolean }>`
  margin: ${props => props.$block ? props.theme.spacing.md : '0'} 0;
  padding: ${props => props.$block ? props.theme.spacing.md : '0'};
  background: ${props => props.$block ? props.theme.colors.surface : 'transparent'};
  border-radius: ${props => props.$block ? props.theme.radii.md : '0'};
  border: ${props => props.$block ? `1px solid ${props.theme.colors.border}` : 'none'};
  overflow-x: auto;
  
  .katex {
    font-size: ${props => props.$block ? '1.1em' : '1em'};
  }

  .katex-display {
    margin: 0;
  }
`;

const MathError = styled.span`
  color: ${props => props.theme.colors.error};
  background: ${props => props.theme.colors.surface};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.sm};
  font-family: ${props => props.theme.fonts.mono};
  font-size: 0.9em;
`;

const Math: React.FC<MathComponentProps> = ({ children, block = false, className }) => {
  try {
    return (
      <MathContainer 
        $block={block}
        className={className}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {block ? (
          <BlockMath math={children} />
        ) : (
          <InlineMath math={children} />
        )}
      </MathContainer>
    );
  } catch (error) {
    return (
      <MathError>
        Math Error: {children}
      </MathError>
    );
  }
};

export default Math;
