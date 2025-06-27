import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

interface CardProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
  variant?: 'default' | 'accent' | 'warning' | 'success' | 'purple';
  hover?: boolean;
}

const CardContainer = styled(motion.div)<{ $variant: string; $hover: boolean }>`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => {
    switch (props.$variant) {
      case 'accent': return props.theme.colors.accent;
      case 'warning': return props.theme.colors.warning;
      case 'success': return props.theme.colors.success;
      case 'purple': return props.theme.colors.purple;
      default: return props.theme.colors.border;
    }
  }};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.sm};
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: ${props => {
      switch (props.$variant) {
        case 'accent': return props.theme.colors.accent;
        case 'warning': return props.theme.colors.warning;
        case 'success': return props.theme.colors.success;
        case 'purple': return props.theme.colors.purple;
        default: return props.theme.colors.primary;
      }
    }};
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
  }

  ${props => props.$hover && `
    &:hover {
      transform: translateY(-4px);
      box-shadow: ${props.theme.shadows.lg};
      border-color: ${props.theme.colors.primary};

      &::before {
        transform: scaleX(1);
      }
    }
  `}
`;

const CardTitle = styled.h3<{ $variant: string }>`
  margin: 0 0 ${props => props.theme.spacing.md} 0;
  color: ${props => {
    switch (props.$variant) {
      case 'accent': return props.theme.colors.accent;
      case 'warning': return props.theme.colors.warning;
      case 'success': return props.theme.colors.success;
      case 'purple': return props.theme.colors.purple;
      default: return props.theme.colors.primary;
    }
  }};
  font-weight: 600;
  font-size: 1.25rem;
`;

const CardContent = styled.div`
  color: ${props => props.theme.colors.text};
  line-height: 1.6;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin: ${props => props.theme.spacing.md} 0 ${props => props.theme.spacing.sm} 0;
    font-weight: 600;
  }

  h5 {
    color: ${props => props.theme.colors.accent};
    margin: ${props => props.theme.spacing.sm} 0;
    font-weight: 600;
  }

  ul, ol {
    margin: ${props => props.theme.spacing.sm} 0;
    padding-left: ${props => props.theme.spacing.lg};
  }

  li {
    margin-bottom: ${props => props.theme.spacing.xs};
  }

  p {
    margin-bottom: ${props => props.theme.spacing.md};
  }

  code {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
    border-radius: ${props => props.theme.radii.sm};
    font-family: ${props => props.theme.fonts.mono};
    font-size: 0.9em;
    color: ${props => props.theme.colors.primary};
  }

  pre {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.md};
    border-radius: ${props => props.theme.radii.md};
    overflow-x: auto;
    margin: ${props => props.theme.spacing.md} 0;
    border: 1px solid ${props => props.theme.colors.border};

    code {
      background: none;
      padding: 0;
      color: ${props => props.theme.colors.text};
    }
  }
`;

const Card: React.FC<CardProps> = ({ 
  children, 
  title, 
  className, 
  variant = 'default',
  hover = true 
}) => {
  return (
    <CardContainer
      className={className}
      $variant={variant}
      $hover={hover}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
    >
      {title && <CardTitle $variant={variant}>{title}</CardTitle>}
      <CardContent>{children}</CardContent>
    </CardContainer>
  );
};

export default Card;
