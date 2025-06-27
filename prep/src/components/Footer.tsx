import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Github, Twitter, Linkedin, Heart } from 'lucide-react';

const FooterContainer = styled(motion.footer)`
  background: ${props => props.theme.colors.surface};
  border-top: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.xl};
  margin-top: auto;
`;

const FooterContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: ${props => props.theme.spacing.xl};
  align-items: center;

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-template-columns: 1fr;
    text-align: center;
    gap: ${props => props.theme.spacing.md};
  }
`;

const Copyright = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    justify-content: center;
  }
`;

const SocialLinks = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  align-items: center;
`;

const SocialLink = styled.a`
  color: ${props => props.theme.colors.textSecondary};
  transition: all 0.2s ease;
  padding: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.sm};

  &:hover {
    color: ${props => props.theme.colors.primary};
    background: ${props => props.theme.colors.surfaceLight};
    transform: translateY(-2px);
  }

  svg {
    width: 18px;
    height: 18px;
  }
`;

const Links = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.lg};
  
  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    justify-content: center;
  }
`;

const Link = styled.a`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  transition: color 0.2s ease;

  &:hover {
    color: ${props => props.theme.colors.primary};
  }
`;

const Footer: React.FC = () => {
  return (
    <FooterContainer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <FooterContent>
        <Copyright>
          Made with <Heart size={16} fill="currentColor" /> for AI researchers and engineers
        </Copyright>
        
        <Links>
          <Link href="#about">About</Link>
          <Link href="#feedback">Feedback</Link>
          <Link href="#contribute">Contribute</Link>
        </Links>

        <SocialLinks>
          <SocialLink href="https://github.com" target="_blank" rel="noopener noreferrer">
            <Github />
          </SocialLink>
          <SocialLink href="https://twitter.com" target="_blank" rel="noopener noreferrer">
            <Twitter />
          </SocialLink>
          <SocialLink href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
            <Linkedin />
          </SocialLink>
        </SocialLinks>
      </FooterContent>
    </FooterContainer>
  );
};

export default Footer;
