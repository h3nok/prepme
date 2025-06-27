import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, ExternalLink, Code, BookOpen, Users, Zap, Sun, Moon } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const HeaderContainer = styled(motion.header)`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.lg} ${props => props.theme.spacing.xl};
  background: transparent;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
  border-bottom: none;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: none;

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    padding: ${props => props.theme.spacing.md};
  }
`;

const ExternalLinks = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};

  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    display: none;
  }
`;

const ExternalLinkButton = styled.a`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: rgba(255, 107, 53, 0.1);
  border: 1px solid rgba(255, 107, 53, 0.2);
  border-radius: 20px;
  color: ${props => props.theme.colors.primary};
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 500;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);

  &:hover {
    background: rgba(255, 107, 53, 0.2);
    border-color: rgba(255, 107, 53, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 107, 53, 0.2);
  }

  svg {
    width: 14px;
    height: 14px;
  }
`;

const ResourcesDropdown = styled.div`
  position: relative;
`;

const ResourcesButton = styled.button`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 20px;
  color: #3b82f6;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);

  &:hover {
    background: rgba(59, 130, 246, 0.2);
    border-color: rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
  }

  svg {
    width: 14px;
    height: 14px;
  }
`;

const DropdownMenu = styled(motion.div)`
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: ${props => props.theme.spacing.sm};
  background: rgba(30, 41, 59, 0.95);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 15px;
  padding: ${props => props.theme.spacing.md};
  min-width: 250px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
  z-index: 1000;
`;

const DropdownSection = styled.div`
  margin-bottom: ${props => props.theme.spacing.md};

  &:last-child {
    margin-bottom: 0;
  }
`;

const DropdownTitle = styled.h4`
  color: ${props => props.theme.colors.text};
  font-size: 0.875rem;
  font-weight: 600;
  margin: 0 0 ${props => props.theme.spacing.sm} 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const DropdownLink = styled.a`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textSecondary};
  text-decoration: none;
  border-radius: ${props => props.theme.radii.sm};
  transition: all 0.2s ease;
  font-size: 0.875rem;

  &:hover {
    background: rgba(255, 255, 255, 0.05);
    color: ${props => props.theme.colors.text};
    transform: translateX(4px);
  }

  svg {
    width: 16px;
    height: 16px;
    color: ${props => props.theme.colors.primary};
  }
`;

const MenuButton = styled.button`
  display: none;
  background: rgba(255, 107, 53, 0.1);
  border: 1px solid rgba(255, 107, 53, 0.2);
  color: ${props => props.theme.colors.primary};
  cursor: pointer;
  padding: ${props => props.theme.spacing.sm};
  border-radius: 10px;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);

  &:hover {
    background: rgba(255, 107, 53, 0.2);
    transform: translateY(-2px);
  }

  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  svg {
    width: 20px;
    height: 20px;
  }
`;

const ThemeToggle = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);

  &:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  }

  svg {
    width: 18px;
    height: 18px;
  }
`;

const Header: React.FC = () => {
  const [isResourcesOpen, setIsResourcesOpen] = useState(false);
  const { toggleTheme, isDark } = useTheme();

  const codingPlatforms = [
    { name: 'LeetCode', url: 'https://leetcode.com', icon: Code },
    { name: 'HackerRank', url: 'https://hackerrank.com', icon: Code },
    { name: 'CodeSignal', url: 'https://codesignal.com', icon: Zap }
  ];

  const researchResources = [
    { name: 'Papers with Code', url: 'https://paperswithcode.com', icon: BookOpen },
    { name: 'ArXiv', url: 'https://arxiv.org', icon: BookOpen },
    { name: 'Hugging Face', url: 'https://huggingface.co', icon: Users }
  ];

  return (
    <HeaderContainer
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <ExternalLinks>
        {codingPlatforms.map((platform) => (
          <ExternalLinkButton
            key={platform.name}
            href={platform.url}
            target="_blank"
            rel="noopener noreferrer"
          >
            <platform.icon />
            {platform.name}
            <ExternalLink />
          </ExternalLinkButton>
        ))}

        <ResourcesDropdown>
          <ResourcesButton
            onClick={() => setIsResourcesOpen(!isResourcesOpen)}
          >
            <BookOpen />
            Resources
          </ResourcesButton>

          <AnimatePresence>
            {isResourcesOpen && (
              <DropdownMenu
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.2 }}
              >
                <DropdownSection>
                  <DropdownTitle>Research Papers</DropdownTitle>
                  {researchResources.map((resource) => (
                    <DropdownLink
                      key={resource.name}
                      href={resource.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <resource.icon />
                      {resource.name}
                    </DropdownLink>
                  ))}
                </DropdownSection>

                <DropdownSection>
                  <DropdownTitle>AI Communities</DropdownTitle>
                  <DropdownLink
                    href="https://www.reddit.com/r/MachineLearning"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Users />
                    r/MachineLearning
                  </DropdownLink>
                  <DropdownLink
                    href="https://distill.pub"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <BookOpen />
                    Distill
                  </DropdownLink>
                </DropdownSection>
              </DropdownMenu>
            )}
          </AnimatePresence>
        </ResourcesDropdown>
      </ExternalLinks>

      <ThemeToggle onClick={toggleTheme} title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}>
        {isDark ? <Sun /> : <Moon />}
      </ThemeToggle>

      <MenuButton>
        <Menu />
      </MenuButton>
    </HeaderContainer>
  );
};

export default Header;
