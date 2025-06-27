import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Brain, Search, Menu } from 'lucide-react';

const HeaderContainer = styled(motion.header)`
  display: flex;
  align-items: center;
  justify-content: between;
  padding: ${props => props.theme.spacing.lg} ${props => props.theme.spacing.xl};
  background: ${props => props.theme.colors.surface};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 100;

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    padding: ${props => props.theme.spacing.md};
  }
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => props.theme.colors.primary};
`;

const SearchContainer = styled.div`
  flex: 1;
  max-width: 400px;
  margin: 0 ${props => props.theme.spacing.xl};
  position: relative;

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    display: none;
  }
`;

const SearchInput = styled.input`
  width: 100%;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  padding-left: 2.5rem;
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  color: ${props => props.theme.colors.text};
  font-size: 0.9rem;
  transition: all 0.2s ease;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
    box-shadow: 0 0 0 3px ${props => props.theme.colors.primary}20;
  }

  &::placeholder {
    color: ${props => props.theme.colors.textSecondary};
  }
`;

const SearchIcon = styled(Search)`
  position: absolute;
  left: ${props => props.theme.spacing.sm};
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  color: ${props => props.theme.colors.textSecondary};
`;

const MenuButton = styled.button`
  display: none;
  background: none;
  border: none;
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  padding: ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.sm};
  transition: background-color 0.2s ease;

  &:hover {
    background: ${props => props.theme.colors.surfaceLight};
  }

  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    display: flex;
    align-items: center;
    justify-content: center;
  }
`;

const Header: React.FC = () => {
  return (
    <HeaderContainer
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Logo>
        <Brain size={28} />
        <span>Prep</span>
      </Logo>
      
      <SearchContainer>
        <SearchIcon />
        <SearchInput 
          type="text" 
          placeholder="Search topics, formulas, concepts..." 
        />
      </SearchContainer>

      <MenuButton>
        <Menu size={24} />
      </MenuButton>
    </HeaderContainer>
  );
};

export default Header;
