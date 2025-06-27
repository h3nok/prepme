import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Layers, 
  Brain, 
  Wand2, 
  Eye, 
  Cloud, 
  HelpCircle,
  Target,
  Menu,
  X,
  Users,
  BarChart
} from 'lucide-react';

import { useSidebar } from '../context/SidebarContext';

const SidebarContainer = styled(motion.aside)<{ $isCollapsed: boolean; $isMobileOpen: boolean }>`
  position: fixed;
  left: 0;
  top: 0;
  width: ${props => props.$isCollapsed ? '80px' : '280px'};
  height: 100vh;
  background: ${props => props.theme.colors.background === '#ffffff' 
    ? 'rgba(255, 255, 255, 0.95)' 
    : 'rgba(15, 23, 42, 0.95)'};
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-right: 1px solid ${props => props.theme.colors.background === '#ffffff' 
    ? 'rgba(0, 0, 0, 0.1)' 
    : 'rgba(255, 255, 255, 0.1)'};
  padding: ${props => props.theme.spacing.lg};
  overflow-y: auto;
  z-index: 50;
  transition: width 0.3s ease;
  box-shadow: ${props => props.theme.colors.background === '#ffffff' 
    ? '4px 0 20px rgba(0, 0, 0, 0.1)' 
    : '4px 0 20px rgba(0, 0, 0, 0.3)'};

  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    width: 280px;
    transform: translateX(${props => props.$isMobileOpen ? '0' : '-100%'});
    transition: transform 0.3s ease;
  }

  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
  }

  &::-webkit-scrollbar-thumb {
    background: rgba(255, 107, 53, 0.3);
    border-radius: 3px;
    
    &:hover {
      background: rgba(255, 107, 53, 0.5);
    }
  }
`;

const SidebarHeader = styled.div<{ $isCollapsed: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: ${props => props.theme.spacing.xl};
  padding-bottom: ${props => props.theme.spacing.md};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  min-height: 60px;
  position: relative;
`;

const Logo = styled.div<{ $isCollapsed: boolean }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  flex: 1;
  
  .logo-icon {
    width: 32px;
    height: 32px;
    background: ${props => props.theme.colors.primary};
    border-radius: ${props => props.theme.radii.md};
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 1.2rem;
    flex-shrink: 0;
  }
  
  .logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    color: ${props => props.theme.colors.text};
    display: ${props => props.$isCollapsed ? 'none' : 'block'};
    white-space: nowrap;
    overflow: hidden;
    
    span {
      color: ${props => props.theme.colors.primary};
    }
  }
`;

const CollapseButton = styled.button<{ $isCollapsed: boolean }>`
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: ${props => props.theme.colors.text};
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;
  
  &:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
  }

  &:active {
    transform: scale(0.95);
  }
  
  svg {
    width: 16px;
    height: 16px;
    transition: transform 0.2s ease;
  }
  
  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    display: none;
  }
`;

const MobileMenuButton = styled.button`
  position: fixed;
  top: 20px;
  left: 20px;
  z-index: 60;
  background: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  width: 50px;
  height: 50px;
  border-radius: ${props => props.theme.radii.md};
  display: none;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;
  
  &:hover {
    background: ${props => props.theme.colors.primary}dd;
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
  }

  &:active {
    transform: scale(0.95);
  }
  
  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    display: flex;
  }
`;

const MobileOverlay = styled(motion.div)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 40;
  display: none;
  
  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    display: block;
  }
`;

const SidebarSection = styled.div`
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const SectionTitle = styled.h3<{ $isCollapsed: boolean }>`
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  color: rgba(255, 107, 53, 0.8);
  margin-bottom: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.lg};
  letter-spacing: 1px;
  display: ${props => props.$isCollapsed ? 'none' : 'block'};
  position: relative;
  padding-bottom: ${props => props.theme.spacing.xs};
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 24px;
    height: 2px;
    background: linear-gradient(90deg, #ff6b35, #ff8c42);
    border-radius: 1px;
  }
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const NavItem = styled.li<{ $isActive?: boolean }>`
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const NavLink = styled(Link)<{ $isActive?: boolean; $isCollapsed?: boolean }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.radii.md};
  color: ${props => props.$isActive ? '#fff' : props.theme.colors.text};
  background: ${props => props.$isActive 
    ? 'linear-gradient(135deg, rgba(255, 107, 53, 0.8), rgba(255, 140, 66, 0.6))'
    : 'transparent'};
  text-decoration: none;
  font-weight: ${props => props.$isActive ? '600' : '500'};
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  justify-content: ${props => props.$isCollapsed ? 'center' : 'flex-start'};
  min-height: 44px;
  border: 1px solid ${props => props.$isActive ? 'rgba(255, 107, 53, 0.3)' : 'transparent'};
  box-shadow: ${props => props.$isActive ? '0 4px 15px rgba(255, 107, 53, 0.2)' : 'none'};

  &:hover {
    background: ${props => props.$isActive 
      ? 'linear-gradient(135deg, rgba(255, 107, 53, 0.9), rgba(255, 140, 66, 0.7))'
      : 'rgba(255, 107, 53, 0.1)'};
    color: ${props => props.$isActive ? '#fff' : props.theme.colors.primary};
    transform: translateX(${props => props.$isCollapsed ? '0' : '6px'});
    border-color: rgba(255, 107, 53, 0.2);
    box-shadow: 0 6px 20px rgba(255, 107, 53, 0.15);
  }

  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 4px;
    background: linear-gradient(180deg, #ff6b35, #ff8c42);
    transform: scaleY(${props => props.$isActive ? 1 : 0});
    transform-origin: top;
    transition: transform 0.3s ease;
    border-radius: 0 2px 2px 0;
  }

  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at center, rgba(255, 107, 53, 0.1) 0%, transparent 70%);
    opacity: ${props => props.$isActive ? 0.5 : 0};
    transition: opacity 0.3s ease;
  }

  svg {
    width: 18px;
    height: 18px;
    opacity: ${props => props.$isActive ? 1 : 0.8};
    flex-shrink: 0;
    z-index: 2;
    position: relative;
  }
  
  .nav-text {
    display: ${props => props.$isCollapsed ? 'none' : 'block'};
    white-space: nowrap;
    z-index: 2;
    position: relative;
  }
`;

const ProgressSection = styled.div`
  background: ${props => props.theme.colors.background};
  padding: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
`;

const ProgressTitle = styled.h4`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
  font-size: 0.9rem;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: ${props => props.theme.colors.border};
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const ProgressFill = styled(motion.div)<{ progress: number }>`
  height: 100%;
  background: linear-gradient(90deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.accent});
  width: ${props => props.progress}%;
  border-radius: 4px;
`;

const ProgressText = styled.p`
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
  margin: 0;
`;

const navigationItems = [
  { path: '/', label: 'Home', icon: Home },
  { path: '/learning', label: 'Interactive Learning', icon: Brain },
  { path: '/transformers', label: 'Transformer Architecture', icon: Layers },
  { path: '/llms', label: 'Large Language Models', icon: Brain },
  { path: '/diffusion', label: 'Diffusion Models', icon: Wand2 },
  { path: '/multimodal', label: 'Multimodal AI', icon: Eye },
  { path: '/aws', label: 'Production & Deployment', icon: Cloud },
];

const practiceItems = [
  { path: '/quiz', label: 'Knowledge Assessment', icon: HelpCircle },
  { path: '/progress', label: 'Learning Analytics', icon: BarChart },
  { path: '/interview-sim', label: 'Interview Simulator', icon: Target },
  { path: '/teams', label: 'Team Management', icon: Users },
];

interface SidebarProps {}

const Sidebar: React.FC<SidebarProps> = () => {
  const location = useLocation();
  const { isCollapsed, setIsCollapsed, isMobileOpen, setIsMobileOpen } = useSidebar();

  return (
    <>
      <MobileMenuButton onClick={() => setIsMobileOpen(!isMobileOpen)}>
        {isMobileOpen ? <X /> : <Menu />}
      </MobileMenuButton>

      <AnimatePresence>
        {isMobileOpen && (
          <MobileOverlay
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsMobileOpen(false)}
          />
        )}
      </AnimatePresence>

      <SidebarContainer
        $isCollapsed={isCollapsed}
        $isMobileOpen={isMobileOpen}
        initial={{ x: -280 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      >
        <SidebarHeader $isCollapsed={isCollapsed}>
          <Logo $isCollapsed={isCollapsed}>
            <div className="logo-icon">P</div>
            <div className="logo-text">
              Prep<span>Me</span>
            </div>
          </Logo>
          <CollapseButton 
            $isCollapsed={isCollapsed}
            onClick={() => setIsCollapsed(!isCollapsed)}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <Menu size={16} />
          </CollapseButton>
        </SidebarHeader>

        <SidebarSection>
          <SectionTitle $isCollapsed={isCollapsed}>Core Topics</SectionTitle>
          <NavList>
            {navigationItems.map((item, index) => (
              <NavItem key={item.path}>
                <NavLink 
                  to={item.path} 
                  $isActive={location.pathname === item.path}
                  $isCollapsed={isCollapsed}
                  title={isCollapsed ? item.label : undefined}
                >
                  <item.icon />
                  <span className="nav-text">{item.label}</span>
                </NavLink>
              </NavItem>
            ))}
          </NavList>
        </SidebarSection>

        <SidebarSection>
          <SectionTitle $isCollapsed={isCollapsed}>Practice & Analytics</SectionTitle>
          <NavList>
            {practiceItems.map((item, index) => (
              <NavItem key={item.path}>
                <NavLink 
                  to={item.path} 
                  $isActive={location.pathname === item.path}
                  $isCollapsed={isCollapsed}
                  title={isCollapsed ? item.label : undefined}
                >
                  <item.icon />
                  <span className="nav-text">{item.label}</span>
                </NavLink>
              </NavItem>
            ))}
          </NavList>
        </SidebarSection>

        {!isCollapsed && (
          <SidebarSection>
            <ProgressSection>
              <ProgressTitle>Study Progress</ProgressTitle>
              <ProgressBar>
                <ProgressFill 
                  progress={68}
                  initial={{ width: 0 }}
                  animate={{ width: '68%' }}
                  transition={{ duration: 1, delay: 0.5 }}
                />
              </ProgressBar>
              <ProgressText>68% Complete • 12/18 Topics</ProgressText>
            </ProgressSection>
          </SidebarSection>
        )}
      </SidebarContainer>
    </>
  );
};

export default Sidebar;
