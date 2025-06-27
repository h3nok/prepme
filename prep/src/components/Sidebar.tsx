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
  ChevronLeft,
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
  background: ${props => props.theme.colors.surface};
  border-right: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.lg};
  overflow-y: auto;
  z-index: 50;
  transition: width 0.3s ease;

  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    width: 280px;
    transform: translateX(${props => props.$isMobileOpen ? '0' : '-100%'});
    transition: transform 0.3s ease;
  }
`;

const SidebarHeader = styled.div<{ $isCollapsed: boolean }>`
  display: flex;
  align-items: center;
  justify-content: ${props => props.$isCollapsed ? 'center' : 'space-between'};
  margin-bottom: ${props => props.theme.spacing.xl};
  padding-bottom: ${props => props.theme.spacing.md};
  border-bottom: 1px solid ${props => props.theme.colors.border};
`;

const Logo = styled.div<{ $isCollapsed: boolean }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  
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
  }
  
  .logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    color: ${props => props.theme.colors.text};
    display: ${props => props.$isCollapsed ? 'none' : 'block'};
    
    span {
      color: ${props => props.theme.colors.primary};
    }
  }
`;

const CollapseButton = styled.button<{ $isCollapsed: boolean }>`
  background: transparent;
  border: 1px solid ${props => props.theme.colors.border};
  color: ${props => props.theme.colors.textSecondary};
  width: 32px;
  height: 32px;
  border-radius: ${props => props.theme.radii.sm};
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.theme.colors.surfaceLight};
    color: ${props => props.theme.colors.text};
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
  width: 48px;
  height: 48px;
  border-radius: ${props => props.theme.radii.md};
  display: none;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: ${props => props.theme.shadows.lg};
  
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
  font-size: 0.85rem;
  font-weight: 600;
  text-transform: uppercase;
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.md};
  letter-spacing: 0.5px;
  display: ${props => props.$isCollapsed ? 'none' : 'block'};
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const NavItem = styled.li<{ $isActive?: boolean }>`
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const NavLink = styled(Link)<{ $isActive?: boolean; $isCollapsed?: boolean }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.radii.md};
  color: ${props => props.$isActive ? props.theme.colors.primary : props.theme.colors.text};
  background: ${props => props.$isActive ? `${props.theme.colors.primary}15` : 'transparent'};
  text-decoration: none;
  font-weight: ${props => props.$isActive ? '600' : '400'};
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
  justify-content: ${props => props.$isCollapsed ? 'center' : 'flex-start'};
  min-height: 40px;

  &:hover {
    background: ${props => props.$isActive ? `${props.theme.colors.primary}20` : props.theme.colors.surfaceLight};
    color: ${props => props.theme.colors.primary};
    transform: translateX(${props => props.$isCollapsed ? '0' : '4px'});
  }

  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 3px;
    background: ${props => props.theme.colors.primary};
    transform: scaleY(${props => props.$isActive ? 1 : 0});
    transform-origin: top;
    transition: transform 0.2s ease;
  }

  svg {
    width: 18px;
    height: 18px;
    opacity: ${props => props.$isActive ? 1 : 0.7};
    flex-shrink: 0;
  }
  
  .nav-text {
    display: ${props => props.$isCollapsed ? 'none' : 'block'};
    white-space: nowrap;
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
  { path: '/', label: 'Dashboard', icon: Home },
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
          >
            <ChevronLeft style={{ transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }} />
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
