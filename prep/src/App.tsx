import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import styled, { ThemeProvider, createGlobalStyle } from 'styled-components';
import 'katex/dist/katex.min.css';

// Context
import { SidebarProvider, useSidebar } from './context/SidebarContext';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import TransformersPage from './pages/TransformersPage';
import LLMsPage from './pages/LLMsPage';
import DiffusionPage from './pages/DiffusionPage';
import MultimodalPage from './pages/MultimodalPage';
import AWSPage from './pages/AWSPage';
import QuizPage from './pages/QuizPage';
import Footer from './components/Footer';

// Theme
const theme = {
  colors: {
    primary: '#ff6b35',
    secondary: '#1e3a8a',
    accent: '#059669',
    purple: '#7c3aed',
    background: '#0f172a',
    surface: '#1e293b',
    surfaceLight: '#334155',
    text: '#f8fafc',
    textSecondary: '#cbd5e1',
    border: '#475569',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
  },
  fonts: {
    main: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
    mono: '"JetBrains Mono", "Fira Code", "Consolas", monospace',
    math: '"KaTeX_Main", "Times New Roman", serif',
  },
  breakpoints: {
    mobile: '480px',
    tablet: '768px',
    desktop: '1024px',
    wide: '1200px',
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    xxl: '3rem',
  },
  radii: {
    sm: '0.375rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
  },
};

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
  
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    scroll-behavior: smooth;
  }

  body {
    font-family: ${props => props.theme.fonts.main};
    background: ${props => props.theme.colors.background};
    color: ${props => props.theme.colors.text};
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  code {
    font-family: ${props => props.theme.fonts.mono};
  }

  h1, h2, h3, h4, h5, h6 {
    font-weight: 600;
    line-height: 1.3;
    margin-bottom: 0.5em;
  }

  p {
    margin-bottom: 1em;
  }

  a {
    color: ${props => props.theme.colors.primary};
    text-decoration: none;
    transition: color 0.2s ease;

    &:hover {
      color: ${props => props.theme.colors.accent};
    }
  }

  ul, ol {
    margin-bottom: 1em;
    padding-left: 1.5em;
  }

  li {
    margin-bottom: 0.25em;
  }

  blockquote {
    border-left: 4px solid ${props => props.theme.colors.primary};
    padding-left: 1rem;
    margin: 1rem 0;
    font-style: italic;
    color: ${props => props.theme.colors.textSecondary};
  }

  pre {
    background: ${props => props.theme.colors.surface};
    border: 1px solid ${props => props.theme.colors.border};
    border-radius: ${props => props.theme.radii.md};
    padding: 1rem;
    overflow-x: auto;
    margin: 1rem 0;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.surface};
  }

  ::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.border};
    border-radius: 4px;

    &:hover {
      background: ${props => props.theme.colors.primary};
    }
  }

  /* Selection */
  ::selection {
    background: ${props => props.theme.colors.primary};
    color: white;
  }

  /* Focus styles */
  :focus {
    outline: 2px solid ${props => props.theme.colors.primary};
    outline-offset: 2px;
  }

  /* Loading animation */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  /* Utility classes */
  .animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }

  .animate-slideInUp {
    animation: slideInUp 0.6s ease-out;
  }

  .animate-fadeIn {
    animation: fadeIn 0.6s ease-out;
  }
`;

const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  background: ${props => props.theme.colors.background};
`;

const MainContent = styled(motion.main)<{ $isCollapsed: boolean }>`
  flex: 1;
  display: flex;
  flex-direction: column;
  margin-left: ${props => props.$isCollapsed ? '80px' : '280px'};
  min-height: 100vh;
  transition: margin-left 0.3s ease;

  @media (max-width: ${props => props.theme.breakpoints.desktop}) {
    margin-left: 0;
  }
`;

const ContentArea = styled.div`
  flex: 1;
  padding: ${props => props.theme.spacing.xl};
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    padding: ${props => props.theme.spacing.md};
  }
`;

const AppContent: React.FC = () => {
  const { isCollapsed } = useSidebar();

  return (
    <AppContainer>
      <Sidebar />
      <MainContent
        $isCollapsed={isCollapsed}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Header />
        <ContentArea>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/transformers" element={<TransformersPage />} />
            <Route path="/llms" element={<LLMsPage />} />
            <Route path="/diffusion" element={<DiffusionPage />} />
            <Route path="/multimodal" element={<MultimodalPage />} />
            <Route path="/aws" element={<AWSPage />} />
            <Route path="/quiz" element={<QuizPage />} />
          </Routes>
        </ContentArea>
        <Footer />
      </MainContent>
    </AppContainer>
  );
};

function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <SidebarProvider>
        <Router>
          <AppContent />
        </Router>
      </SidebarProvider>
    </ThemeProvider>
  );
}

export default App;
