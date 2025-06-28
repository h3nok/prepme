import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  BarChart3,
  TrendingUp,
  Target,
  Clock,
  CheckCircle,
  AlertCircle,
  Brain,
  Zap,
  Award,
  Calendar,
  User,
  Settings,
  Filter,
  Download,
  Layers,
  Eye
} from 'lucide-react';

const PageContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing.xl};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    padding: ${props => props.theme.spacing.lg};
  }
`;

const PageHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.xxl};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    flex-direction: column;
    gap: ${props => props.theme.spacing.lg};
    text-align: center;
  }
`;

const HeaderContent = styled.div`
  flex: 1;
`;

const PageTitle = styled.h1`
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.accent});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: ${props => props.theme.spacing.sm};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    font-size: 2rem;
  }
`;

const PageDescription = styled.p`
  font-size: 1.25rem;
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
`;

const HeaderActions = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
`;

const ActionButton = styled(motion.button)<{ $variant?: 'primary' | 'secondary' }>`
  background: ${props => 
    props.$variant === 'primary' 
      ? `linear-gradient(135deg, ${props.theme.colors.primary}, ${props.theme.colors.accent})`
      : props.theme.colors.surface};
  color: ${props => props.$variant === 'primary' ? 'white' : props.theme.colors.text};
  border: 1px solid ${props => props.$variant === 'primary' ? 'transparent' : props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.lg};
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.md};
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const StatCard = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.xl};
  text-align: center;
  box-shadow: ${props => props.theme.shadows.lg};
`;

const StatIcon = styled.div<{ $color: string }>`
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, ${props => props.$color}20, ${props => props.$color}10);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto ${props => props.theme.spacing.lg};
  color: ${props => props.$color};
`;

const StatValue = styled.div`
  font-size: 2.5rem;
  font-weight: 800;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const StatLabel = styled.div`
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const StatChange = styled.div<{ $isPositive: boolean }>`
  color: ${props => props.$isPositive ? props.theme.colors.success : props.theme.colors.error};
  font-size: 0.9rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${props => props.theme.spacing.xs};
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: ${props => props.theme.spacing.xl};
  margin-bottom: ${props => props.theme.spacing.xxl};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-template-columns: 1fr;
  }
`;

const ChartCard = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  border: 1px solid ${props => props.theme.colors.border};
  overflow: hidden;
  box-shadow: ${props => props.theme.shadows.lg};
`;

const ChartHeader = styled.div`
  padding: ${props => props.theme.spacing.xl};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ChartTitle = styled.h3`
  color: ${props => props.theme.colors.text};
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const ChartContent = styled.div`
  padding: ${props => props.theme.spacing.xl};
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: ${props => props.theme.colors.background};
`;

const ProgressSection = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.xl};
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const SectionTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.lg};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const ModuleList = styled.div`
  display: grid;
  gap: ${props => props.theme.spacing.lg};
`;

const ModuleItem = styled.div`
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.lg};
`;

const ModuleHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const ModuleName = styled.h4`
  color: ${props => props.theme.colors.text};
  margin: 0;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const ModuleStats = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.lg};
  font-size: 0.9rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: ${props => props.theme.colors.border};
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const ProgressFill = styled(motion.div)<{ $progress: number; $color: string }>`
  height: 100%;
  background: linear-gradient(90deg, ${props => props.$color}, ${props => props.$color}cc);
  width: ${props => props.$progress}%;
  border-radius: 4px;
`;

const ProgressText = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const ActivityFeed = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.xl};
`;

const ActivityItem = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  padding: ${props => props.theme.spacing.md} 0;
  border-bottom: 1px solid ${props => props.theme.colors.border};

  &:last-child {
    border-bottom: none;
  }
`;

const ActivityIcon = styled.div<{ $color: string }>`
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, ${props => props.$color}20, ${props => props.$color}10);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.$color};
  flex-shrink: 0;
`;

const ActivityContent = styled.div`
  flex: 1;
`;

const ActivityTitle = styled.div`
  color: ${props => props.theme.colors.text};
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const ActivityDescription = styled.div`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const ActivityTime = styled.div`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.8rem;
`;

const moduleData = [
  {
    name: 'Transformer Architecture',
    icon: Layers,
    progress: 85,
    timeSpent: '12h 30m',
    quizScore: '92%',
    color: '#7c3aed'
  },
  {
    name: 'Large Language Models',
    icon: Brain,
    progress: 92,
    timeSpent: '18h 45m',
    quizScore: '88%',
    color: '#059669'
  },
  {
    name: 'Diffusion Models',
    icon: Zap,
    progress: 67,
    timeSpent: '8h 15m',
    quizScore: '85%',
    color: '#dc2626'
  },
  {
    name: 'Multimodal AI',
    icon: Eye,
    progress: 45,
    timeSpent: '5h 20m',
    quizScore: '78%',
    color: '#0ea5e9'
  }
];

const activityData = [
  {
    title: 'Completed LLMs Quiz',
    description: 'Scored 88% on Advanced Large Language Models assessment',
    time: '2 hours ago',
    icon: CheckCircle,
    color: '#10b981'
  },
  {
    title: 'Started Multimodal AI Module',
    description: 'Began studying Vision-Language models and CLIP architecture',
    time: '1 day ago',
    icon: Brain,
    color: '#0ea5e9'
  },
  {
    title: 'Interview Simulation',
    description: 'Practiced system design interview for ML recommendation systems',
    time: '2 days ago',
    icon: Target,
    color: '#ff6b35'
  },
  {
    title: 'Achievement Unlocked',
    description: 'Earned "Transformer Expert" badge for completing all transformer modules',
    time: '3 days ago',
    icon: Award,
    color: '#7c3aed'
  }
];

export const LearningAnalyticsPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('7d');

  return (
    <PageContainer>
      <PageHeader>
        <HeaderContent>
          <PageTitle>Learning Analytics</PageTitle>
          <PageDescription>
            Track your progress, analyze performance, and optimize your learning journey
          </PageDescription>
        </HeaderContent>
        <HeaderActions>
          <ActionButton>
            <Filter size={16} />
            Filter
          </ActionButton>
          <ActionButton>
            <Download size={16} />
            Export
          </ActionButton>
          <ActionButton $variant="primary">
            <Settings size={16} />
            Settings
          </ActionButton>
        </HeaderActions>
      </PageHeader>

      <StatsGrid>
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <StatIcon $color="#10b981">
            <TrendingUp size={24} />
          </StatIcon>
          <StatValue>73%</StatValue>
          <StatLabel>Overall Progress</StatLabel>
          <StatChange $isPositive={true}>
            <TrendingUp size={16} />
            +12% this week
          </StatChange>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <StatIcon $color="#ff6b35">
            <Clock size={24} />
          </StatIcon>
          <StatValue>44h</StatValue>
          <StatLabel>Study Time</StatLabel>
          <StatChange $isPositive={true}>
            <TrendingUp size={16} />
            +8h this week
          </StatChange>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <StatIcon $color="#7c3aed">
            <Target size={24} />
          </StatIcon>
          <StatValue>86%</StatValue>
          <StatLabel>Quiz Average</StatLabel>
          <StatChange $isPositive={true}>
            <TrendingUp size={16} />
            +4% this month
          </StatChange>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <StatIcon $color="#059669">
            <Award size={24} />
          </StatIcon>
          <StatValue>7</StatValue>
          <StatLabel>Badges Earned</StatLabel>
          <StatChange $isPositive={true}>
            <TrendingUp size={16} />
            +2 this month
          </StatChange>
        </StatCard>
      </StatsGrid>

      <ChartsGrid>
        <ChartCard
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <ChartHeader>
            <ChartTitle>
              <BarChart3 size={20} />
              Weekly Progress
            </ChartTitle>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              style={{
                background: 'var(--background)',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                padding: '8px 12px',
                color: 'var(--text)'
              }}
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
          </ChartHeader>
          <ChartContent>
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
              📊 Interactive progress chart would be displayed here
              <br />
              <small>Showing study hours, quiz scores, and module completion over time</small>
            </div>
          </ChartContent>
        </ChartCard>

        <ChartCard
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <ChartHeader>
            <ChartTitle>
              <Target size={20} />
              Skill Breakdown
            </ChartTitle>
          </ChartHeader>
          <ChartContent>
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
              🎯 Skill radar chart would be displayed here
              <br />
              <small>Showing proficiency in different AI/ML areas</small>
            </div>
          </ChartContent>
        </ChartCard>
      </ChartsGrid>

      <ProgressSection>
        <SectionTitle>
          <Brain size={24} />
          Module Progress
        </SectionTitle>
        <ModuleList>
          {moduleData.map((module, index) => (
            <ModuleItem key={index}>
              <ModuleHeader>
                <ModuleName>
                  <module.icon size={20} />
                  {module.name}
                </ModuleName>
                <ModuleStats>
                  <span>⏱️ {module.timeSpent}</span>
                  <span>📝 {module.quizScore}</span>
                </ModuleStats>
              </ModuleHeader>
              <ProgressBar>
                <ProgressFill
                  $progress={module.progress}
                  $color={module.color}
                  initial={{ width: 0 }}
                  animate={{ width: `${module.progress}%` }}
                  transition={{ duration: 1, delay: index * 0.1 }}
                />
              </ProgressBar>
              <ProgressText>
                <span>{module.progress}% Complete</span>
                <span>{100 - module.progress}% Remaining</span>
              </ProgressText>
            </ModuleItem>
          ))}
        </ModuleList>
      </ProgressSection>

      <ActivityFeed>
        <SectionTitle>
          <Calendar size={24} />
          Recent Activity
        </SectionTitle>
        {activityData.map((activity, index) => (
          <ActivityItem key={index}>
            <ActivityIcon $color={activity.color}>
              <activity.icon size={20} />
            </ActivityIcon>
            <ActivityContent>
              <ActivityTitle>{activity.title}</ActivityTitle>
              <ActivityDescription>{activity.description}</ActivityDescription>
              <ActivityTime>{activity.time}</ActivityTime>
            </ActivityContent>
          </ActivityItem>
        ))}
      </ActivityFeed>
    </PageContainer>
  );
};

export default LearningAnalyticsPage;
