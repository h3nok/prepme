import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  TrendingUp,
  Zap,
  DollarSign,
  Monitor,
  Settings,
  BarChart3,
  Cpu,
  HardDrive,
  Activity,
  Clock,
  Target,
  Layers,
  CheckCircle,
  AlertTriangle,
  Info
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
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const PageTitle = styled.h1`
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.accent});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: ${props => props.theme.spacing.md};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    font-size: 2rem;
  }
`;

const PageDescription = styled.p`
  font-size: 1.25rem;
  color: ${props => props.theme.colors.textSecondary};
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
`;

const OptimizationGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: ${props => props.theme.spacing.xl};
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const OptimizationCard = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  border: 1px solid ${props => props.theme.colors.border};
  overflow: hidden;
  box-shadow: ${props => props.theme.shadows.lg};
`;

const CardHeader = styled.div<{ $color: string }>`
  background: linear-gradient(135deg, ${props => props.$color}20, ${props => props.$color}10);
  padding: ${props => props.theme.spacing.xl};
  border-bottom: 1px solid ${props => props.theme.colors.border};
`;

const CardTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const CardDescription = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
`;

const CardContent = styled.div`
  padding: ${props => props.theme.spacing.xl};
`;

const TechniqueList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const TechniqueItem = styled.li`
  padding: ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  margin-bottom: ${props => props.theme.spacing.md};
  border: 1px solid ${props => props.theme.colors.border};
`;

const TechniqueName = styled.h4`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xs};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const TechniqueDescription = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  line-height: 1.5;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const TechniqueMetrics = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  font-size: 0.8rem;
`;

const MetricBadge = styled.span<{ $type: 'performance' | 'cost' | 'memory' }>`
  background: ${props => {
    if (props.$type === 'performance') return `${props.theme.colors.success}20`;
    if (props.$type === 'cost') return `${props.theme.colors.primary}20`;
    return `${props.theme.colors.accent}20`;
  }};
  color: ${props => {
    if (props.$type === 'performance') return props.theme.colors.success;
    if (props.$type === 'cost') return props.theme.colors.primary;
    return props.theme.colors.accent;
  }};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.sm};
  font-weight: 600;
`;

const InteractiveSection = styled.div`
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

const CalculatorGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.xl};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-template-columns: 1fr;
  }
`;

const InputGroup = styled.div`
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const Label = styled.label`
  display: block;
  color: ${props => props.theme.colors.text};
  font-weight: 600;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const Input = styled.input`
  width: 100%;
  padding: ${props => props.theme.spacing.md};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text};
  font-size: 1rem;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const Select = styled.select`
  width: 100%;
  padding: ${props => props.theme.spacing.md};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text};
  font-size: 1rem;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const ResultsPanel = styled.div`
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.lg};
`;

const ResultItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${props => props.theme.spacing.md} 0;
  border-bottom: 1px solid ${props => props.theme.colors.border};

  &:last-child {
    border-bottom: none;
  }
`;

const ResultLabel = styled.span`
  color: ${props => props.theme.colors.textSecondary};
`;

const ResultValue = styled.span`
  color: ${props => props.theme.colors.text};
  font-weight: 600;
`;

const ComparisonTable = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  overflow: hidden;
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const TableHeader = styled.div`
  background: ${props => props.theme.colors.background};
  padding: ${props => props.theme.spacing.lg};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
  gap: ${props => props.theme.spacing.md};
  font-weight: 600;
  color: ${props => props.theme.colors.text};
`;

const TableRow = styled.div`
  padding: ${props => props.theme.spacing.lg};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
  gap: ${props => props.theme.spacing.md};
  align-items: center;

  &:last-child {
    border-bottom: none;
  }
`;

const TechniqueName2 = styled.div`
  color: ${props => props.theme.colors.text};
  font-weight: 600;
`;

const MetricCell = styled.div<{ $isGood?: boolean }>`
  color: ${props => props.$isGood ? props.theme.colors.success : props.theme.colors.textSecondary};
  font-weight: 500;
`;

const AlertBox = styled.div<{ $type: 'info' | 'warning' | 'success' }>`
  background: ${props => {
    if (props.$type === 'warning') return `${props.theme.colors.warning}20`;
    if (props.$type === 'success') return `${props.theme.colors.success}20`;
    return `${props.theme.colors.primary}20`;
  }};
  border: 1px solid ${props => {
    if (props.$type === 'warning') return props.theme.colors.warning;
    if (props.$type === 'success') return props.theme.colors.success;
    return props.theme.colors.primary;
  }};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.md} 0;
  display: flex;
  align-items: flex-start;
  gap: ${props => props.theme.spacing.md};
`;

const optimizationTechniques = {
  performance: [
    {
      name: "Model Quantization",
      description: "Reduce model size by converting weights from FP32 to INT8/INT4, maintaining accuracy while improving speed",
      metrics: { performance: "2-4x faster", memory: "75% reduction", cost: "50% savings" }
    },
    {
      name: "Dynamic Batching",
      description: "Automatically batch multiple requests to improve throughput and GPU utilization",
      metrics: { performance: "3-5x throughput", memory: "Constant", cost: "60% reduction" }
    },
    {
      name: "KV-Cache Optimization",
      description: "Cache key-value pairs in attention layers to avoid recomputation during generation",
      metrics: { performance: "10-50x faster", memory: "2x increase", cost: "30% reduction" }
    }
  ],
  cost: [
    {
      name: "Spot Instances",
      description: "Use spare AWS capacity at up to 90% discount for training workloads",
      metrics: { performance: "Same", memory: "Same", cost: "90% savings" }
    },
    {
      name: "Multi-Model Endpoints",
      description: "Host multiple models on a single endpoint, loading them on-demand",
      metrics: { performance: "Cold start latency", memory: "Shared", cost: "80% reduction" }
    },
    {
      name: "Auto Scaling",
      description: "Automatically scale endpoints based on traffic patterns",
      metrics: { performance: "Variable", memory: "Dynamic", cost: "40% savings" }
    }
  ],
  memory: [
    {
      name: "Gradient Checkpointing",
      description: "Trade computation for memory by recomputing activations during backprop",
      metrics: { performance: "20% slower", memory: "50% reduction", cost: "30% savings" }
    },
    {
      name: "Model Parallelism",
      description: "Split large models across multiple GPUs or nodes",
      metrics: { performance: "Same", memory: "Linear scaling", cost: "Same" }
    },
    {
      name: "Offloading",
      description: "Move inactive model layers to CPU/disk memory",
      metrics: { performance: "10% slower", memory: "80% reduction", cost: "60% savings" }
    }
  ]
};

const comparisonData = [
  {
    technique: "Baseline (FP32)",
    latency: "500ms",
    throughput: "2 req/s",
    memory: "16GB",
    cost: "$100/hour"
  },
  {
    technique: "INT8 Quantization",
    latency: "125ms",
    throughput: "8 req/s",
    memory: "4GB",
    cost: "$25/hour"
  },
  {
    technique: "Dynamic Batching",
    latency: "200ms",
    throughput: "10 req/s",
    memory: "16GB",
    cost: "$40/hour"
  },
  {
    technique: "Combined Optimizations",
    latency: "100ms",
    throughput: "20 req/s",
    memory: "4GB",
    cost: "$20/hour"
  }
];

export const ScalingOptimizationPage: React.FC = () => {
  const [requestsPerSecond, setRequestsPerSecond] = useState(100);
  const [modelSize, setModelSize] = useState(7);
  const [instanceType, setInstanceType] = useState('ml.g4dn.xlarge');

  const calculateCosts = () => {
    const instanceCosts: { [key: string]: number } = {
      'ml.g4dn.xlarge': 0.526,
      'ml.g4dn.2xlarge': 0.752,
      'ml.g5.xlarge': 1.006,
      'ml.g5.2xlarge': 1.212
    };

    const baseHourlyCost = instanceCosts[instanceType] || 1.0;
    const dailyCost = baseHourlyCost * 24;
    const monthlyCost = dailyCost * 30;
    const estimatedLatency = Math.max(50, 500 - (requestsPerSecond * 2));
    const maxThroughput = Math.floor(3600 / (estimatedLatency / 1000));

    return {
      hourlyCost: baseHourlyCost,
      dailyCost,
      monthlyCost,
      estimatedLatency,
      maxThroughput
    };
  };

  const costs = calculateCosts();

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Scaling & Optimization</PageTitle>
        <PageDescription>
          Master performance optimization, cost reduction, and scaling strategies for 
          production ML systems with real-world techniques and calculations
        </PageDescription>
      </PageHeader>

      <OptimizationGrid>
        <OptimizationCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <CardHeader $color="#10b981">
            <CardTitle>
              <Zap size={24} />
              Performance Optimization
            </CardTitle>
            <CardDescription>
              Techniques to improve inference speed, reduce latency, and increase throughput
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TechniqueList>
              {optimizationTechniques.performance.map((technique, index) => (
                <TechniqueItem key={index}>
                  <TechniqueName>
                    <Activity size={16} />
                    {technique.name}
                  </TechniqueName>
                  <TechniqueDescription>
                    {technique.description}
                  </TechniqueDescription>
                  <TechniqueMetrics>
                    <MetricBadge $type="performance">{technique.metrics.performance}</MetricBadge>
                    <MetricBadge $type="memory">{technique.metrics.memory}</MetricBadge>
                    <MetricBadge $type="cost">{technique.metrics.cost}</MetricBadge>
                  </TechniqueMetrics>
                </TechniqueItem>
              ))}
            </TechniqueList>
          </CardContent>
        </OptimizationCard>

        <OptimizationCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <CardHeader $color="#ff6b35">
            <CardTitle>
              <DollarSign size={24} />
              Cost Optimization
            </CardTitle>
            <CardDescription>
              Strategies to reduce infrastructure costs while maintaining performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TechniqueList>
              {optimizationTechniques.cost.map((technique, index) => (
                <TechniqueItem key={index}>
                  <TechniqueName>
                    <TrendingUp size={16} />
                    {technique.name}
                  </TechniqueName>
                  <TechniqueDescription>
                    {technique.description}
                  </TechniqueDescription>
                  <TechniqueMetrics>
                    <MetricBadge $type="performance">{technique.metrics.performance}</MetricBadge>
                    <MetricBadge $type="memory">{technique.metrics.memory}</MetricBadge>
                    <MetricBadge $type="cost">{technique.metrics.cost}</MetricBadge>
                  </TechniqueMetrics>
                </TechniqueItem>
              ))}
            </TechniqueList>
          </CardContent>
        </OptimizationCard>

        <OptimizationCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <CardHeader $color="#7c3aed">
            <CardTitle>
              <HardDrive size={24} />
              Memory Optimization
            </CardTitle>
            <CardDescription>
              Techniques to reduce memory usage and enable larger model deployment
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TechniqueList>
              {optimizationTechniques.memory.map((technique, index) => (
                <TechniqueItem key={index}>
                  <TechniqueName>
                    <Cpu size={16} />
                    {technique.name}
                  </TechniqueName>
                  <TechniqueDescription>
                    {technique.description}
                  </TechniqueDescription>
                  <TechniqueMetrics>
                    <MetricBadge $type="performance">{technique.metrics.performance}</MetricBadge>
                    <MetricBadge $type="memory">{technique.metrics.memory}</MetricBadge>
                    <MetricBadge $type="cost">{technique.metrics.cost}</MetricBadge>
                  </TechniqueMetrics>
                </TechniqueItem>
              ))}
            </TechniqueList>
          </CardContent>
        </OptimizationCard>
      </OptimizationGrid>

      <InteractiveSection>
        <SectionTitle>
          <BarChart3 size={24} />
          Cost Calculator
        </SectionTitle>
        
        <CalculatorGrid>
          <div>
            <InputGroup>
              <Label>Expected Requests per Second</Label>
              <Input
                type="number"
                value={requestsPerSecond}
                onChange={(e) => setRequestsPerSecond(Number(e.target.value))}
                min="1"
                max="10000"
              />
            </InputGroup>

            <InputGroup>
              <Label>Model Size (Billion Parameters)</Label>
              <Select
                value={modelSize}
                onChange={(e) => setModelSize(Number(e.target.value))}
              >
                <option value={7}>7B (Llama 2 7B)</option>
                <option value={13}>13B (Llama 2 13B)</option>
                <option value={70}>70B (Llama 2 70B)</option>
                <option value={175}>175B (GPT-3)</option>
              </Select>
            </InputGroup>

            <InputGroup>
              <Label>Instance Type</Label>
              <Select
                value={instanceType}
                onChange={(e) => setInstanceType(e.target.value)}
              >
                <option value="ml.g4dn.xlarge">ml.g4dn.xlarge (1 GPU, $0.526/hr)</option>
                <option value="ml.g4dn.2xlarge">ml.g4dn.2xlarge (1 GPU, $0.752/hr)</option>
                <option value="ml.g5.xlarge">ml.g5.xlarge (1 GPU, $1.006/hr)</option>
                <option value="ml.g5.2xlarge">ml.g5.2xlarge (1 GPU, $1.212/hr)</option>
              </Select>
            </InputGroup>
          </div>

          <ResultsPanel>
            <h3 style={{ marginBottom: '1rem' }}>Estimated Costs</h3>
            <ResultItem>
              <ResultLabel>Hourly Cost:</ResultLabel>
              <ResultValue>${costs.hourlyCost.toFixed(2)}</ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Daily Cost:</ResultLabel>
              <ResultValue>${costs.dailyCost.toFixed(2)}</ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Monthly Cost:</ResultLabel>
              <ResultValue>${costs.monthlyCost.toFixed(2)}</ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Est. Latency:</ResultLabel>
              <ResultValue>{costs.estimatedLatency}ms</ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Max Throughput:</ResultLabel>
              <ResultValue>{costs.maxThroughput} req/hr</ResultValue>
            </ResultItem>
          </ResultsPanel>
        </CalculatorGrid>

        <AlertBox $type="info">
          <Info size={20} color="#ff6b35" />
          <div>
            <strong>Cost Optimization Tip:</strong> Consider using spot instances for training 
            workloads (up to 90% savings) and auto-scaling for inference endpoints to match 
            actual demand patterns.
          </div>
        </AlertBox>
      </InteractiveSection>

      <ComparisonTable>
        <h2 style={{ padding: '1.5rem', margin: 0, borderBottom: '1px solid var(--border)' }}>
          Optimization Techniques Comparison
        </h2>
        <TableHeader>
          <div>Technique</div>
          <div>Latency</div>
          <div>Throughput</div>
          <div>Memory</div>
          <div>Cost/Hour</div>
        </TableHeader>
        {comparisonData.map((row, index) => (
          <TableRow key={index}>
            <TechniqueName2>{row.technique}</TechniqueName2>
            <MetricCell $isGood={index > 0}>{row.latency}</MetricCell>
            <MetricCell $isGood={index > 0}>{row.throughput}</MetricCell>
            <MetricCell $isGood={index > 0}>{row.memory}</MetricCell>
            <MetricCell $isGood={index > 0}>{row.cost}</MetricCell>
          </TableRow>
        ))}
      </ComparisonTable>
    </PageContainer>
  );
};

export default ScalingOptimizationPage;
