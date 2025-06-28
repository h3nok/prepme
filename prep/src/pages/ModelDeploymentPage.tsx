import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Cloud,
  Server,
  Settings,
  Monitor,
  Zap,
  Shield,
  DollarSign,
  TrendingUp,
  CheckCircle,
  AlertTriangle,
  Play,
  Code,
  Database,
  GitBranch
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

const ContentGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.xl};
  margin-bottom: ${props => props.theme.spacing.xxl};

  @media (max-width: ${props => props.theme.breakpoints.tablet}) {
    grid-template-columns: 1fr;
  }
`;

const DeploymentCard = styled(motion.div)`
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

const StepList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const StepItem = styled.li`
  padding: ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  margin-bottom: ${props => props.theme.spacing.md};
  border: 1px solid ${props => props.theme.colors.border};
  display: flex;
  align-items: flex-start;
  gap: ${props => props.theme.spacing.md};
`;

const StepNumber = styled.div`
  background: ${props => props.theme.colors.primary};
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: 600;
  flex-shrink: 0;
`;

const StepContent = styled.div`
  flex: 1;
`;

const StepTitle = styled.h4`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const StepDescription = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  line-height: 1.5;
`;

const CodeBlock = styled.pre`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.lg};
  font-family: ${props => props.theme.fonts.mono};
  font-size: 0.9rem;
  overflow-x: auto;
  color: ${props => props.theme.colors.text};
  margin: ${props => props.theme.spacing.md} 0;
`;

const TabContainer = styled.div`
  margin-bottom: ${props => props.theme.spacing.xxl};
`;

const TabList = styled.div`
  display: flex;
  border-bottom: 1px solid ${props => props.theme.colors.border};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const Tab = styled.button<{ $isActive: boolean }>`
  background: ${props => props.$isActive ? props.theme.colors.surface : 'transparent'};
  border: none;
  border-bottom: 2px solid ${props => props.$isActive ? props.theme.colors.primary : 'transparent'};
  color: ${props => props.$isActive ? props.theme.colors.text : props.theme.colors.textSecondary};
  padding: ${props => props.theme.spacing.lg} ${props => props.theme.spacing.xl};
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;

  &:hover {
    color: ${props => props.theme.colors.primary};
  }
`;

const TabContent = styled(motion.div)`
  min-height: 400px;
`;

const MetricGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const MetricCard = styled.div`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.lg};
  text-align: center;
`;

const MetricValue = styled.div`
  font-size: 2rem;
  font-weight: 800;
  color: ${props => props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const MetricLabel = styled.div`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
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

const AlertContent = styled.div`
  flex: 1;
`;

const AlertTitle = styled.h4`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const AlertText = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  line-height: 1.5;
  margin: 0;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  flex-wrap: wrap;
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

const deploymentSteps = {
  sagemaker: [
    {
      title: "Model Preparation",
      description: "Prepare your model artifacts and create a model package"
    },
    {
      title: "Create Model",
      description: "Register the model in SageMaker Model Registry"
    },
    {
      title: "Endpoint Configuration",
      description: "Configure instance types, auto-scaling, and data capture"
    },
    {
      title: "Deploy Endpoint",
      description: "Create and deploy the real-time inference endpoint"
    },
    {
      title: "Test & Monitor",
      description: "Validate deployment and set up monitoring dashboards"
    }
  ],
  bedrock: [
    {
      title: "Model Selection",
      description: "Choose from foundation models available in Bedrock"
    },
    {
      title: "API Integration",
      description: "Integrate with Bedrock APIs for inference"
    },
    {
      title: "Fine-tuning (Optional)",
      description: "Customize models with your specific data"
    },
    {
      title: "Security Setup",
      description: "Configure IAM roles and VPC settings"
    },
    {
      title: "Production Launch",
      description: "Deploy with monitoring and cost optimization"
    }
  ],
  kubernetes: [
    {
      title: "Containerization",
      description: "Package your model in Docker containers"
    },
    {
      title: "Model Server Setup",
      description: "Configure serving framework (TorchServe, TensorFlow Serving)"
    },
    {
      title: "Kubernetes Manifests",
      description: "Create deployment, service, and ingress configurations"
    },
    {
      title: "Horizontal Pod Autoscaling",
      description: "Configure auto-scaling based on CPU/GPU metrics"
    },
    {
      title: "Production Deployment",
      description: "Deploy with rolling updates and health checks"
    }
  ]
};

export const ModelDeploymentPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('sagemaker');

  const tabItems = [
    { id: 'sagemaker', label: 'SageMaker Deployment', icon: Cloud },
    { id: 'bedrock', label: 'Amazon Bedrock', icon: Shield },
    { id: 'kubernetes', label: 'Kubernetes', icon: Server }
  ];

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Model Deployment</PageTitle>
        <PageDescription>
          Learn production-ready model deployment strategies, from cloud-native solutions to 
          self-managed infrastructure with monitoring and optimization
        </PageDescription>
      </PageHeader>

      <ContentGrid>
        <DeploymentCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <CardHeader $color="#ff6b35">
            <CardTitle>
              <Cloud size={24} />
              Cloud Deployment
            </CardTitle>
            <CardDescription>
              Leverage managed services like AWS SageMaker and Bedrock for scalable, 
              production-ready model deployment with minimal infrastructure management.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AlertBox $type="success">
              <CheckCircle size={20} color="#10b981" />
              <AlertContent>
                <AlertTitle>Recommended for Most Cases</AlertTitle>
                <AlertText>
                  Managed services provide automatic scaling, monitoring, and maintenance
                  with enterprise-grade security and compliance.
                </AlertText>
              </AlertContent>
            </AlertBox>
            <ButtonGroup>
              <ActionButton $variant="primary">
                <Play size={16} />
                Start SageMaker Tutorial
              </ActionButton>
              <ActionButton>
                <Code size={16} />
                View Examples
              </ActionButton>
            </ButtonGroup>
          </CardContent>
        </DeploymentCard>

        <DeploymentCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <CardHeader $color="#059669">
            <CardTitle>
              <Server size={24} />
              Self-Managed
            </CardTitle>
            <CardDescription>
              Deploy on your own infrastructure using containers, Kubernetes, 
              and custom serving solutions for maximum control and customization.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AlertBox $type="warning">
              <AlertTriangle size={20} color="#f59e0b" />
              <AlertContent>
                <AlertTitle>Requires Expertise</AlertTitle>
                <AlertText>
                  Self-managed deployments require deep DevOps knowledge and ongoing
                  maintenance but offer complete control over the infrastructure.
                </AlertText>
              </AlertContent>
            </AlertBox>
            <ButtonGroup>
              <ActionButton $variant="primary">
                <GitBranch size={16} />
                Kubernetes Guide
              </ActionButton>
              <ActionButton>
                <Database size={16} />
                Docker Setup
              </ActionButton>
            </ButtonGroup>
          </CardContent>
        </DeploymentCard>
      </ContentGrid>

      <TabContainer>
        <TabList>
          {tabItems.map((tab) => (
            <Tab
              key={tab.id}
              $isActive={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            >
              <tab.icon size={18} style={{ marginRight: '8px' }} />
              {tab.label}
            </Tab>
          ))}
        </TabList>

        <TabContent
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          <h3 style={{ marginBottom: '1.5rem' }}>
            {tabItems.find(t => t.id === activeTab)?.label} Deployment Steps
          </h3>
          
          <StepList>
            {deploymentSteps[activeTab as keyof typeof deploymentSteps].map((step, index) => (
              <StepItem key={index}>
                <StepNumber>{index + 1}</StepNumber>
                <StepContent>
                  <StepTitle>{step.title}</StepTitle>
                  <StepDescription>{step.description}</StepDescription>
                </StepContent>
              </StepItem>
            ))}
          </StepList>

          {activeTab === 'sagemaker' && (
            <>
              <h4 style={{ margin: '2rem 0 1rem' }}>Sample SageMaker Deployment Code</h4>
              <CodeBlock>
{`import boto3
from sagemaker import Model
from sagemaker.predictor import Predictor

# Create SageMaker model
model = Model(
    image_uri="your-model-image-uri",
    model_data="s3://your-bucket/model.tar.gz",
    role="arn:aws:iam::account:role/SageMakerRole"
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="llm-inference-endpoint"
)

# Make prediction
result = predictor.predict({
    "prompt": "Explain transformer architecture",
    "max_tokens": 512
})`}
              </CodeBlock>
            </>
          )}

          {activeTab === 'bedrock' && (
            <>
              <h4 style={{ margin: '2rem 0 1rem' }}>Bedrock API Integration</h4>
              <CodeBlock>
{`import boto3
import json

bedrock = boto3.client('bedrock-runtime')

# Invoke foundation model
response = bedrock.invoke_model(
    modelId='anthropic.claude-v2',
    contentType='application/json',
    accept='application/json',
    body=json.dumps({
        "prompt": "\\n\\nHuman: Explain machine learning\\n\\nAssistant:",
        "max_tokens_to_sample": 1000,
        "temperature": 0.7
    })
)

result = json.loads(response['body'].read())
print(result['completion'])`}
              </CodeBlock>
            </>
          )}

          {activeTab === 'kubernetes' && (
            <>
              <h4 style={{ margin: '2rem 0 1rem' }}>Kubernetes Deployment Manifest</h4>
              <CodeBlock>
{`apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: your-registry/model-server:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1`}
              </CodeBlock>
            </>
          )}
        </TabContent>
      </TabContainer>

      <MetricGrid>
        <MetricCard>
          <MetricValue>99.9%</MetricValue>
          <MetricLabel>Uptime SLA</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>&lt;100ms</MetricValue>
          <MetricLabel>Avg Latency</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>1000+</MetricValue>
          <MetricLabel>RPS Capacity</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>$0.50</MetricValue>
          <MetricLabel>Cost per 1K Requests</MetricLabel>
        </MetricCard>
      </MetricGrid>
    </PageContainer>
  );
};

export default ModelDeploymentPage;
