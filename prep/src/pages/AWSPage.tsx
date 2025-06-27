import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowLeft, Cloud, Server, Database, CheckCircle, Book } from 'lucide-react';
import { Link } from 'react-router-dom';

import Card from '../components/Card';
import Quiz from '../components/Quiz';

const PageHeader = styled.div`
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const BackButton = styled(Link)`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textSecondary};
  text-decoration: none;
  margin-bottom: ${props => props.theme.spacing.md};
  transition: color 0.2s ease;

  &:hover {
    color: ${props => props.theme.colors.primary};
  }
`;

const PageTitle = styled.h1`
  font-size: 2.5rem;
  font-weight: 800;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
  
  span {
    color: ${props => props.theme.colors.primary};
  }
`;

const PageDescription = styled.p`
  font-size: 1.2rem;
  color: ${props => props.theme.colors.textSecondary};
  line-height: 1.6;
  max-width: 800px;
`;

const ContentSection = styled.section`
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const SectionTitle = styled.h2`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.lg};
  font-size: 1.75rem;
  font-weight: 700;
  
  &:before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.75rem;
    background: ${props => props.theme.colors.primary};
    margin-right: ${props => props.theme.spacing.md};
    vertical-align: bottom;
  }
`;

const ConceptGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const ServicesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const ServiceCard = styled(Card)`
  .service-header {
    display: flex;
    align-items: center;
    gap: ${props => props.theme.spacing.sm};
    margin-bottom: ${props => props.theme.spacing.md};
    
    .service-icon {
      width: 40px;
      height: 40px;
      background: ${props => props.theme.colors.primary}20;
      border-radius: ${props => props.theme.radii.md};
      display: flex;
      align-items: center;
      justify-content: center;
      
      svg {
        color: ${props => props.theme.colors.primary};
      }
    }
    
    .service-name {
      font-weight: 600;
      color: ${props => props.theme.colors.primary};
      font-size: 1.1rem;
    }
  }
  
  .use-cases {
    margin-top: ${props => props.theme.spacing.md};
    
    .use-case {
      background: ${props => props.theme.colors.background};
      padding: ${props => props.theme.spacing.sm};
      margin: ${props => props.theme.spacing.xs} 0;
      border-radius: ${props => props.theme.radii.sm};
      font-size: 0.9rem;
      border-left: 2px solid ${props => props.theme.colors.accent};
    }
  }
`;

const ArchitecturePattern = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.lg} 0;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.md};
    text-align: center;
  }

  .pattern-flow {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: ${props => props.theme.spacing.md};
    margin-top: ${props => props.theme.spacing.lg};
  }

  .flow-step {
    background: ${props => props.theme.colors.background};
    padding: ${props => props.theme.spacing.lg};
    border-radius: ${props => props.theme.radii.md};
    border: 1px solid ${props => props.theme.colors.border};
    text-align: center;
    position: relative;

    .step-number {
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
      margin: 0 auto ${props => props.theme.spacing.sm};
    }

    .step-title {
      font-weight: 600;
      color: ${props => props.theme.colors.primary};
      margin-bottom: ${props => props.theme.spacing.sm};
    }

    .step-desc {
      font-size: 0.9rem;
      color: ${props => props.theme.colors.textSecondary};
    }
  }
`;

const KeyPoints = styled.ul`
  list-style: none;
  padding: 0;
  
  li {
    display: flex;
    align-items: flex-start;
    gap: ${props => props.theme.spacing.sm};
    margin-bottom: ${props => props.theme.spacing.sm};
    padding: ${props => props.theme.spacing.sm};
    background: ${props => props.theme.colors.surface};
    border-radius: ${props => props.theme.radii.md};
    border-left: 3px solid ${props => props.theme.colors.primary};
    
    svg {
      color: ${props => props.theme.colors.success};
      margin-top: 2px;
      flex-shrink: 0;
    }
  }
`;

const CodeExample = styled.pre`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.lg};
  overflow-x: auto;
  font-family: ${props => props.theme.fonts.mono};
  font-size: 0.9rem;
  line-height: 1.5;
  margin: ${props => props.theme.spacing.lg} 0;
`;

const awsQuestions = [
  {
    id: "aws-1",
    question: "Which AWS service is best for training large language models at scale?",
    options: [
      "EC2 with custom setup",
      "SageMaker Training Jobs with distributed training",
      "Lambda functions",
      "ECS containers"
    ],
    correct: 1,
    explanation: "SageMaker Training Jobs provide managed infrastructure for distributed training with automatic scaling, model parallelism, and optimized ML instances like p4d.24xlarge.",
    difficulty: "medium" as const
  },
  {
    id: "aws-2",
    question: "What is the key advantage of using Amazon Bedrock for LLM applications?",
    options: [
      "Cheaper than training custom models",
      "Access to foundation models via API without managing infrastructure",
      "Better performance than custom models",
      "Automatic model fine-tuning"
    ],
    correct: 1,
    explanation: "Bedrock provides serverless access to foundation models from companies like Anthropic, Cohere, and Stability AI without needing to provision or manage infrastructure.",
    difficulty: "easy" as const
  },
  {
    id: "aws-3",
    question: "How would you implement real-time inference for a large language model on AWS?",
    options: [
      "Use Lambda with the model in memory",
      "Deploy on SageMaker real-time endpoints with auto-scaling",
      "Use EC2 instances with load balancers",
      "Store model in S3 and load on demand"
    ],
    correct: 1,
    explanation: "SageMaker real-time endpoints provide managed hosting with auto-scaling, A/B testing, and optimized inference containers for ML models, ideal for production LLM serving.",
    difficulty: "hard" as const
  }
];

const AWSPage: React.FC = () => {
  const [showQuiz, setShowQuiz] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <PageHeader>
        <BackButton to="/">
          <ArrowLeft size={20} />
          Back to Home
        </BackButton>
        <PageTitle>
          ☁️ <span>Production</span> & MLOps
        </PageTitle>
        <PageDescription>
          Master deploying AI systems at enterprise scale. From AWS to GCP and Azure, 
          learn the tools and techniques for building production-ready AI applications that 
          scale to millions of users across different cloud platforms.
        </PageDescription>
      </PageHeader>

      <ContentSection>
        <SectionTitle>Core AWS AI Services</SectionTitle>
        <ServicesGrid>
          <ServiceCard>
            <div className="service-header">
              <div className="service-icon">
                <Server size={20} />
              </div>
              <div className="service-name">Amazon SageMaker</div>
            </div>
            <p>
              End-to-end machine learning platform for building, training, and deploying 
              models at scale. Includes notebooks, training jobs, and hosted endpoints.
            </p>
            <div className="use-cases">
              <div className="use-case">Distributed training of large models</div>
              <div className="use-case">Real-time and batch inference</div>
              <div className="use-case">Model versioning and A/B testing</div>
            </div>
          </ServiceCard>

          <ServiceCard>
            <div className="service-header">
              <div className="service-icon">
                <Cloud size={20} />
              </div>
              <div className="service-name">Amazon Bedrock</div>
            </div>
            <p>
              Fully managed service providing access to foundation models from leading 
              AI companies via API, without managing infrastructure.
            </p>
            <div className="use-cases">
              <div className="use-case">Text generation and summarization</div>
              <div className="use-case">Image generation with Stable Diffusion</div>
              <div className="use-case">Custom model fine-tuning</div>
            </div>
          </ServiceCard>

          <ServiceCard>
            <div className="service-header">
              <div className="service-icon">
                <Database size={20} />
              </div>
              <div className="service-name">Amazon Textract</div>
            </div>
            <p>
              Automatically extract text, handwriting, and data from documents using 
              machine learning, with high accuracy and structured output.
            </p>
            <div className="use-cases">
              <div className="use-case">Document digitization</div>
              <div className="use-case">Form processing automation</div>
              <div className="use-case">Table and key-value extraction</div>
            </div>
          </ServiceCard>
        </ServicesGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>MLOps Architecture Patterns</SectionTitle>
        
        <ArchitecturePattern>
          <h4>Production ML Pipeline on AWS</h4>
          <div className="pattern-flow">
            <div className="flow-step">
              <div className="step-number">1</div>
              <div className="step-title">Data Ingestion</div>
              <div className="step-desc">S3, Kinesis, DynamoDB for storing and streaming training data</div>
            </div>
            <div className="flow-step">
              <div className="step-number">2</div>
              <div className="step-title">Data Processing</div>
              <div className="step-desc">SageMaker Processing, EMR for large-scale data preparation</div>
            </div>
            <div className="flow-step">
              <div className="step-number">3</div>
              <div className="step-title">Model Training</div>
              <div className="step-desc">SageMaker Training with distributed training on GPU clusters</div>
            </div>
            <div className="flow-step">
              <div className="step-number">4</div>
              <div className="step-title">Model Validation</div>
              <div className="step-desc">Automated testing and validation pipelines</div>
            </div>
            <div className="flow-step">
              <div className="step-number">5</div>
              <div className="step-title">Model Deployment</div>
              <div className="step-desc">SageMaker Endpoints with auto-scaling and monitoring</div>
            </div>
            <div className="flow-step">
              <div className="step-number">6</div>
              <div className="step-title">Monitoring</div>
              <div className="step-desc">CloudWatch, SageMaker Model Monitor for drift detection</div>
            </div>
          </div>
        </ArchitecturePattern>
      </ContentSection>

      <ContentSection>
        <SectionTitle>SageMaker Deep Dive</SectionTitle>
        <ConceptGrid>
          <Card>
            <h4>Training at Scale</h4>
            <p>
              <strong>Distributed Training:</strong> Model and data parallelism across multiple 
              GPU instances. Support for frameworks like PyTorch DDP and Horovod.
            </p>
            <CodeExample>{`# SageMaker PyTorch Estimator
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p4d.24xlarge',
    instance_count=4,  # 4 nodes
    framework_version='1.12',
    py_version='py38',
    distribution={
        'pytorchddp': {
            'enabled': True
        }
    }
)

estimator.fit({'training': 's3://bucket/data/'})`}</CodeExample>
          </Card>

          <Card variant="accent">
            <h4>Real-time Inference</h4>
            <p>
              <strong>Multi-Model Endpoints:</strong> Host multiple models on a single endpoint 
              with automatic scaling and load balancing.
            </p>
            <CodeExample>{`# Deploy model to endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='llm-endpoint'
)

# Make predictions
response = predictor.predict({
    "inputs": "Translate to French: Hello world",
    "parameters": {
        "max_length": 100,
        "temperature": 0.7
    }
})`}</CodeExample>
          </Card>

          <Card variant="purple">
            <h4>Batch Transform</h4>
            <p>
              <strong>Large-scale Inference:</strong> Process massive datasets using managed 
              batch transform jobs with automatic scaling.
            </p>
            <CodeExample>{`# Batch transform job
transformer = estimator.transformer(
    instance_count=2,
    instance_type='ml.m5.xlarge',
    output_path='s3://bucket/output/'
)

transformer.transform(
    data='s3://bucket/input-data/',
    content_type='application/json',
    split_type='Line'
)`}</CodeExample>
          </Card>
        </ConceptGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Cost Optimization Strategies</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Spot Instances:</strong> Use spot instances for training jobs to reduce costs by up to 90%</li>
          <li><CheckCircle size={16} /><strong>Auto Scaling:</strong> Configure endpoint auto-scaling based on invocation metrics and latency</li>
          <li><CheckCircle size={16} /><strong>Multi-Model Endpoints:</strong> Host multiple models on single instance to improve utilization</li>
          <li><CheckCircle size={16} /><strong>Model Compression:</strong> Use techniques like quantization and pruning to reduce inference costs</li>
          <li><CheckCircle size={16} /><strong>Reserved Instances:</strong> Commit to long-term usage for predictable workloads</li>
          <li><CheckCircle size={16} /><strong>Lifecycle Policies:</strong> Automatically transition data to cheaper storage classes</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Security & Compliance</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>VPC Configuration:</strong> Deploy SageMaker in private subnets with security groups</li>
          <li><CheckCircle size={16} /><strong>IAM Roles:</strong> Least privilege access with fine-grained permissions</li>
          <li><CheckCircle size={16} /><strong>Encryption:</strong> Data encryption at rest and in transit using KMS</li>
          <li><CheckCircle size={16} /><strong>Network Isolation:</strong> VPC endpoints and private connectivity</li>
          <li><CheckCircle size={16} /><strong>Audit Logging:</strong> CloudTrail and CloudWatch for comprehensive logging</li>
          <li><CheckCircle size={16} /><strong>Compliance:</strong> SOC, HIPAA, GDPR compliance certifications</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Interview Focus Areas</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Architecture Design:</strong> How to design scalable ML systems on AWS</li>
          <li><CheckCircle size={16} /><strong>Cost Management:</strong> Strategies for optimizing ML workload costs</li>
          <li><CheckCircle size={16} /><strong>Performance Optimization:</strong> Techniques for improving training and inference speed</li>
          <li><CheckCircle size={16} /><strong>Monitoring & Observability:</strong> Setting up comprehensive ML monitoring</li>
          <li><CheckCircle size={16} /><strong>Security Best Practices:</strong> Securing ML workloads and data</li>
          <li><CheckCircle size={16} /><strong>Disaster Recovery:</strong> Backup and recovery strategies for ML systems</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Test Your Knowledge</SectionTitle>
        {!showQuiz ? (
          <Card>
            <h4><Book />Ready for the Quiz?</h4>
            <p>Test your understanding of AWS ML services with these challenging questions.</p>
            <button
              onClick={() => setShowQuiz(true)}
              style={{
                background: '#ff6b35',
                color: 'white',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: '600',
                cursor: 'pointer',
                marginTop: '1rem'
              }}
            >
              Start Quiz
            </button>
          </Card>
        ) : (
          <Quiz 
            questions={awsQuestions}
          />
        )}
      </ContentSection>
    </motion.div>
  );
};

export default AWSPage;
