import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { ArrowLeft, Brain, Code, Eye, CheckCircle, Book } from 'lucide-react';
import { Link } from 'react-router-dom';

import Card from '../components/Card';
import Math from '../components/Math';
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

const CodeBlock = styled.pre`
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

const ArchitectureDiagram = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  text-align: center;
  margin: ${props => props.theme.spacing.lg} 0;

  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.md};
  }
`;

const FormulaCard = styled(Card)`
  margin: ${props => props.theme.spacing.lg} 0;
  
  h4 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.md};
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

const transformerQuestions = [
  {
    id: "transformer-1",
    question: "What is the main innovation of the Transformer architecture?",
    options: [
      "Convolutional layers for sequence processing",
      "Self-attention mechanism for parallel processing",
      "Recurrent connections for memory",
      "Pooling layers for dimensionality reduction"
    ],
    correct: 1,
    explanation: "The Transformer's key innovation is the self-attention mechanism that allows parallel processing of sequences, eliminating the sequential bottleneck of RNNs.",
    difficulty: "medium" as const
  },
  {
    id: "transformer-2",
    question: "In the attention formula, what does the √d_k normalization factor prevent?",
    options: [
      "Overfitting during training",
      "Gradient vanishing problems",
      "Softmax saturation in high dimensions",
      "Memory overflow issues"
    ],
    correct: 2,
    explanation: "The √d_k factor prevents the dot products from becoming too large in high dimensions, which would cause the softmax to saturate and produce very small gradients.",
    difficulty: "hard" as const
  },
  {
    id: "transformer-3",
    question: "What is the purpose of positional encoding in Transformers?",
    options: [
      "To reduce computational complexity",
      "To provide sequence order information",
      "To prevent overfitting",
      "To enable parallel training"
    ],
    correct: 1,
    explanation: "Since attention is permutation-invariant, positional encodings are added to give the model information about the position of tokens in the sequence.",
    difficulty: "easy" as const
  }
];

const TransformersPage: React.FC = () => {
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
          🏗️ <span>Transformer</span> Architecture
        </PageTitle>
        <PageDescription>
          Master the revolutionary architecture that transformed AI. From attention mechanisms 
          to positional encoding, understand every component that makes Transformers the 
          foundation of modern language models.
        </PageDescription>
      </PageHeader>

      <ContentSection>
        <SectionTitle>Core Concepts</SectionTitle>
        <ConceptGrid>
          <Card>
            <h4><Brain />Self-Attention Mechanism</h4>
            <p>
              The heart of Transformers. Learn how queries, keys, and values work together 
              to create context-aware representations that capture long-range dependencies.
            </p>
          </Card>
          
          <Card variant="accent">
            <h4><Eye />Multi-Head Attention</h4>
            <p>
              Multiple attention mechanisms running in parallel, each learning different 
              types of relationships between tokens in the sequence.
            </p>
          </Card>
          
          <Card variant="purple">
            <h4><Code />Positional Encoding</h4>
            <p>
              Since attention is permutation-invariant, positional encodings inject 
              sequence order information using sinusoidal functions.
            </p>
          </Card>
        </ConceptGrid>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Mathematical Foundation</SectionTitle>
        
        <FormulaCard>
          <h4>Scaled Dot-Product Attention</h4>
          <p>The fundamental attention mechanism that computes attention weights:</p>
          <Math block>
            {"\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V"}
          </Math>
          <p>
            Where Q (queries), K (keys), and V (values) are learned projections of the input, 
            and d_k is the dimension of the key vectors.
          </p>
        </FormulaCard>

        <FormulaCard>
          <h4>Multi-Head Attention</h4>
          <p>Parallel attention mechanisms for different representation subspaces:</p>
          <Math block>
            {"\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O"}
          </Math>
          <Math block>
            {"\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)"}
          </Math>
        </FormulaCard>

        <FormulaCard>
          <h4>Positional Encoding</h4>
          <p>Sinusoidal functions to encode position information:</p>
          <Math block>
            {"PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)"}
          </Math>
          <Math block>
            {"PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)"}
          </Math>
        </FormulaCard>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Architecture Overview</SectionTitle>
        <ArchitectureDiagram>
          <h4>Transformer Block Structure</h4>
          <div style={{ textAlign: 'left', maxWidth: '600px', margin: '0 auto' }}>
            <KeyPoints>
              <li><CheckCircle size={16} />Input Embeddings + Positional Encoding</li>
              <li><CheckCircle size={16} />Multi-Head Self-Attention</li>
              <li><CheckCircle size={16} />Add & Norm (Residual Connection)</li>
              <li><CheckCircle size={16} />Feed-Forward Network (MLP)</li>
              <li><CheckCircle size={16} />Add & Norm (Residual Connection)</li>
              <li><CheckCircle size={16} />Stack N layers (typically 6-24)</li>
            </KeyPoints>
          </div>
        </ArchitectureDiagram>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Implementation Details</SectionTitle>
        
        <Card>
          <h4>PyTorch Attention Implementation</h4>
          <CodeBlock>{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(attn_output)`}</CodeBlock>
        </Card>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Key Interview Topics</SectionTitle>
        <KeyPoints>
          <li><CheckCircle size={16} /><strong>Attention vs. RNN/CNN:</strong> Explain why attention enables parallelization and better long-range dependencies</li>
          <li><CheckCircle size={16} /><strong>Computational Complexity:</strong> O(n²d) for attention vs O(nd²) for RNNs, where n is sequence length</li>
          <li><CheckCircle size={16} /><strong>Positional Encoding:</strong> Why sinusoidal functions work and alternatives like learned embeddings</li>
          <li><CheckCircle size={16} /><strong>Layer Normalization:</strong> Pre-norm vs post-norm and their effects on training stability</li>
          <li><CheckCircle size={16} /><strong>Residual Connections:</strong> How they enable training of deep networks and gradient flow</li>
          <li><CheckCircle size={16} /><strong>Attention Patterns:</strong> What different heads learn (syntax, semantics, long-range dependencies)</li>
        </KeyPoints>
      </ContentSection>

      <ContentSection>
        <SectionTitle>Test Your Knowledge</SectionTitle>
        {!showQuiz ? (
          <Card>
            <h4><Book />Ready for the Quiz?</h4>
            <p>Test your understanding of Transformer architecture with these challenging questions.</p>
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
            questions={transformerQuestions}
          />
        )}
      </ContentSection>
    </motion.div>
  );
};

export default TransformersPage;
