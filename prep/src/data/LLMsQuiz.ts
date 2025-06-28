import { Quiz } from '../types/LearningModule';

export const llmsQuiz: Quiz = {
  id: 'llms-quiz',
  title: 'Advanced Large Language Models Quiz',
  description: 'Comprehensive assessment covering LLM architecture, training, production deployment, AWS services, multimodal AI, and leadership principles',
  moduleId: 'llms',
  timeLimit: 60,
  passingScore: 85,
  questions: [
    {
      id: 'llm-1',
      type: 'multiple-choice',
      question: 'What is the primary architectural pattern used in modern Large Language Models like GPT?',
      options: [
        'Encoder-only (like BERT)',
        'Decoder-only (like GPT)',
        'Encoder-decoder (like T5)',
        'Recurrent Neural Network'
      ],
      correctAnswer: 1,
      explanation: 'Modern LLMs like GPT use decoder-only architecture with causal attention, meaning each token can only attend to previous tokens. This enables autoregressive text generation.',
      difficulty: 'Intermediate',
      category: 'llm-architecture',
      interviewFrequency: 'Very High'
    },
    {
      id: 'llm-2',
      type: 'multiple-choice',
      question: 'According to scaling laws research, which factor has the strongest correlation with model performance?',
      options: [
        'Number of parameters (N)',
        'Dataset size (D)',
        'Compute budget (C)',
        'All three equally'
      ],
      correctAnswer: 2,
      explanation: 'The Chinchilla paper showed that compute budget C is the strongest predictor, and models are often under-trained for their parameter count. Optimal scaling requires balanced increases in N, D, and C.',
      difficulty: 'Advanced',
      category: 'scaling-laws',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-3',
      type: 'multiple-choice',
      question: 'In the context of LLM training, what does RLHF stand for and why is it important?',
      options: [
        'Reinforcement Learning from Human Feedback - aligns model outputs with human preferences',
        'Rapid Learning with High Frequency - speeds up training process',
        'Recursive Language Head Formation - improves attention mechanisms',
        'Random Loss Hierarchical Function - regularizes training'
      ],
      correctAnswer: 0,
      explanation: 'RLHF uses human feedback to train a reward model, which then guides the LLM via reinforcement learning to produce outputs that better align with human values and preferences.',
      difficulty: 'Advanced',
      category: 'training-optimization',
      interviewFrequency: 'Very High'
    },
    {
      id: 'llm-4',
      type: 'multiple-choice',
      question: 'Which AWS service is specifically designed for hosting and serving foundation models at scale?',
      options: [
        'Amazon EC2 with custom containers',
        'Amazon SageMaker JumpStart',
        'Amazon Bedrock',
        'AWS Lambda with large memory'
      ],
      correctAnswer: 2,
      explanation: 'Amazon Bedrock is a fully managed service that makes foundation models available via APIs, handling scaling, security, and compliance automatically.',
      difficulty: 'Intermediate',
      category: 'aws-services',
      interviewFrequency: 'Very High'
    },
    {
      id: 'llm-5',
      type: 'multiple-choice',
      question: 'What is the key advantage of using SageMaker Model Parallel Library for large model training?',
      options: [
        'Automatically distributes model layers across multiple GPUs/nodes',
        'Reduces memory usage by quantizing weights',
        'Speeds up inference by caching activations',
        'Optimizes data loading from S3'
      ],
      correctAnswer: 0,
      explanation: 'SageMaker Model Parallel Library enables training models that don\'t fit on a single GPU by automatically partitioning the model across devices, handling gradient synchronization.',
      difficulty: 'Advanced',
      category: 'aws-production',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-6',
      type: 'multiple-choice',
      question: 'In Vision-Language models like CLIP, what is the primary training objective?',
      options: [
        'Masked image modeling',
        'Contrastive learning between image-text pairs',
        'Next token prediction on concatenated inputs',
        'Cross-modal reconstruction loss'
      ],
      correctAnswer: 1,
      explanation: 'CLIP uses contrastive learning to learn a shared embedding space where semantically similar image-text pairs are close together, enabling zero-shot classification.',
      difficulty: 'Advanced',
      category: 'multimodal-ai',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-7',
      type: 'multiple-choice',
      question: 'What is the primary benefit of using KV-caching in transformer inference?',
      options: [
        'Reduces model size by compressing weights',
        'Speeds up autoregressive generation by caching key-value pairs',
        'Improves accuracy by storing attention patterns',
        'Enables batch processing of variable-length sequences'
      ],
      correctAnswer: 1,
      explanation: 'KV-caching stores previously computed key-value pairs in attention layers, avoiding redundant computation during autoregressive generation, significantly speeding up inference.',
      difficulty: 'Advanced',
      category: 'production-optimization',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-8',
      type: 'multiple-choice',
      question: 'Which technique is most effective for reducing hallucinations in LLM outputs?',
      options: [
        'Increasing model temperature',
        'Retrieval-Augmented Generation (RAG)',
        'Using larger context windows',
        'Fine-tuning on more data'
      ],
      correctAnswer: 1,
      explanation: 'RAG grounds the model\'s responses in retrieved factual information, significantly reducing hallucinations by providing relevant context from a knowledge base.',
      difficulty: 'Intermediate',
      category: 'capabilities-limitations',
      interviewFrequency: 'Very High'
    },
    {
      id: 'llm-9',
      type: 'multiple-choice',
      question: 'In the context of model deployment, what is the primary purpose of using Amazon SageMaker Multi-Model Endpoints?',
      options: [
        'Deploy multiple versions of the same model for A/B testing',
        'Host thousands of models behind a single endpoint to reduce costs',
        'Enable real-time model retraining',
        'Automatically scale based on traffic patterns'
      ],
      correctAnswer: 1,
      explanation: 'Multi-Model Endpoints allow hosting thousands of models behind a single endpoint, loading models on-demand and reducing infrastructure costs for serving many models.',
      difficulty: 'Advanced',
      category: 'aws-cost-optimization',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-10',
      type: 'multiple-choice',
      question: 'What is the key architectural difference between GPT-4V and traditional language models?',
      options: [
        'Larger parameter count',
        'Multimodal architecture that processes both text and images',
        'Different attention mechanism',
        'Extended context length'
      ],
      correctAnswer: 1,
      explanation: 'GPT-4V is a multimodal model that can process and understand both text and images, enabling applications like visual question answering and image description.',
      difficulty: 'Intermediate',
      category: 'multimodal-capabilities',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-11',
      type: 'multiple-choice',
      question: 'When implementing a production LLM system, which Amazon Leadership Principle is most relevant for handling model failures gracefully?',
      options: [
        'Customer Obsession',
        'Ownership',
        'Learn and Be Curious',
        'Bias for Action'
      ],
      correctAnswer: 1,
      explanation: 'Ownership emphasizes end-to-end responsibility, including building robust systems that handle failures gracefully, monitor performance, and ensure reliability.',
      difficulty: 'Intermediate',
      category: 'leadership-principles',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-12',
      type: 'multiple-choice',
      question: 'What is the primary advantage of using Parameter-Efficient Fine-tuning (PEFT) methods like LoRA?',
      options: [
        'Improves model accuracy significantly',
        'Reduces fine-tuning time and storage requirements',
        'Enables training larger models',
        'Automatically optimizes hyperparameters'
      ],
      correctAnswer: 1,
      explanation: 'LoRA and other PEFT methods fine-tune only a small subset of parameters, drastically reducing storage, compute, and time requirements while maintaining performance.',
      difficulty: 'Advanced',
      category: 'training-efficiency',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-13',
      type: 'multiple-choice',
      question: 'In production LLM systems, what is the purpose of implementing circuit breakers?',
      options: [
        'Prevent electrical overload in GPUs',
        'Stop cascading failures when downstream services are unavailable',
        'Optimize memory usage during inference',
        'Enable graceful model version rollbacks'
      ],
      correctAnswer: 1,
      explanation: 'Circuit breakers prevent cascading failures by temporarily stopping requests to failing services, allowing systems to recover and maintaining overall system stability.',
      difficulty: 'Advanced',
      category: 'production-reliability',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-14',
      type: 'multiple-choice',
      question: 'Which evaluation metric is most appropriate for assessing the factual accuracy of LLM-generated content?',
      options: [
        'BLEU score',
        'Perplexity',
        'Human evaluation with fact-checking',
        'Token-level accuracy'
      ],
      correctAnswer: 2,
      explanation: 'Factual accuracy requires human evaluation or automated fact-checking against reliable sources, as traditional metrics like BLEU don\'t assess truthfulness.',
      difficulty: 'Intermediate',
      category: 'evaluation-metrics',
      interviewFrequency: 'High'
    },
    {
      id: 'llm-15',
      type: 'multiple-choice',
      question: 'What is the primary challenge when scaling LLM inference to handle millions of requests per day?',
      options: [
        'Model accuracy degradation',
        'Memory bandwidth and compute cost optimization',
        'Training data quality',
        'Hyperparameter tuning'
      ],
      correctAnswer: 1,
      explanation: 'At scale, the primary challenges are managing memory bandwidth bottlenecks, optimizing compute costs through batching and caching, and maintaining low latency.',
      difficulty: 'Advanced',
      category: 'production-scaling',
      interviewFrequency: 'Very High'
    }
  ]
};
