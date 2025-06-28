import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Clock, 
  Users, 
  Target, 
  Brain, 
  CheckCircle,
  AlertCircle,
  ChevronRight,
  Mic,
  MicOff
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

const SimulatorContainer = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.xl};
  border: 1px solid ${props => props.theme.colors.border};
  overflow: hidden;
  box-shadow: ${props => props.theme.shadows.lg};
`;

const SimulatorHeader = styled.div`
  padding: ${props => props.theme.spacing.xl};
  background: linear-gradient(135deg, 
    ${props => props.theme.colors.primary}20, 
    ${props => props.theme.colors.accent}20
  );
  border-bottom: 1px solid ${props => props.theme.colors.border};
`;

const InterviewTypeSelector = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const InterviewTypeCard = styled(motion.button)<{ $isActive: boolean }>`
  background: ${props => props.$isActive 
    ? `linear-gradient(135deg, ${props.theme.colors.primary}20, ${props.theme.colors.accent}20)`
    : props.theme.colors.background};
  border: 2px solid ${props => props.$isActive 
    ? props.theme.colors.primary 
    : props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.lg};
  text-align: left;
  cursor: pointer;
  transition: all 0.3s ease;

  &:hover {
    border-color: ${props => props.theme.colors.primary};
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.md};
  }
`;

const TypeTitle = styled.h3`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.sm};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const TypeDescription = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.9rem;
  margin-bottom: ${props => props.theme.spacing.sm};
`;

const TypeDetails = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  font-size: 0.8rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const SimulatorContent = styled.div`
  padding: ${props => props.theme.spacing.xl};
`;

const QuestionDisplay = styled(motion.div)`
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin-bottom: ${props => props.theme.spacing.xl};
  border: 1px solid ${props => props.theme.colors.border};
`;

const QuestionText = styled.h2`
  color: ${props => props.theme.colors.text};
  font-size: 1.5rem;
  line-height: 1.5;
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const QuestionMeta = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.lg};
  flex-wrap: wrap;
`;

const MetaTag = styled.span`
  background: ${props => props.theme.colors.primary}20;
  color: ${props => props.theme.colors.primary};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.sm};
  font-size: 0.8rem;
  font-weight: 600;
`;

const ControlPanel = styled.div`
  display: flex;
  justify-content: center;
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
  flex-wrap: wrap;
`;

const ControlButton = styled(motion.button)<{ $variant?: 'primary' | 'secondary' }>`
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

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

const TimerDisplay = styled.div`
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const Timer = styled.div`
  font-size: 3rem;
  font-weight: 800;
  color: ${props => props.theme.colors.primary};
  font-family: ${props => props.theme.fonts.mono};
`;

const TimerLabel = styled.p`
  color: ${props => props.theme.colors.textSecondary};
  margin-top: ${props => props.theme.spacing.sm};
`;

const AnswerArea = styled.div`
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};
`;

const AnswerTextarea = styled.textarea`
  width: 100%;
  min-height: 200px;
  background: transparent;
  border: none;
  color: ${props => props.theme.colors.text};
  font-size: 1rem;
  line-height: 1.6;
  resize: vertical;
  outline: none;

  &::placeholder {
    color: ${props => props.theme.colors.textSecondary};
  }
`;

const FeedbackSection = styled(motion.div)`
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.radii.lg};
  border: 1px solid ${props => props.theme.colors.border};
  padding: ${props => props.theme.spacing.xl};
`;

const FeedbackTitle = styled.h3`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.lg};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const FeedbackItem = styled.div`
  padding: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.theme.colors.background};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const FeedbackCategory = styled.h4`
  color: ${props => props.theme.colors.primary};
  margin-bottom: ${props => props.theme.spacing.sm};
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const FeedbackText = styled.p`
  color: ${props => props.theme.colors.text};
  line-height: 1.6;
`;

const interviewTypes = [
  {
    id: 'technical',
    title: 'Technical Deep Dive',
    description: 'Architecture, algorithms, and implementation questions',
    duration: '45-60 min',
    difficulty: 'Advanced',
    icon: Brain
  },
  {
    id: 'system-design',
    title: 'ML System Design',
    description: 'End-to-end ML system architecture and scaling',
    duration: '60-90 min',
    difficulty: 'Expert',
    icon: Target
  },
  {
    id: 'behavioral',
    title: 'Behavioral + Leadership',
    description: 'STAR method, leadership principles, and scenarios',
    duration: '30-45 min',
    difficulty: 'Intermediate',
    icon: Users
  }
];

const sampleQuestions = {
  technical: [
    {
      question: "Explain the attention mechanism in transformers. How would you implement multi-head attention from scratch?",
      category: "Architecture",
      difficulty: "Advanced",
      timeLimit: 15
    },
    {
      question: "You have a large language model that's hallucinating. Walk me through your debugging and mitigation strategy.",
      category: "Debugging",
      difficulty: "Expert",
      timeLimit: 20
    },
    {
      question: "Compare and contrast different approaches to fine-tuning large language models. When would you use each?",
      category: "Training",
      difficulty: "Advanced",
      timeLimit: 12
    }
  ],
  'system-design': [
    {
      question: "Design a recommendation system for a streaming platform that serves 100M+ users. Include both real-time and batch components.",
      category: "System Design",
      difficulty: "Expert",
      timeLimit: 45
    },
    {
      question: "You need to build a real-time content moderation system using ML. Design the architecture from data ingestion to serving.",
      category: "System Design",
      difficulty: "Expert",
      timeLimit: 40
    }
  ],
  behavioral: [
    {
      question: "Tell me about a time when you had to make a difficult technical decision with limited information. How did you approach it?",
      category: "Decision Making",
      difficulty: "Intermediate",
      timeLimit: 8
    },
    {
      question: "Describe a situation where you had to convince stakeholders to adopt a new ML approach. What was your strategy?",
      category: "Influence",
      difficulty: "Intermediate",
      timeLimit: 10
    }
  ]
};

export const InterviewSimulatorPage: React.FC = () => {
  const [selectedType, setSelectedType] = useState<string>('technical');
  const [isActive, setIsActive] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState<any>(null);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [answer, setAnswer] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isActive && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining(time => time - 1);
      }, 1000);
    } else if (timeRemaining === 0 && isActive) {
      setIsActive(false);
      setShowFeedback(true);
    }
    return () => clearInterval(interval);
  }, [isActive, timeRemaining]);

  const startInterview = () => {
    const questions = sampleQuestions[selectedType as keyof typeof sampleQuestions];
    const randomQuestion = questions[Math.floor(Math.random() * questions.length)];
    setCurrentQuestion(randomQuestion);
    setTimeRemaining(randomQuestion.timeLimit * 60);
    setIsActive(true);
    setShowFeedback(false);
    setAnswer('');
  };

  const pauseInterview = () => {
    setIsActive(!isActive);
  };

  const resetInterview = () => {
    setIsActive(false);
    setCurrentQuestion(null);
    setTimeRemaining(0);
    setAnswer('');
    setShowFeedback(false);
  };

  const submitAnswer = () => {
    setIsActive(false);
    setShowFeedback(true);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const generateFeedback = () => {
    const feedbackItems = [
      {
        category: "Structure & Clarity",
        feedback: "Your answer was well-structured and easy to follow. Consider adding more specific examples to strengthen your points."
      },
      {
        category: "Technical Depth",
        feedback: "Good technical understanding demonstrated. Try to dive deeper into implementation details and potential edge cases."
      },
      {
        category: "Communication",
        feedback: "Clear communication style. Consider explaining your thought process more explicitly as you work through the problem."
      }
    ];
    return feedbackItems;
  };

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Interview Simulator</PageTitle>
        <PageDescription>
          Practice technical interviews, system design, and behavioral questions with real-time feedback and timing
        </PageDescription>
      </PageHeader>

      <SimulatorContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <SimulatorHeader>
          <h2 style={{ marginBottom: '1.5rem', color: 'var(--text)' }}>Choose Interview Type</h2>
          <InterviewTypeSelector>
            {interviewTypes.map((type) => (
              <InterviewTypeCard
                key={type.id}
                $isActive={selectedType === type.id}
                onClick={() => setSelectedType(type.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <TypeTitle>
                  <type.icon size={20} />
                  {type.title}
                </TypeTitle>
                <TypeDescription>{type.description}</TypeDescription>
                <TypeDetails>
                  <span>⏱️ {type.duration}</span>
                  <span>🎯 {type.difficulty}</span>
                </TypeDetails>
              </InterviewTypeCard>
            ))}
          </InterviewTypeSelector>
        </SimulatorHeader>

        <SimulatorContent>
          {!currentQuestion ? (
            <div style={{ textAlign: 'center', padding: '3rem' }}>
              <h3 style={{ marginBottom: '1rem' }}>Ready to start your interview simulation?</h3>
              <p style={{ marginBottom: '2rem', color: 'var(--text-secondary)' }}>
                Select an interview type above and click start to begin
              </p>
              <ControlButton $variant="primary" onClick={startInterview}>
                <Play size={20} />
                Start Interview
              </ControlButton>
            </div>
          ) : (
            <>
              <QuestionDisplay
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <QuestionMeta>
                  <MetaTag>{currentQuestion.category}</MetaTag>
                  <MetaTag>{currentQuestion.difficulty}</MetaTag>
                  <MetaTag>{currentQuestion.timeLimit} minutes</MetaTag>
                </QuestionMeta>
                <QuestionText>{currentQuestion.question}</QuestionText>
              </QuestionDisplay>

              <TimerDisplay>
                <Timer>{formatTime(timeRemaining)}</Timer>
                <TimerLabel>Time Remaining</TimerLabel>
              </TimerDisplay>

              <ControlPanel>
                <ControlButton onClick={pauseInterview}>
                  {isActive ? <Pause size={20} /> : <Play size={20} />}
                  {isActive ? 'Pause' : 'Resume'}
                </ControlButton>
                <ControlButton onClick={() => setIsRecording(!isRecording)}>
                  {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
                  {isRecording ? 'Stop Recording' : 'Start Recording'}
                </ControlButton>
                <ControlButton onClick={resetInterview}>
                  <RotateCcw size={20} />
                  Reset
                </ControlButton>
                <ControlButton $variant="primary" onClick={submitAnswer}>
                  <CheckCircle size={20} />
                  Submit Answer
                </ControlButton>
              </ControlPanel>

              <AnswerArea>
                <AnswerTextarea
                  placeholder="Type your answer here or use voice recording..."
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                />
              </AnswerArea>

              <AnimatePresence>
                {showFeedback && (
                  <FeedbackSection
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.5 }}
                  >
                    <FeedbackTitle>
                      <Target size={20} />
                      Interview Feedback
                    </FeedbackTitle>
                    {generateFeedback().map((item, index) => (
                      <FeedbackItem key={index}>
                        <FeedbackCategory>{item.category}</FeedbackCategory>
                        <FeedbackText>{item.feedback}</FeedbackText>
                      </FeedbackItem>
                    ))}
                  </FeedbackSection>
                )}
              </AnimatePresence>
            </>
          )}
        </SimulatorContent>
      </SimulatorContainer>
    </PageContainer>
  );
};

export default InterviewSimulatorPage;
