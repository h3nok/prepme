import React, { useState } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, XCircle, RotateCcw, ArrowRight } from 'lucide-react';

interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correct: number;
  explanation: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

interface QuizProps {
  questions: QuizQuestion[];
  title?: string;
}

const QuizContainer = styled.div`
  background: ${props => props.theme.colors.surface};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.lg};
  padding: ${props => props.theme.spacing.xl};
  margin: ${props => props.theme.spacing.lg} 0;
`;

const QuizHeader = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.lg};
  padding-bottom: ${props => props.theme.spacing.md};
  border-bottom: 1px solid ${props => props.theme.colors.border};
`;

const QuizTitle = styled.h3`
  color: ${props => props.theme.colors.primary};
  margin: 0;
  font-weight: 600;
`;

const Progress = styled.div`
  font-size: 0.9rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const QuestionContainer = styled(motion.div)`
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const QuestionText = styled.h4`
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
  line-height: 1.5;
`;

const DifficultyBadge = styled.span<{ $difficulty: string }>`
  display: inline-block;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.radii.sm};
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  margin-bottom: ${props => props.theme.spacing.md};
  
  background: ${props => {
    switch (props.$difficulty) {
      case 'easy': return props.theme.colors.success + '20';
      case 'medium': return props.theme.colors.warning + '20';
      case 'hard': return props.theme.colors.error + '20';
      default: return props.theme.colors.border;
    }
  }};
  
  color: ${props => {
    switch (props.$difficulty) {
      case 'easy': return props.theme.colors.success;
      case 'medium': return props.theme.colors.warning;
      case 'hard': return props.theme.colors.error;
      default: return props.theme.colors.text;
    }
  }};
`;

const OptionsList = styled.div`
  display: grid;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const Option = styled(motion.button)<{ $selected?: boolean; $correct?: boolean; $wrong?: boolean }>`
  text-align: left;
  padding: ${props => props.theme.spacing.md};
  border: 2px solid ${props => {
    if (props.$correct) return props.theme.colors.success;
    if (props.$wrong) return props.theme.colors.error;
    if (props.$selected) return props.theme.colors.primary;
    return props.theme.colors.border;
  }};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => {
    if (props.$correct) return props.theme.colors.success + '10';
    if (props.$wrong) return props.theme.colors.error + '10';
    if (props.$selected) return props.theme.colors.primary + '10';
    return props.theme.colors.background;
  }};
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};

  &:hover:not(:disabled) {
    border-color: ${props => props.theme.colors.primary};
    background: ${props => props.theme.colors.primary + '05'};
  }

  &:disabled {
    cursor: not-allowed;
  }
`;

const OptionIcon = styled.div<{ $correct?: boolean; $wrong?: boolean }>`
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  
  svg {
    width: 16px;
    height: 16px;
    color: ${props => {
      if (props.$correct) return props.theme.colors.success;
      if (props.$wrong) return props.theme.colors.error;
      return 'transparent';
    }};
  }
`;

const Explanation = styled(motion.div)`
  background: ${props => props.theme.colors.background};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  padding: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.md};
  
  h5 {
    color: ${props => props.theme.colors.accent};
    margin: 0 0 ${props => props.theme.spacing.sm} 0;
  }

  p {
    color: ${props => props.theme.colors.textSecondary};
    margin: 0;
    line-height: 1.6;
  }
`;

const Controls = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  justify-content: space-between;
  align-items: center;
`;

const Button = styled.button<{ $variant?: 'primary' | 'secondary' }>`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border: 2px solid ${props => props.$variant === 'primary' ? props.theme.colors.primary : props.theme.colors.border};
  border-radius: ${props => props.theme.radii.md};
  background: ${props => props.$variant === 'primary' ? props.theme.colors.primary : 'transparent'};
  color: ${props => props.$variant === 'primary' ? 'white' : props.theme.colors.text};
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.$variant === 'primary' ? props.theme.colors.accent : props.theme.colors.surfaceLight};
    border-color: ${props => props.$variant === 'primary' ? props.theme.colors.accent : props.theme.colors.primary};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  svg {
    width: 16px;
    height: 16px;
  }
`;

const Score = styled.div`
  text-align: center;
  padding: ${props => props.theme.spacing.xl};
  
  h3 {
    color: ${props => props.theme.colors.primary};
    margin-bottom: ${props => props.theme.spacing.md};
  }

  p {
    color: ${props => props.theme.colors.textSecondary};
    font-size: 1.1rem;
  }
`;

const Quiz: React.FC<QuizProps> = ({ questions, title = "Quiz" }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [score, setScore] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  const handleOptionSelect = (optionIndex: number) => {
    if (showAnswer) return;
    setSelectedOption(optionIndex);
  };

  const handleSubmit = () => {
    if (selectedOption === null) return;
    
    setShowAnswer(true);
    
    if (selectedOption === questions[currentQuestion].correct) {
      setScore(score + 1);
    }
  };

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedOption(null);
      setShowAnswer(false);
    } else {
      setIsComplete(true);
    }
  };

  const handleRestart = () => {
    setCurrentQuestion(0);
    setSelectedOption(null);
    setShowAnswer(false);
    setScore(0);
    setIsComplete(false);
  };

  const question = questions[currentQuestion];

  if (isComplete) {
    return (
      <QuizContainer>
        <Score>
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <h3>Quiz Complete!</h3>
            <p>Your Score: {score} / {questions.length} ({Math.round((score / questions.length) * 100)}%)</p>
            <Button $variant="primary" onClick={handleRestart}>
              <RotateCcw />
              Retake Quiz
            </Button>
          </motion.div>
        </Score>
      </QuizContainer>
    );
  }

  return (
    <QuizContainer>
      <QuizHeader>
        <QuizTitle>{title}</QuizTitle>
        <Progress>
          Question {currentQuestion + 1} of {questions.length}
        </Progress>
      </QuizHeader>

      <AnimatePresence mode="wait">
        <QuestionContainer
          key={currentQuestion}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          <DifficultyBadge $difficulty={question.difficulty}>
            {question.difficulty}
          </DifficultyBadge>
          
          <QuestionText>{question.question}</QuestionText>

          <OptionsList>
            {question.options.map((option, index) => (
              <Option
                key={index}
                onClick={() => handleOptionSelect(index)}
                disabled={showAnswer}
                $selected={selectedOption === index}
                $correct={showAnswer && index === question.correct}
                $wrong={showAnswer && selectedOption === index && index !== question.correct}
                whileHover={{ scale: showAnswer ? 1 : 1.02 }}
                whileTap={{ scale: showAnswer ? 1 : 0.98 }}
              >
                <OptionIcon 
                  $correct={showAnswer && index === question.correct}
                  $wrong={showAnswer && selectedOption === index && index !== question.correct}
                >
                  {showAnswer && index === question.correct && <CheckCircle />}
                  {showAnswer && selectedOption === index && index !== question.correct && <XCircle />}
                </OptionIcon>
                {option}
              </Option>
            ))}
          </OptionsList>

          <AnimatePresence>
            {showAnswer && (
              <Explanation
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h5>Explanation</h5>
                <p>{question.explanation}</p>
              </Explanation>
            )}
          </AnimatePresence>

          <Controls>
            <div>
              Score: {score} / {currentQuestion + (showAnswer ? 1 : 0)}
            </div>
            <div>
              {!showAnswer ? (
                <Button 
                  $variant="primary" 
                  onClick={handleSubmit}
                  disabled={selectedOption === null}
                >
                  Submit Answer
                </Button>
              ) : (
                <Button $variant="primary" onClick={handleNext}>
                  {currentQuestion < questions.length - 1 ? 'Next Question' : 'Finish Quiz'}
                  <ArrowRight />
                </Button>
              )}
            </div>
          </Controls>
        </QuestionContainer>
      </AnimatePresence>
    </QuizContainer>
  );
};

export default Quiz;
