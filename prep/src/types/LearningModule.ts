export interface LearningModule {
  id: string;
  title: string;
  description: string;
  color: string;
  icon: string;
  progress: number;
  estimatedHours: number;
  prerequisites: string[];
  concepts: Concept[];
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
}

export interface Concept {
  id: string;
  title: string;
  description: string;
  slides: Slide[];
  interactive?: InteractiveElement[];
  prerequisites?: string[];
}

export interface Slide {
  id: string;
  title: string;
  content: SlideContent;
  visualizations?: Visualization[];
  interactiveElements?: InteractiveElement[];
  mathNotations?: MathNotation[];
  progressiveDisclosure?: DisclosureLayer[];
  keyPoints?: string[];
  practiceQuestions?: string[];
  commonMistakes?: string[];
  interviewTips?: string[];
}

export interface SlideContent {
  tier1: string; // Core concept - always visible
  tier2?: string; // Mechanism - expandable
  tier3?: string; // Technical details - toggle button
}

export interface Visualization {
  id: string;
  type: 'animation' | 'interactive' | 'diagram' | 'heatmap' | '3d';
  component: string;
  data: any;
  controls?: VisualizationControl[];
}

export interface VisualizationControl {
  id: string;
  type: 'slider' | 'toggle' | 'dropdown' | 'button';
  label: string;
  range?: [number, number];
  options?: string[];
  defaultValue?: any;
}

export interface InteractiveElement {
  id: string;
  type: 'calculator' | 'simulator' | 'playground' | 'quiz';
  component: string;
  props: any;
}

export interface MathNotation {
  id: string;
  latex: string;
  explanation: string;
  interactive?: boolean;
}

export interface DisclosureLayer {
  level: 1 | 2 | 3;
  content: string;
  trigger: string;
}

export interface UserProgress {
  moduleId: string;
  conceptId: string;
  slideId: string;
  completedSlides: string[];
  bookmarks: string[];
  mathComfortLevel: number; // 1-5
  preferredAnalogies: string[];
  timeSpent: number;
}

export interface LearningPath {
  modules: LearningModule[];
  dependencies: Record<string, string[]>;
  estimatedTotalHours: number;
}

export interface QuizQuestion {
  id: string;
  type: 'multiple-choice' | 'true-false' | 'short-answer' | 'coding';
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  category: string;
  interviewFrequency: 'Very High' | 'High' | 'Medium' | 'Low';
}

export interface Quiz {
  id: string;
  title: string;
  description: string;
  moduleId: string;
  timeLimit: number; // in minutes
  passingScore: number; // percentage
  questions: QuizQuestion[];
}
