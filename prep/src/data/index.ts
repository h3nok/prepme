// Import modules and quizzes
import { fundamentalsModule } from './FundamentalsModule';
import { transformerModule } from './TransformerModule';
import { llmsModule } from './LLMsModule';
import { diffusionModule } from './DiffusionModule';
import { genAIInterviewModule } from './GenAIInterviewModule';
import { llmsQuiz } from './LLMsQuiz';
import { transformerQuiz } from './TransformerQuiz';
import { diffusionQuiz } from './DiffusionQuiz';
import { genAIInterviewQuiz } from './GenAIInterviewQuiz';

// Export all learning modules
export { fundamentalsModule } from './FundamentalsModule';
export { transformerModule } from './TransformerModule';
export { llmsModule } from './LLMsModule';
export { diffusionModule } from './DiffusionModule';
export { genAIInterviewModule } from './GenAIInterviewModule';

// Export all quizzes
export { llmsQuiz } from './LLMsQuiz';
export { transformerQuiz } from './TransformerQuiz';
export { diffusionQuiz } from './DiffusionQuiz';
export { genAIInterviewQuiz } from './GenAIInterviewQuiz';

// Create the complete learning modules array
export const allModules = [
  fundamentalsModule,
  transformerModule,
  llmsModule,
  diffusionModule,
  genAIInterviewModule
];

// Create the complete quizzes array
export const allQuizzes = [
  transformerQuiz,
  llmsQuiz,
  diffusionQuiz,
  genAIInterviewQuiz
];
