# PrepMe - Professional AI Interview Preparation Platform## 🚀 OverviewPrepMe is a modern, enterprise-ready React platform designed for advanced AI scientist interview preparation. The platform provides comprehensive coverage of cutting-edge AI topics, interactive quizzes, and professional-grade content suitable for senior roles across all major technology companies.## ✨ Features### 🎯 Advanced AI Topics- **Transformers & Attention Mechanisms** - In-depth coverage of modern NLP architectures- **Large Language Models (LLMs)** - Comprehensive LLM theory and applications- **Diffusion Models** - Complete guide to generative AI for images and beyond- **Multimodal AI** - Cross-modal learning and unified architectures- **Production & MLOps** - Cloud-agnostic deployment and scaling strategies### 💡 Interactive Learning- **Mathematical Rendering** - KaTeX-powered formula display- **Interactive Quizzes** - Immediate feedback and explanations- **Code Examples** - Syntax-highlighted implementation examples- **Collapsible Sidebar** - Responsive navigation for all device sizes### 🏢 Enterprise Ready- **Scalable Architecture** - Built to support hundreds of concurrent users- **Professional UI/UX** - Modern, accessible design with Framer Motion animations- **GitHub Pages Deployment** - Zero-config deployment pipeline- **Cross-Platform** - Responsive design for desktop, tablet, and mobile## 🛠️ Technology Stack- **Frontend**: React 19 with TypeScript- **Styling**: Styled Components with responsive design- **Animations**: Framer Motion for smooth transitions- **Math**: KaTeX for professional mathematical notation- **Icons**: Lucide React for consistent iconography- **Routing**: React Router for seamless navigation- **Build**: Create React App with optimized production builds## 🚀 Quick Start### Prerequisites- Node.js 16+ and npm- Git for version control### Installation1. **Clone the repository**   ```bash
   git clone https://github.com/yourusername/prepme.git
   cd prepme
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```
   Open [http://localhost:3000](http://localhost:3000) to view the application.

## 📦 Deployment

### GitHub Pages (Recommended)

1. **Update package.json homepage**
   ```json
   "homepage": "https://yourusername.github.io/prepme"
   ```

2. **Deploy using the smart deploy script**
   ```bash
   npm run deploy
   ```
   This will build the project, deploy to GitHub Pages, and show you the URL.

3. **Alternative deployment methods**
   ```bash
   # Simple deployment (no URL display)
   npm run deploy:simple
   
   # Using shell script (shows URL)
   ./deploy.sh
   ```

### Manual Deployment

1. **Build for production**
   ```bash
   npm run build
   ```

2. **Deploy the `build` folder** to your hosting provider of choice.

## 🏗️ Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Card.tsx        # Content cards
│   ├── Footer.tsx      # Site footer
│   ├── Header.tsx      # Navigation header
│   ├── Math.tsx        # Mathematical formula renderer
│   ├── Quiz.tsx        # Interactive quiz component
│   └── Sidebar.tsx     # Collapsible navigation
├── context/            # React context providers
│   └── SidebarContext.tsx
├── pages/              # Main application pages
│   ├── HomePage.tsx    # Landing page
│   ├── TransformersPage.tsx
│   ├── LLMsPage.tsx
│   ├── DiffusionPage.tsx
│   ├── MultimodalPage.tsx
│   ├── AWSPage.tsx     # Production & MLOps
│   └── QuizPage.tsx    # Interactive assessments
└── App.tsx             # Main application component
```

## 🎯 Target Audience

- **Senior AI Scientists** preparing for industry interviews
- **ML Engineers** transitioning to advanced AI roles
- **Research Scientists** moving to applied AI positions
- **Technical Leaders** staying current with AI developments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔧 Available Scripts

- `npm start` - Start development server
- `npm test` - Run test suite
- `npm run build` - Build for production
- `npm run deploy` - Deploy to GitHub Pages
- `npm run eject` - Eject from Create React App (not recommended)

---

**Built with ❤️ for the AI community**
