# PrepMe - Professional AI Interview Preparation Platform

## 🚀 Live React Application
**Visit: https://h3nok.github.io/Interviews**

A modern, enterprise-ready React platform for advanced AI scientist interview preparation.

## 🎯 Current Features

### � React Application (Main Platform)
- **Interactive Learning**: Comprehensive AI topics with quizzes
- **Enterprise Ready**: Professional UI for senior roles
- **Responsive Design**: Collapsible sidebar, mobile-friendly
- **Advanced Topics**: Transformers, LLMs, Diffusion, Multimodal AI

### 🎨 Professional Design
- **Responsive design** works on all devices
- **Dark theme** optimized for long study sessions
- **Color-coded sections** for easy topic identification
- **Smooth animations** and transitions

### ⌨️ Keyboard Navigation
- **Arrow keys**: Navigate slides
- **Space bar**: Next slide
- **ESC**: Overview mode
- **F**: Fullscreen
- **H**: Help menu
- **M**: Quick navigation menu

### 📱 Mobile Friendly
- Touch navigation
- Responsive layouts
- Optimized for tablets and phones
- Works offline after first load

## 🚀 Quick Start

### Local Development
```bash
# Navigate to website directory
cd website

# Open in browser (any local server works)
python -m http.server 8000
# or
npx serve .

# Visit http://localhost:8000
```

### Deploy to GitHub Pages
See `GITHUB_PAGES_SETUP.md` for detailed instructions.

## 📁 File Structure

```
website/
├── index.html                    # Main slideshow
├── diffusion-multimodal.html     # Advanced topics slideshow
├── navigation.html               # Study portal homepage
├── styles.css                    # All styling
├── script.js                     # Interactive features
├── GITHUB_PAGES_SETUP.md        # Deployment guide
└── README.md                     # This file
```

## 🎓 Study Workflow

1. **Start at Navigation Portal** (`navigation.html`)
2. **Main Overview** for broad concepts
3. **Specific Slideshows** for deep dives
4. **Practice explanations** out loud
5. **Return to written guides** for implementation details

## 🛠️ Customization

### Adding New Slideshows
1. Copy existing HTML structure
2. Update content sections
3. Add navigation links
4. Test responsive design

### Modifying Styles
- Edit `styles.css` for global changes
- Use CSS custom properties for consistent theming
- Test on mobile devices

### Adding Interactive Features
- Extend `script.js` for new functionality
- Use Reveal.js API for slide control
- Add progress tracking or analytics

## 📊 Built With

- **[Reveal.js](https://revealjs.com/)**: Presentation framework
- **[Highlight.js](https://highlightjs.org/)**: Code syntax highlighting
- **Vanilla CSS**: Custom styling and animations
- **Vanilla JavaScript**: Interactive features

## 🎯 Key Benefits

### For Interview Prep
- **Visual learning**: Complex concepts made clear
- **Practice presentations**: Simulate interview scenarios
- **Progress tracking**: Monitor your study advancement
- **Quick reference**: Essential formulas and concepts

### For Long-term Use
- **Portfolio piece**: Demonstrate your web skills
- **Knowledge base**: Reference for future projects
- **Teaching tool**: Share with others learning AI
- **Template**: Adapt for other technical topics

## 🔧 Troubleshooting

### Common Issues
- **Slides not loading**: Check internet connection for CDN resources
- **Navigation not working**: Ensure JavaScript is enabled
- **Mobile layout issues**: Test responsive breakpoints
- **Performance problems**: Optimize images and reduce animations

### Browser Support
- **Chrome/Edge**: Full support ✅
- **Firefox**: Full support ✅
- **Safari**: Full support ✅
- **Mobile browsers**: Optimized experience ✅

## 📈 Performance

- **Fast loading**: CDN resources and optimized assets
- **Offline ready**: Service worker caching (can be added)
- **SEO optimized**: Proper meta tags and structure
- **Accessibility**: Keyboard navigation and screen reader support

## 🎨 Design Philosophy

- **Clean and professional**: Suitable for interview contexts
- **Information dense**: Maximum learning in minimum time
- **Intuitive navigation**: Focus on content, not interface
- **Consistent theming**: AWS orange and tech blues throughout

## 🚀 Future Enhancements

Potential additions:
- **Audio narration**: Voice explanations for slides
- **Interactive code examples**: Live coding environments
- **Quiz mode**: Self-testing capabilities
- **Study analytics**: Detailed progress tracking
- **Collaboration features**: Study group functionality

---

**Ready to ace your interview!** 🎯

Start studying: [Launch Navigation Portal](navigation.html)

### 📚 Legacy HTML Files (Preserved)
- **legacy-index.html**: Original interview prep slideshow
- **diffusion-multimodal.html**: Advanced AI topics slideshow  
- **navigation.html**: Navigation portal

## 📂 Repository Structure
- `/prep/` - React application source code
- `gh-pages` branch - Deployed React application
- Legacy HTML files preserved for reference
