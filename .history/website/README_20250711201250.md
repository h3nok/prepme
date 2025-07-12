# PrepMe Website

This directory contains the PrepMe website - a comprehensive guide for Generative AI and Applied Science interview preparation.

## 📁 File Structure

```
website/
├── index.html              # Main homepage
├── core-concepts.html      # Core concepts overview
├── transformer-architecture.html  # Transformer architecture deep dive
├── large-language-models.html     # LLMs content (to be created)
├── diffusion-models.html          # Diffusion models content (to be created)
├── multimodal-ai.html             # Multimodal AI content (to be created)
├── research.html           # Research & implementation methodology
├── slides.html             # AWS production content
├── code.html               # Code examples
├── resources.html          # External resources
├── pdf.html                # PDF guide
├── deploy-timestamp.txt    # Deployment tracking
└── README.md              # This file
```

## 🚀 Deployment

The website is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

### Manual Deployment

1. **Using the deployment script:**

   ```bash
   ./deploy.sh
   ```

2. **Manual git commands:**

   ```bash
   git add .
   git commit -m "Update website content"
   git push origin main
   ```

### GitHub Pages Configuration

- **Source:** GitHub Actions (from `main` branch)
- **Publish directory:** `./website`
- **Workflow:** `.github/workflows/gh-pages.yml`

## 🎨 Design Features

- **Responsive Design:** Works on desktop, tablet, and mobile
- **Interactive Elements:** Progress tracking, checklists, search functionality
- **Modern UI:** Gradient backgrounds, smooth animations, card-based layout
- **Code Highlighting:** Syntax highlighting for code examples
- **Math Rendering:** LaTeX math support via MathJax

## 📚 Content Sections

### Core Concepts

- **Transformer Architecture:** Self-attention, multi-head attention, positional encoding
- **Large Language Models:** GPT, BERT, T5, training strategies
- **Diffusion Models:** DDPM, DDIM, noise scheduling, guidance
- **Multimodal AI:** Vision-language models, cross-modal attention

### Research & Implementation

- **Research Methodology:** Problem identification, experimental design
- **Training Optimization:** Learning rate scheduling, optimization algorithms
- **Evaluation Metrics:** Performance, robustness, fairness metrics

### AWS Production

- **AWS Services:** SageMaker, Lambda, ECS, etc.
- **Production ML:** Model deployment, monitoring, scaling
- **Leadership:** Team collaboration, project management

## 🔧 Development

### Adding New Content

1. Create a new HTML file in the `website/` directory
2. Use the existing CSS classes for consistent styling
3. Add navigation links in the header and footer
4. Update the main index.html navigation if needed

### Styling Guidelines

- Use CSS custom properties (variables) defined in `:root`
- Follow the card-based layout pattern
- Use semantic HTML elements
- Ensure responsive design with mobile-first approach

### Code Examples

- Use `<div class="code-block">` for code snippets
- Include language-specific syntax highlighting
- Use `highlight.js` for automatic highlighting

## 📊 Progress Tracking

The website includes local storage-based progress tracking:

- Users can check off completed topics
- Progress is saved in browser localStorage
- Progress bars show completion percentages

## 🌐 External Resources

- **MathJax:** For mathematical notation rendering
- **Highlight.js:** For code syntax highlighting
- **CDN Resources:** All external resources loaded via CDN

## 📝 Maintenance

- Update `deploy-timestamp.txt` when making changes
- Test responsive design on different screen sizes
- Validate HTML and CSS
- Check all links and navigation

## 🐛 Troubleshooting

### Common Issues

1. **GitHub Pages not updating:**
   - Check GitHub Actions tab for deployment status
   - Verify the workflow completed successfully
   - Clear browser cache and try again

2. **Styling issues:**
   - Check CSS custom properties are defined
   - Verify responsive breakpoints
   - Test on different browsers

3. **Code highlighting not working:**
   - Ensure `highlight.js` is loaded
   - Check language classes are correct
   - Verify code blocks have proper structure

## 📞 Support

For issues or questions:

1. Check the main repository README
2. Review GitHub Actions logs
3. Create an issue in the repository

---

**Last Updated:** December 19, 2024
**Version:** 2.0.0
