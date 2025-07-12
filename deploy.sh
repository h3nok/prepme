#!/bin/bash

# PrepMe Website Deployment Script
# This script helps deploy the website to GitHub Pages

echo "🚀 PrepMe Website Deployment Script"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "website/index.html" ]; then
    echo "❌ Error: Please run this script from the prepme root directory"
    exit 1
fi

# Update deployment timestamp
echo "📝 Updating deployment timestamp..."
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
cat > website/deploy-timestamp.txt << EOF
Last deployed: $TIMESTAMP
Content includes:
- Core Concepts (Transformer Architecture, LLMs, Diffusion Models, Multimodal AI)
- Research & Implementation methodology
- Updated navigation and progress tracking
- All markdown content converted to HTML
- Interview practice materials
- AWS production guides
EOF

# Check git status
echo "🔍 Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    echo "📦 Changes detected. Committing and pushing..."
    
    # Add all changes
    git add .
    
    # Commit with timestamp
    git commit -m "🚀 Deploy website updates - $TIMESTAMP
    
    - Added comprehensive core concepts pages
    - Integrated research methodology content
    - Updated navigation and progress tracking
    - Converted markdown content to HTML
    - Enhanced user experience with interactive elements"
    
    # Push to main branch (triggers GitHub Pages deployment)
    git push origin main
    
    echo "✅ Changes pushed to GitHub!"
    echo "🌐 GitHub Pages deployment will start automatically..."
    echo "📊 Check deployment status at: https://github.com/[username]/prepme/actions"
else
    echo "✅ No changes detected. Website is up to date!"
fi

echo ""
echo "🎉 Deployment script completed!"
echo "📖 Your website should be available at: https://[username].github.io/prepme/"
echo ""
echo "📋 Next steps:"
echo "   1. Wait for GitHub Actions to complete (usually 2-3 minutes)"
echo "   2. Check the Actions tab in your GitHub repository"
echo "   3. Visit your GitHub Pages URL to verify the deployment"
echo ""
echo "🔧 To make changes:"
echo "   1. Edit files in the website/ directory"
echo "   2. Run this script again: ./deploy.sh"
echo "   3. Or manually commit and push to trigger deployment" 