#!/bin/bash

echo ""
echo "🚀 Starting deployment of PrepMe..."
echo ""

# Build the project
echo "📦 Building project..."
npm run build

if [ $? -eq 0 ]; then
    echo ""
    echo "🌐 Deploying to GitHub Pages..."
    npx gh-pages -d build
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 PrepMe deployed successfully!"
        
        # Read homepage from package.json
        HOMEPAGE=$(node -p "require('./package.json').homepage")
        echo "🔗 Visit your site at: $HOMEPAGE"
        
        if [[ $HOMEPAGE == *"username.github.io"* ]]; then
            echo ""
            echo "⚠️  IMPORTANT: Update the homepage URL in package.json with your actual GitHub username!"
            echo "   Change 'username' to your GitHub username in the homepage field."
        fi
        
        echo ""
        echo "⏱️  It may take a few minutes for GitHub Pages to update"
        echo "📝 Make sure your repository is public and GitHub Pages is enabled"
        echo ""
    else
        echo ""
        echo "❌ Deployment failed!"
        echo ""
        echo "💡 Troubleshooting tips:"
        echo "   1. Make sure you have committed and pushed your changes"
        echo "   2. Ensure your repository is public"
        echo "   3. Check that GitHub Pages is enabled in repository settings"
        echo "   4. Update the homepage URL in package.json with your GitHub username"
        echo "   5. Run 'npm install' to ensure all dependencies are installed"
        echo ""
        exit 1
    fi
else
    echo ""
    echo "❌ Build failed! Fix build errors before deploying."
    echo ""
    exit 1
fi
