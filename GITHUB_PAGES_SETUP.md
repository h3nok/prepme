# 🌐 GitHub Pages Setup Guide

This guide will help you host your interview preparation website on GitHub Pages for free!

## 📋 Prerequisites
- GitHub account
- Git installed on your machine
- Your interview preparation repository

## 🚀 Step-by-Step Setup

### 1. Push Your Website to GitHub

First, make sure your website files are in your repository:

```bash
# Navigate to your interview prep directory
cd /Users/hghebrechristos/Repo/Interviews

# Add all files to git
git add .

# Commit the changes
git commit -m "Add interactive interview preparation website"

# Push to GitHub (replace with your repository URL)
git push origin main
```

### 2. Enable GitHub Pages

1. Go to your GitHub repository in a web browser
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **Deploy from a branch**
5. Choose **main** branch
6. Select **/ (root)** as the folder
7. Click **Save**

### 3. Configure for Website Subdirectory

Since your website is in the `website/` folder, you have two options:

#### Option A: Move website files to root (Recommended)
```bash
# Move website files to root for easier GitHub Pages hosting
cp website/* .
git add .
git commit -m "Move website to root for GitHub Pages"
git push origin main
```

#### Option B: Change GitHub Pages source folder
1. In GitHub repository settings
2. Under **Pages** → **Source**
3. Select **main** branch and **website** folder
4. Click **Save**

### 4. Access Your Website

After a few minutes, your website will be available at:
```
https://[your-username].github.io/[repository-name]
```

For example: `https://hghebrechristos.github.io/Interviews`

## 🎯 Custom Domain (Optional)

If you want to use a custom domain:

1. Create a file named `CNAME` in your repository root
2. Add your domain name (e.g., `my-interview-prep.com`)
3. Configure your domain's DNS settings to point to GitHub Pages

## 🔧 Configuration Files

### .gitignore (if needed)
```gitignore
# macOS
.DS_Store

# Editor files
.vscode/
*.swp
*.swo

# Logs
*.log
```

### 404.html (Optional custom 404 page)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Page Not Found</title>
    <meta http-equiv="refresh" content="0; url=/">
</head>
<body>
    <p>Redirecting to home page...</p>
</body>
</html>
```

## ⚡ Optimization Tips

### 1. Fast Loading
- Your current setup uses CDNs for libraries ✅
- Images are optimized for web ✅
- CSS is minified in production ✅

### 2. SEO Optimization
Add to your HTML `<head>` sections:

```html
<meta name="description" content="AWS Senior Applied Scientist Interview Preparation - Interactive study materials">
<meta name="keywords" content="AWS, AI, Machine Learning, Interview, Preparation">
<meta property="og:title" content="AWS Interview Prep">
<meta property="og:description" content="Interactive study materials for AWS AI roles">
<meta property="og:type" content="website">
```

### 3. Analytics (Optional)
Add Google Analytics to track your study progress:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_TRACKING_ID');
</script>
```

## 🔒 Security Considerations

### Environment Variables
Never commit sensitive information. Use GitHub Secrets for any API keys:

```javascript
// Safe: Use environment variables
const apiKey = process.env.API_KEY;

// Unsafe: Hard-coded secrets
const apiKey = "sk-abcd1234..."; // Don't do this!
```

### HTTPS
GitHub Pages automatically provides HTTPS for your site ✅

## 📱 Mobile Optimization

Your website is already mobile-responsive! Test on different devices:

- **iPhone/Android**: Touch navigation works
- **Tablet**: Larger slides, better readability
- **Desktop**: Full experience with keyboard shortcuts

## 🛠️ Troubleshooting

### Common Issues

**Website not loading?**
- Check repository is public
- Verify GitHub Pages is enabled
- Wait 5-10 minutes for deployment

**Slides not working?**
- Check browser console for JavaScript errors
- Ensure CDN links are accessible
- Try hard refresh (Ctrl+F5 or Cmd+Shift+R)

**Navigation broken?**
- Verify all file paths are relative
- Check file names match exactly (case-sensitive)

### Build Status
Check your Pages deployment status:
1. Go to repository **Actions** tab
2. Look for **pages-build-deployment** workflows
3. Green checkmark = success, red X = error

## 🎉 Success!

Once deployed, you'll have:

✅ **Professional presentation** of your study materials  
✅ **Mobile-friendly** interface for studying anywhere  
✅ **Interactive navigation** with keyboard shortcuts  
✅ **Progress tracking** with local storage  
✅ **Shareable link** for mentors/peers  

## 📚 Next Steps

1. **Share the link** with study partners
2. **Practice presentations** using the slides
3. **Track your progress** through the materials
4. **Update content** as you learn new concepts
5. **Add more slideshows** for additional topics

## 🔄 Updating Your Website

To add new content or fix issues:

```bash
# Make your changes locally
# Test in browser

# Commit and push changes
git add .
git commit -m "Update study materials"
git push origin main

# GitHub Pages will auto-deploy in ~5 minutes
```

---

**Pro Tip**: Bookmark your GitHub Pages URL and use it for daily study sessions. The interactive format makes complex topics much easier to understand and remember! 🚀
