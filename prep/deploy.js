#!/usr/bin/env node

const { execSync } = require('child_process');
const packageJson = require('./package.json');

console.log('\n🚀 Starting deployment of PrepMe...\n');

try {
  // First build the project
  console.log('📦 Building project...');
  execSync('npm run build', { stdio: 'inherit' });
  
  // Then deploy
  console.log('\n🌐 Deploying to GitHub Pages...');
  execSync('npx gh-pages -d build', { stdio: 'inherit' });
  
  console.log('\n🎉 PrepMe deployed successfully!');
  console.log('\n' + '='.repeat(60));
  console.log(`🚀 YOUR SITE IS LIVE AT: ${packageJson.homepage}`);
  console.log('='.repeat(60));
  
  console.log('\n📋 Quick Actions:');
  console.log(`   • Open in browser: ${packageJson.homepage}`);
  console.log(`   • Share this URL with others`);
  console.log(`   • Bookmark for easy access`);
  
  console.log('\n⏱️  It may take a few minutes for GitHub Pages to update');
  console.log('📝 Make sure your repository is public and GitHub Pages is enabled\n');
  
} catch (error) {
  console.error('\n❌ Deployment failed:', error.message);
  console.log('\n💡 Troubleshooting tips:');
  console.log('   1. Make sure you have committed and pushed your changes');
  console.log('   2. Ensure your repository is public');
  console.log('   3. Check that GitHub Pages is enabled in repository settings');
  console.log('   4. Update the homepage URL in package.json with your GitHub username');
  console.log('   5. Run "npm install" to ensure all dependencies are installed\n');
  process.exit(1);
}
