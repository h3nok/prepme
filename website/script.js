// Initialize Reveal.js with custom configuration
Reveal.initialize({
    // Display configuration
    hash: true,
    center: true,
    touch: true,
    loop: false,
    rtl: false,
    
    // Navigation configuration
    controls: true,
    controlsTutorial: true,
    controlsLayout: 'bottom-right',
    controlsBackArrows: 'faded',
    progress: true,
    slideNumber: 'c/t',
    showSlideNumber: 'all',
    
    // Presentation size
    width: 1200,
    height: 700,
    margin: 0.04,
    minScale: 0.2,
    maxScale: 2.0,
    
    // Transition configuration
    transition: 'slide',
    transitionSpeed: 'default',
    backgroundTransition: 'fade',
    
    // Parallax background
    parallaxBackgroundImage: '',
    parallaxBackgroundSize: '',
    parallaxBackgroundRepeat: '',
    parallaxBackgroundPosition: '',
    parallaxBackgroundHorizontal: null,
    parallaxBackgroundVertical: null,
    
    // Keyboard navigation
    keyboard: {
        13: 'next', // Enter key
        32: 'next', // Space bar
        37: 'prev', // Left arrow
        39: 'next', // Right arrow
        38: 'prev', // Up arrow
        40: 'next', // Down arrow
        72: function() { // H key for help
            showHelp();
        },
        77: function() { // M key for menu
            showMenu();
        }
    },
    
    // Touch navigation
    touchNavigation: true,
    
    // Plugins
    plugins: [
        RevealMarkdown,
        RevealHighlight,
        RevealNotes
    ],
    
    // Plugin configurations
    markdown: {
        smartypants: true
    },
    
    highlight: {
        highlightOnLoad: true,
        tabReplace: '  '
    },
    
    // Speaker notes
    showNotes: false,
    
    // Auto-slide (disabled by default)
    autoSlide: 0,
    autoSlideStoppable: true,
    autoSlideMethod: 'next',
    
    // Mouse wheel navigation
    mouseWheel: false,
    
    // Hide address bar on mobile
    hideAddressBar: true,
    
    // Preview links
    previewLinks: false,
    
    // Focus body when page changes on hash change
    focusBodyOnPageVisibilityChange: true,
    
    // Experimental features
    embedded: false,
    postMessage: true,
    postMessageEvents: false,
    
    // View distance for lazy loading
    viewDistance: 3,
    
    // Mobile view distance
    mobileViewDistance: 2,
    
    // Display mode used to show slides
    display: 'block'
});

// Custom functions
function showHelp() {
    const helpModal = document.createElement('div');
    helpModal.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                    background: rgba(0,0,0,0.8); z-index: 1000; display: flex; 
                    align-items: center; justify-content: center;">
            <div style="background: #1a1a1a; padding: 2rem; border-radius: 12px; 
                        max-width: 500px; color: white; border: 2px solid #ff6b35;">
                <h2 style="color: #ff6b35; margin-bottom: 1rem;">Navigation Help</h2>
                <ul style="text-align: left; line-height: 1.6;">
                    <li><strong>→ / Space</strong> - Next slide</li>
                    <li><strong>← / Backspace</strong> - Previous slide</li>
                    <li><strong>ESC</strong> - Slide overview</li>
                    <li><strong>F</strong> - Fullscreen mode</li>
                    <li><strong>S</strong> - Speaker notes</li>
                    <li><strong>B / .</strong> - Pause (black screen)</li>
                    <li><strong>H</strong> - Show this help</li>
                </ul>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: #ff6b35; color: white; border: none; 
                               padding: 0.5rem 1rem; border-radius: 6px; 
                               margin-top: 1rem; cursor: pointer;">Close</button>
            </div>
        </div>
    `;
    document.body.appendChild(helpModal);
}

function showMenu() {
    const menuModal = document.createElement('div');
    menuModal.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                    background: rgba(0,0,0,0.8); z-index: 1000; display: flex; 
                    align-items: center; justify-content: center;">
            <div style="background: #1a1a1a; padding: 2rem; border-radius: 12px; 
                        max-width: 600px; color: white; border: 2px solid #7c3aed;">
                <h2 style="color: #7c3aed; margin-bottom: 1rem;">Study Sections</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <button onclick="Reveal.slide(1); this.parentElement.parentElement.parentElement.remove()" 
                            style="background: #1e3a8a; color: white; border: none; 
                                   padding: 1rem; border-radius: 6px; cursor: pointer;">
                        Transformers
                    </button>
                    <button onclick="Reveal.slide(7); this.parentElement.parentElement.parentElement.remove()" 
                            style="background: #059669; color: white; border: none; 
                                   padding: 1rem; border-radius: 6px; cursor: pointer;">
                        Large Language Models
                    </button>
                    <button onclick="Reveal.slide(11); this.parentElement.parentElement.parentElement.remove()" 
                            style="background: #ff6b35; color: white; border: none; 
                                   padding: 1rem; border-radius: 6px; cursor: pointer;">
                        AWS Services
                    </button>
                    <button onclick="Reveal.slide(14); this.parentElement.parentElement.parentElement.remove()" 
                            style="background: #7c3aed; color: white; border: none; 
                                   padding: 1rem; border-radius: 6px; cursor: pointer;">
                        Interview Questions
                    </button>
                </div>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: #dc2626; color: white; border: none; 
                               padding: 0.5rem 1rem; border-radius: 6px; 
                               margin-top: 1rem; cursor: pointer;">Close</button>
            </div>
        </div>
    `;
    document.body.appendChild(menuModal);
}

// Add progress tracking
Reveal.on('slidechanged', event => {
    // Store progress in localStorage
    localStorage.setItem('interview-prep-progress', JSON.stringify({
        slideIndex: event.indexh,
        timestamp: new Date().toISOString()
    }));
    
    // Update page title with current section
    const sectionTitles = {
        0: 'Welcome',
        1: 'Study Plan',
        2: 'Transformers',
        7: 'Large Language Models',
        11: 'AWS Services',
        14: 'Interview Questions',
        17: 'Quick Reference',
        19: 'Next Steps'
    };
    
    const currentSection = sectionTitles[event.indexh] || 'AWS Interview Prep';
    document.title = `${currentSection} - AWS Senior Applied Scientist`;
});

// Load previous progress
document.addEventListener('DOMContentLoaded', () => {
    const progress = localStorage.getItem('interview-prep-progress');
    if (progress) {
        const { slideIndex } = JSON.parse(progress);
        // Optionally resume from last position
        // Reveal.slide(slideIndex);
    }
});

// Add keyboard shortcuts info to slides
Reveal.on('ready', event => {
    // Add a floating help button
    const helpButton = document.createElement('button');
    helpButton.innerHTML = '?';
    helpButton.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: #ff6b35;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    `;
    
    helpButton.onmouseover = () => {
        helpButton.style.transform = 'scale(1.1)';
        helpButton.style.boxShadow = '0 4px 20px rgba(255,107,53,0.4)';
    };
    
    helpButton.onmouseout = () => {
        helpButton.style.transform = 'scale(1)';
        helpButton.style.boxShadow = '0 2px 10px rgba(0,0,0,0.3)';
    };
    
    helpButton.onclick = showHelp;
    document.body.appendChild(helpButton);
    
    // Add menu button
    const menuButton = document.createElement('button');
    menuButton.innerHTML = '☰';
    menuButton.style.cssText = `
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 1000;
        background: #7c3aed;
        color: white;
        border: none;
        border-radius: 6px;
        width: 40px;
        height: 40px;
        font-size: 18px;
        cursor: pointer;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    `;
    
    menuButton.onmouseover = () => {
        menuButton.style.transform = 'scale(1.1)';
        menuButton.style.boxShadow = '0 4px 20px rgba(124,58,237,0.4)';
    };
    
    menuButton.onmouseout = () => {
        menuButton.style.transform = 'scale(1)';
        menuButton.style.boxShadow = '0 2px 10px rgba(0,0,0,0.3)';
    };
    
    menuButton.onclick = showMenu;
    document.body.appendChild(menuButton);
});

// Add timer functionality
let studyStartTime = null;
let studyTimer = null;

function startStudyTimer() {
    studyStartTime = new Date();
    studyTimer = setInterval(() => {
        const elapsed = new Date() - studyStartTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        
        // Update timer display if element exists
        const timerDisplay = document.querySelector('.study-timer');
        if (timerDisplay) {
            timerDisplay.textContent = `Study time: ${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }, 1000);
}

// Auto-start timer when presentation begins
Reveal.on('ready', () => {
    // Add timer display
    const timerDisplay = document.createElement('div');
    timerDisplay.className = 'study-timer';
    timerDisplay.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 14px;
        z-index: 1000;
    `;
    document.body.appendChild(timerDisplay);
    
    // Start the timer
    startStudyTimer();
});

// Cleanup timer when page unloads
window.addEventListener('beforeunload', () => {
    if (studyTimer) {
        clearInterval(studyTimer);
    }
});
