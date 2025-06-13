// Main application controller
class App {
    constructor() {
        this.currentPage = 'search';
        this.initialized = false;
        this.init();
    }
    
    init() {
        if (this.initialized) return;
        
        this.setupNavigation();
        this.setupModalHandlers();
        this.setupKeyboardShortcuts();
        this.checkAPIConnection();
        
        // Initialize the default page
        this.showPage('search');
        
        this.initialized = true;
        console.log('Face Rescue App initialized');
    }
    
    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                if (page) {
                    this.showPage(page);
                }
            });
        });
    }
    
    showPage(pageName) {
        // Hide all pages
        const pages = document.querySelectorAll('.page');
        pages.forEach(page => page.classList.remove('active'));
        
        // Show selected page
        const targetPage = document.getElementById(`${pageName}-page`);
        if (targetPage) {
            targetPage.classList.add('active');
            this.currentPage = pageName;
            
            // Update navigation
            this.updateNavigation(pageName);
            
            // Call page-specific initialization
            this.onPageActivated(pageName);
            
            // Update page title
            this.updatePageTitle(pageName);
        }
    }
    
    updateNavigation(activePage) {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.dataset.page === activePage) {
                link.classList.add('active');
            }
        });
    }
    
    updatePageTitle(pageName) {
        const titles = {
            search: 'Search Child - Face Rescue',
            register: 'Register Student - Face Rescue',
            students: 'Students - Face Rescue',
            dashboard: 'Dashboard - Face Rescue'
        };
        
        document.title = titles[pageName] || 'Face Rescue';
    }
    
    onPageActivated(pageName) {
        switch (pageName) {
            case 'students':
                if (typeof studentsManager !== 'undefined') {
                    studentsManager.onPageActivated();
                }
                break;
            case 'dashboard':
                if (typeof enhancedDashboard !== 'undefined') {
                    enhancedDashboard.onPageActivated();
                }
                break;
            case 'search':
                // Stop camera if running when leaving search page
                break;
        }
    }
    
    setupModalHandlers() {
        // Close modal when clicking the close button
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-close')) {
                const modal = e.target.closest('.modal');
                if (modal) {
                    modal.classList.remove('active');
                }
            }
        });
        
        // Close modal when pressing Escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const activeModal = document.querySelector('.modal.active');
                if (activeModal) {
                    activeModal.classList.remove('active');
                }
            }
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts if no input is focused
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            // Ctrl/Cmd + number keys for quick navigation
            if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
                switch (e.key) {
                    case '1':
                        e.preventDefault();
                        this.showPage('search');
                        break;
                    case '2':
                        e.preventDefault();
                        this.showPage('register');
                        break;
                    case '3':
                        e.preventDefault();
                        this.showPage('students');
                        break;
                    case '4':
                        e.preventDefault();
                        this.showPage('dashboard');
                        break;
                }
            }
            
            // Other shortcuts
            if (e.key === 'F5' && this.currentPage === 'dashboard') {
                e.preventDefault();
                if (typeof enhancedDashboard !== 'undefined') {
                    enhancedDashboard.refreshDashboard();
                }
            }
        });
    }
    
    async checkAPIConnection() {
        try {
            await api.healthCheck();
            this.showConnectionStatus(true);
        } catch (error) {
            console.error('API connection failed:', error);
            this.showConnectionStatus(false);
        }
    }
    
    showConnectionStatus(connected) {
        // Remove existing status indicators
        const existingIndicator = document.querySelector('.connection-status');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        // Create status indicator
        const indicator = document.createElement('div');
        indicator.className = 'connection-status';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            z-index: 1500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        `;
        
        if (connected) {
            indicator.style.background = '#4CAF50';
            indicator.style.color = 'white';
            indicator.innerHTML = '<i class=\"fas fa-check-circle\"></i> API Connected';
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                if (indicator.parentNode) {
                    indicator.style.opacity = '0';
                    setTimeout(() => indicator.remove(), 300);
                }
            }, 3000);
        } else {
            indicator.style.background = '#F44336';
            indicator.style.color = 'white';
            indicator.innerHTML = '<i class=\"fas fa-exclamation-circle\"></i> API Disconnected';
            
            // Show retry button
            const retryBtn = document.createElement('button');
            retryBtn.innerHTML = '<i class=\"fas fa-sync-alt\"></i>';
            retryBtn.style.cssText = `
                background: none;
                border: 1px solid white;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.7rem;
                margin-left: 0.5rem;
            `;
            
            retryBtn.addEventListener('click', () => {
                this.checkAPIConnection();
            });
            
            indicator.appendChild(retryBtn);
        }
        
        document.body.appendChild(indicator);
    }
    
    // Utility method to dispatch custom events
    dispatchEvent(eventName, detail = {}) {
        const event = new CustomEvent(eventName, { detail });
        document.dispatchEvent(event);
    }
    
    // Method to handle app-wide error notifications
    handleGlobalError(error, context = '') {
        console.error(`Global error ${context}:`, error);
        
        // Check if it's a network error
        if (error.message.includes('fetch') || error.message.includes('network')) {
            this.showConnectionStatus(false);
        }
        
        // Show user-friendly error message
        const userMessage = this.getUserFriendlyErrorMessage(error);
        showToast(userMessage, 'error');
    }
    
    getUserFriendlyErrorMessage(error) {
        if (error.message.includes('fetch') || error.message.includes('network')) {
            return 'Connection error. Please check your internet connection and try again.';
        }
        
        if (error.message.includes('413') || error.message.includes('too large')) {
            return 'File size is too large. Please use a smaller image.';
        }
        
        if (error.message.includes('400')) {
            return 'Invalid request. Please check your input and try again.';
        }
        
        if (error.message.includes('500')) {
            return 'Server error. Please try again later.';
        }
        
        return error.message || 'An unexpected error occurred. Please try again.';
    }
    
    // Method to show app loading state
    showAppLoading(show = true) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    
    // Set up global error handling
    window.addEventListener('error', (event) => {
        app.handleGlobalError(event.error, 'JavaScript error');
    });
    
    window.addEventListener('unhandledrejection', (event) => {
        app.handleGlobalError(event.reason, 'Unhandled promise rejection');
    });
    
    // Periodically check API connection
    setInterval(() => {
        app.checkAPIConnection();
    }, 30000); // Check every 30 seconds
});

// Add some utility CSS for better UX
const appStyles = document.createElement('style');
appStyles.textContent = `
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Focus styles for accessibility */
    .btn:focus,
    input:focus,
    select:focus,
    textarea:focus {
        outline: 2px solid #667eea;
        outline-offset: 2px;
    }
    
    /* Loading states */
    .btn.loading {
        opacity: 0.7;
        pointer-events: none;
        position: relative;
    }
    
    .btn.loading::after {
        content: '';
        position: absolute;
        width: 16px;
        height: 16px;
        border: 2px solid transparent;
        border-top: 2px solid currentColor;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
    }
    
    /* Connection status animations */
    .connection-status {
        animation: slideInFromRight 0.3s ease;
    }
    
    @keyframes slideInFromRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Improved modal backdrop */
    .modal {
        backdrop-filter: blur(5px);
    }
    
    /* Better mobile responsiveness */
    @media (max-width: 480px) {
        .main-content {
            padding: 1rem 0.5rem;
        }
        
        .nav-container {
            padding: 1rem 0.5rem;
        }
        
        .page-header h1 {
            font-size: 2rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* Print styles */
    @media print {
        .navbar,
        .loading-overlay,
        .toast-container,
        .connection-status {
            display: none !important;
        }
        
        .main-content {
            margin-top: 0;
        }
        
        .page {
            display: block !important;
        }
        
        .student-card {
            break-inside: avoid;
        }
    }
`;
document.head.appendChild(appStyles);

// Expose useful functions globally for debugging
window.debugApp = {
    showPage: (page) => app.showPage(page),
    checkAPI: () => app.checkAPIConnection(),
    clearStorage: () => {
        localStorage.clear();
        sessionStorage.clear();
        console.log('Storage cleared');
    },
    getStats: () => enhancedDashboard?.stats,
    version: '1.0.0'
};

console.log('Face Rescue App loaded successfully! 🚀');
console.log('Use window.debugApp for debugging utilities');
