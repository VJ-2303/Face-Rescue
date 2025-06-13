// Dashboard functionality
class DashboardManager {
    constructor() {
        this.stats = null;
        this.logs = [];
        this.initializeElements();
    }
    
    initializeElements() {
        // Dashboard will auto-load when page becomes active
        document.addEventListener('DOMContentLoaded', () => {
            if (document.getElementById('dashboard-page').classList.contains('active')) {
                this.loadDashboard();
            }
        });
    }
    
    async loadDashboard() {
        try {
            showLoading('Loading dashboard...');
            
            // Load stats and logs in parallel
            const [stats, logs] = await Promise.all([
                this.loadStats(),
                this.loadRecentLogs()
            ]);
            
            this.stats = stats;
            this.logs = logs;
            
            this.renderStats();
            this.renderRecentLogs();
            
            hideLoading();
            
        } catch (error) {
            hideLoading();
            handleError(error, 'Failed to load dashboard');
        }
    }
    
    async loadStats() {
        try {
            const response = await api.getSearchStats();
            return response.stats;
        } catch (error) {
            console.error('Failed to load stats:', error);
            // Return default stats if API fails
            return {
                total_searches: 0,
                successful_matches: 0,
                success_rate: 0,
                recent_searches_24h: 0
            };
        }
    }
    
    async loadRecentLogs() {
        try {
            const response = await api.getSearchLogs(10, 0); // Get last 10 logs
            return response.logs;
        } catch (error) {
            console.error('Failed to load logs:', error);
            return [];
        }
    }
    
    renderStats() {
        if (!this.stats) return;
        
        // Update stat values
        document.getElementById('total-students').textContent = this.stats.total_students || '0';
        document.getElementById('total-searches').textContent = this.stats.total_searches || '0';
        document.getElementById('successful-matches').textContent = this.stats.successful_matches || '0';
        document.getElementById('success-rate').textContent = `${this.stats.success_rate || 0}%`;
        
        // Add animation to numbers
        this.animateNumbers();
    }
    
    animateNumbers() {
        const statCards = document.querySelectorAll('.stat-content h3');
        
        statCards.forEach(card => {
            const finalValue = card.textContent;
            const isPercentage = finalValue.includes('%');
            const numericValue = parseInt(finalValue.replace(/[^0-9]/g, ''));
            
            if (numericValue > 0) {
                let currentValue = 0;
                const increment = Math.ceil(numericValue / 30); // Animate over ~30 frames
                
                const timer = setInterval(() => {
                    currentValue += increment;
                    if (currentValue >= numericValue) {
                        currentValue = numericValue;
                        clearInterval(timer);
                    }
                    
                    card.textContent = isPercentage ? `${currentValue}%` : currentValue.toString();
                }, 50);
            }
        });
    }
    
    renderRecentLogs() {
        const container = document.getElementById('recent-logs');
        
        if (!this.logs || this.logs.length === 0) {
            container.innerHTML = `
                <div style=\"text-align: center; padding: 2rem; color: #666;\">
                    <i class=\"fas fa-clock\" style=\"font-size: 2rem; margin-bottom: 1rem; opacity: 0.5;\"></i>
                    <p>No recent search activity</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.logs.map(log => this.generateLogItem(log)).join('');
    }
    
    generateLogItem(log) {
        const timestamp = formatDate(log.timestamp);
        const hasMatch = log.has_match;
        const statusClass = hasMatch ? 'log-success' : 'log-failed';
        const statusText = hasMatch ? 'Match Found' : 'No Match';
        const confidenceText = log.confidence ? ` (${formatConfidence(log.confidence)})` : '';
        
        return `
            <div class=\"log-item\">
                <div class=\"log-info\">
                    <div class=\"log-status ${statusClass}\"></div>
                    <div>
                        <p style=\"margin: 0; font-weight: 500; color: #333;\">
                            Face Search - ${statusText}${confidenceText}
                        </p>
                        <p style=\"margin: 0; font-size: 0.85rem; color: #666;\">
                            ${log.ip_address || 'Unknown IP'}
                        </p>
                    </div>
                </div>
                <div class=\"log-time\">${timestamp}</div>
            </div>
        `;
    }
    
    // Method to refresh dashboard data
    async refreshDashboard() {
        await this.loadDashboard();
        showToast('Dashboard refreshed', 'success');
    }
    
    // Method to be called when dashboard page becomes active
    onPageActivated() {
        if (!this.stats || !this.logs.length) {
            this.loadDashboard();
        }
    }
    
    // Method to update stats in real-time (call after search operations)
    updateStatsAfterSearch(hasMatch, confidence = null) {
        if (!this.stats) return;
        
        this.stats.total_searches++;
        this.stats.recent_searches_24h++;
        
        if (hasMatch) {
            this.stats.successful_matches++;
        }
        
        // Recalculate success rate
        this.stats.success_rate = ((this.stats.successful_matches / this.stats.total_searches) * 100).toFixed(2);
        
        // Update display
        this.renderStats();
        
        // Add new log entry
        const newLog = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            has_match: hasMatch,
            confidence: confidence,
            ip_address: 'Current Session'
        };
        
        this.logs.unshift(newLog);
        if (this.logs.length > 10) {
            this.logs = this.logs.slice(0, 10);
        }
        
        this.renderRecentLogs();
    }
}

// Global dashboard manager
const dashboardManager = new DashboardManager();

// Enhanced dashboard with additional features
class EnhancedDashboard extends DashboardManager {
    constructor() {
        super();
        this.setupRealTimeUpdates();
        this.setupAutoRefresh();
    }
    
    setupRealTimeUpdates() {
        // Listen for search events to update dashboard in real-time
        document.addEventListener('searchCompleted', (event) => {
            const { hasMatch, confidence } = event.detail;
            this.updateStatsAfterSearch(hasMatch, confidence);
        });
    }
    
    setupAutoRefresh() {
        // Auto-refresh dashboard every 5 minutes when active
        setInterval(() => {
            const dashboardPage = document.getElementById('dashboard-page');
            if (dashboardPage && dashboardPage.classList.contains('active')) {
                this.loadStats().then(stats => {
                    this.stats = stats;
                    this.renderStats();
                });
            }
        }, 5 * 60 * 1000); // 5 minutes
    }
    
    // Override loadStats to also get student count
    async loadStats() {
        try {
            const [searchStats, students] = await Promise.all([
                api.getSearchStats(),
                api.getStudents().catch(() => []) // Don't fail if students API fails
            ]);
            
            return {
                ...searchStats.stats,
                total_students: students.length
            };
        } catch (error) {
            console.error('Failed to load stats:', error);
            return {
                total_searches: 0,
                successful_matches: 0,
                success_rate: 0,
                recent_searches_24h: 0,
                total_students: 0
            };
        }
    }
}

// Replace the global dashboard manager with enhanced version
const enhancedDashboard = new EnhancedDashboard();

// Add additional CSS for better dashboard styling
const style = document.createElement('style');
style.textContent = `
    .stats-grid {
        margin-bottom: 2rem;
    }
    
    .stat-card {
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stat-card:hover::before {
        opacity: 1;
    }
    
    .log-item {
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
    }
    
    .log-item:hover {
        background: #f0f0f0;
        border-left-color: #667eea;
        transform: translateX(4px);
    }
    
    .recent-activity {
        position: relative;
    }
    
    .recent-activity::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 12px 12px 0 0;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .log-status {
        animation: pulse 2s infinite;
    }
`;
document.head.appendChild(style);
