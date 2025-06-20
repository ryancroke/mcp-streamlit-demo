// MCP Comparison Chat JavaScript
class MCPComparisonChat {
    constructor() {
        this.baselineThreadId = this.generateThreadId();
        this.enhancedThreadId = this.generateThreadId();
        this.config = null;
        
        this.elements = {
            // Page elements
            pageTitle: document.getElementById('pageTitle'),
            comparisonSelect: document.getElementById('comparisonSelect'),
            
            // Baseline elements
            baselineMessages: document.getElementById('baselineMessages'),
            baselineInput: document.getElementById('baselineInput'),
            baselineSendButton: document.getElementById('baselineSendButton'),
            baselineLoading: document.getElementById('baselineLoading'),
            baselineStatus: document.getElementById('baselineStatus'),
            baselineTitle: document.getElementById('baselineTitle'),
            baselineSubtitle: document.getElementById('baselineSubtitle'),
            baselineAvatar: document.getElementById('baselineAvatar'),
            baselineWelcomeTitle: document.getElementById('baselineWelcomeTitle'),
            baselineWelcomeText: document.getElementById('baselineWelcomeText'),
            
            // Enhanced elements
            enhancedMessages: document.getElementById('enhancedMessages'),
            enhancedInput: document.getElementById('enhancedInput'),
            enhancedSendButton: document.getElementById('enhancedSendButton'),
            enhancedLoading: document.getElementById('enhancedLoading'),
            enhancedStatus: document.getElementById('enhancedStatus'),
            enhancedTitle: document.getElementById('enhancedTitle'),
            enhancedSubtitle: document.getElementById('enhancedSubtitle'),
            enhancedAvatar: document.getElementById('enhancedAvatar'),
            enhancedWelcomeTitle: document.getElementById('enhancedWelcomeTitle'),
            enhancedWelcomeText: document.getElementById('enhancedWelcomeText'),
            
            // Example queries
            exampleQueries: document.querySelectorAll('.example-query')
        };
        
        this.initialize();
    }
    
    generateThreadId() {
        return 'thread_' + Math.random().toString(36).substr(2, 9);
    }
    
    async initialize() {
        await this.loadAvailableComparisons();
        await this.loadConfig();
        this.updateUI();
        await this.checkHealth();
        this.setupEventListeners();
        this.enableInputs();
    }
    
    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (!response.ok) {
                throw new Error(`Failed to load config: ${response.statusText}`);
            }
            this.config = await response.json();
            console.log('Loaded config:', this.config);
        } catch (error) {
            console.error('Failed to load configuration:', error);
            // Fallback to default values
            this.config = {
                comparison_name: "MCP Comparison",
                comparison_description: "Comparing baseline vs enhanced MCP servers",
                baseline: {
                    title: "Baseline MCP",
                    icon: "🔵",
                    color: "#4A90E2"
                },
                enhanced: {
                    title: "Enhanced MCP", 
                    icon: "🔴",
                    color: "#E25A4A"
                }
            };
        }
    }
    
    async loadAvailableComparisons() {
        try {
            const response = await fetch('/api/comparisons/list');
            if (!response.ok) {
                throw new Error(`Failed to load comparisons: ${response.statusText}`);
            }
            const data = await response.json();
            this.populateComparisonSelector(data.comparisons);
        } catch (error) {
            console.error('Failed to load available comparisons:', error);
            this.elements.comparisonSelect.innerHTML = '<option value="">Error loading comparisons</option>';
        }
    }
    
    populateComparisonSelector(comparisons) {
        this.elements.comparisonSelect.innerHTML = '';
        
        comparisons.forEach(comparison => {
            const option = document.createElement('option');
            option.value = comparison.id;
            option.textContent = comparison.name;
            this.elements.comparisonSelect.appendChild(option);
        });
        
        // Get current active comparison and select it
        this.getCurrentComparison().then(currentComparison => {
            if (currentComparison && currentComparison.comparison_id) {
                this.elements.comparisonSelect.value = currentComparison.comparison_id;
            }
        });
    }
    
    async getCurrentComparison() {
        try {
            const response = await fetch('/api/comparisons/current');
            if (!response.ok) {
                throw new Error(`Failed to get current comparison: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to get current comparison:', error);
            return null;
        }
    }
    
    async switchComparison(comparisonId) {
        if (!comparisonId) return;
        
        try {
            // Show loading state
            this.setLoadingState(true);
            
            const response = await fetch('/api/comparisons/switch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ comparison_id: comparisonId })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to switch comparison: ${response.statusText}`);
            }
            
            // Reload config and update UI
            await this.loadConfig();
            this.updateUI();
            
            // Clear chat histories for new comparison
            this.clearAllMessages();
            
            // Generate new thread IDs
            this.baselineThreadId = this.generateThreadId();
            this.enhancedThreadId = this.generateThreadId();
            
            // Check health of new comparison
            await this.checkHealth();
            
        } catch (error) {
            console.error('Failed to switch comparison:', error);
            alert('Failed to switch comparison. Please try again.');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    setLoadingState(loading) {
        this.elements.baselineInput.disabled = loading;
        this.elements.enhancedInput.disabled = loading;
        this.elements.baselineSendButton.disabled = loading;
        this.elements.enhancedSendButton.disabled = loading;
        this.elements.comparisonSelect.disabled = loading;
    }
    
    clearAllMessages() {
        // Clear baseline messages except welcome
        const baselineMessages = this.elements.baselineMessages;
        const baselineWelcome = document.getElementById('baselineWelcome');
        baselineMessages.innerHTML = '';
        baselineMessages.appendChild(baselineWelcome);
        
        // Clear enhanced messages except welcome
        const enhancedMessages = this.elements.enhancedMessages;
        const enhancedWelcome = document.getElementById('enhancedWelcome');
        enhancedMessages.innerHTML = '';
        enhancedMessages.appendChild(enhancedWelcome);
    }
    
    updateUI() {
        if (!this.config) return;
        
        // Update page title
        this.elements.pageTitle.textContent = `🎵 ${this.config.comparison_name}`;
        document.title = `🎵 ${this.config.comparison_name}`;
        
        // Update baseline UI
        const baseline = this.config.baseline;
        this.elements.baselineTitle.textContent = `${baseline.icon} ${baseline.title}`;
        this.elements.baselineSubtitle.textContent = baseline.subtitle || "Baseline Version";
        this.elements.baselineAvatar.textContent = baseline.icon;
        this.elements.baselineWelcomeTitle.textContent = `${baseline.title} Ready!`;
        this.elements.baselineWelcomeText.textContent = `This is the ${baseline.title.toLowerCase()}. ${this.config.comparison_description}`;
        this.elements.baselineInput.placeholder = `Ask the ${baseline.title.toLowerCase()}...`;
        
        // Update enhanced UI  
        const enhanced = this.config.enhanced;
        this.elements.enhancedTitle.textContent = `${enhanced.icon} ${enhanced.title}`;
        this.elements.enhancedSubtitle.textContent = enhanced.subtitle || "Enhanced Version";
        this.elements.enhancedAvatar.textContent = enhanced.icon;
        this.elements.enhancedWelcomeTitle.textContent = `${enhanced.title} Ready!`;
        this.elements.enhancedWelcomeText.textContent = `This is the ${enhanced.title.toLowerCase()}. ${this.config.comparison_description}`;
        this.elements.enhancedInput.placeholder = `Ask the ${enhanced.title.toLowerCase()}...`;
        
        // Update colors via CSS custom properties
        document.documentElement.style.setProperty('--baseline-color', baseline.color);
        document.documentElement.style.setProperty('--enhanced-color', enhanced.color);
    }
    
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            // Update baseline status
            const baselineHealthy = health.baseline_healthy;
            this.updateStatus('baseline', baselineHealthy);
            
            // Update enhanced status
            const enhancedHealthy = health.enhanced_healthy;
            this.updateStatus('enhanced', enhancedHealthy);
            
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus('baseline', false);
            this.updateStatus('enhanced', false);
        }
    }
    
    updateStatus(type, healthy) {
        const statusElement = this.elements[`${type}Status`];
        const statusText = statusElement.querySelector('.status-text');
        const statusDot = statusElement.querySelector('.status-dot');
        
        if (healthy) {
            statusText.textContent = `${type.charAt(0).toUpperCase() + type.slice(1)}: Ready`;
            statusDot.style.backgroundColor = type === 'baseline' ? '#4A90E2' : '#E25A4A';
        } else {
            statusText.textContent = `${type.charAt(0).toUpperCase() + type.slice(1)}: Error`;
            statusDot.style.backgroundColor = '#ff4444';
        }
    }
    
    setupEventListeners() {
        // Comparison selector
        this.elements.comparisonSelect.addEventListener('change', (e) => {
            this.switchComparison(e.target.value);
        });
        
        // Baseline chat
        this.elements.baselineInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage('baseline');
            }
        });
        
        this.elements.baselineSendButton.addEventListener('click', () => {
            this.sendMessage('baseline');
        });
        
        // Enhanced chat
        this.elements.enhancedInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage('enhanced');
            }
        });
        
        this.elements.enhancedSendButton.addEventListener('click', () => {
            this.sendMessage('enhanced');
        });
        
        // Example queries
        this.elements.exampleQueries.forEach(query => {
            query.addEventListener('click', () => {
                const queryText = query.dataset.query;
                this.elements.baselineInput.value = queryText;
                this.elements.enhancedInput.value = queryText;
            });
        });
    }
    
    enableInputs() {
        this.elements.baselineInput.disabled = false;
        this.elements.baselineSendButton.disabled = false;
        this.elements.enhancedInput.disabled = false;
        this.elements.enhancedSendButton.disabled = false;
    }
    
    async sendMessage(type) {
        const input = this.elements[`${type}Input`];
        const message = input.value.trim();
        
        if (!message) return;
        
        // Clear input and show loading
        input.value = '';
        this.setLoading(type, true);
        
        // Add user message to chat
        this.addMessage(type, message, 'user');
        
        try {
            const threadId = type === 'baseline' ? this.baselineThreadId : this.enhancedThreadId;
            
            const response = await fetch(`/api/${type}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    thread_id: threadId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            // Add assistant response
            this.addMessage(type, result.response, 'assistant', result.sql_query);
            
        } catch (error) {
            console.error('Send message failed:', error);
            this.addMessage(type, `Error: ${error.message}`, 'assistant');
        } finally {
            this.setLoading(type, false);
        }
    }
    
    addMessage(type, content, role, sqlQuery = null) {
        const messagesContainer = this.elements[`${type}Messages`];
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role === 'user' ? 'user-message' : ''}`;
        
        let avatar, roleLabel;
        if (role === 'user') {
            avatar = '👤';
            roleLabel = 'You';
        } else {
            // Use config for assistant avatars and labels
            if (this.config && this.config[type]) {
                avatar = this.config[type].icon;
                roleLabel = this.config[type].title;
            } else {
                // Fallback
                avatar = type === 'baseline' ? '🔵' : '🔴';
                roleLabel = type === 'baseline' ? 'Baseline' : 'Enhanced';
            }
        }
        
        let messageHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <strong>${roleLabel}</strong>
                <p>${this.escapeHtml(content)}</p>
        `;
        
        // Add SQL query if present
        if (sqlQuery && role === 'assistant') {
            messageHTML += `<div class="sql-query"><strong>SQL:</strong> ${this.escapeHtml(sqlQuery)}</div>`;
        }
        
        messageHTML += `</div>`;
        messageDiv.innerHTML = messageHTML;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    setLoading(type, loading) {
        const loadingElement = this.elements[`${type}Loading`];
        const sendButton = this.elements[`${type}SendButton`];
        const input = this.elements[`${type}Input`];
        
        if (loading) {
            loadingElement.style.display = 'flex';
            sendButton.disabled = true;
            input.disabled = true;
        } else {
            loadingElement.style.display = 'none';
            sendButton.disabled = false;
            input.disabled = false;
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the chat when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new MCPComparisonChat();
});