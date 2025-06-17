// MCP Comparison Chat JavaScript
class MCPComparisonChat {
    constructor() {
        this.baselineThreadId = this.generateThreadId();
        this.enhancedThreadId = this.generateThreadId();
        
        this.elements = {
            // Baseline elements
            baselineMessages: document.getElementById('baselineMessages'),
            baselineInput: document.getElementById('baselineInput'),
            baselineSendButton: document.getElementById('baselineSendButton'),
            baselineLoading: document.getElementById('baselineLoading'),
            baselineStatus: document.getElementById('baselineStatus'),
            
            // Enhanced elements
            enhancedMessages: document.getElementById('enhancedMessages'),
            enhancedInput: document.getElementById('enhancedInput'),
            enhancedSendButton: document.getElementById('enhancedSendButton'),
            enhancedLoading: document.getElementById('enhancedLoading'),
            enhancedStatus: document.getElementById('enhancedStatus'),
            
            // Example queries
            exampleQueries: document.querySelectorAll('.example-query')
        };
        
        this.initialize();
    }
    
    generateThreadId() {
        return 'thread_' + Math.random().toString(36).substr(2, 9);
    }
    
    async initialize() {
        await this.checkHealth();
        this.setupEventListeners();
        this.enableInputs();
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
        
        const avatar = role === 'user' ? '👤' : (type === 'baseline' ? '🔵' : '🔴');
        const roleLabel = role === 'user' ? 'You' : (type === 'baseline' ? 'Baseline' : 'Enhanced');
        
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