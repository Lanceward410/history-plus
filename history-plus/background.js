// History Plus Background Script
// Handles history collection, session detection, and data storage

class HistoryPlusBackground {
  constructor() {
    console.log('History Plus Background Script Initializing...');
    
    // Initialize database first, then set up everything else
    this.initializeDatabase()
      .then(() => {
        console.log('Database initialized successfully');
        this.setupEventListeners();
        this.setupKeyboardCommands();
        this.startActiveTabMonitoring();
        this.startBackendManager(); // Add backend management
        
        // Update tasks immediately on startup
        this.updateCurrentTasks();
      })
      .catch(error => {
        console.error('Failed to initialize database:', error);
        // Continue with basic functionality even if database fails
        this.setupEventListeners();
        this.setupKeyboardCommands();
        this.startActiveTabMonitoring();
        this.startBackendManager(); // Still try to manage backend
      });
    
    // Background state
    this.isMonitoring = false;
    this.currentTasks = [];
    this.activeTabId = null;
    this.sessionStartTime = null;
    
    // Backend management state
    this.backendProcess = null;
    this.backendStatus = 'stopped'; // 'starting', 'running', 'stopped', 'error'
    this.backendPort = 5000;
    this.maxRetries = 3;
    this.retryCount = 0;
  }

  startActiveTabMonitoring() {
    // Monitor active tab changes and update current tasks
    console.log('Starting active tab monitoring...');
    this.isMonitoring = true;
    
    // Start monitoring active tab
    setInterval(() => {
      if (this.isMonitoring) {
        this.checkActiveTab();
      }
    }, 5000); // Check every 5 seconds
  }

  async checkActiveTab() {
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs.length > 0) {
        const activeTab = tabs[0];
        if (this.activeTabId !== activeTab.id) {
          this.activeTabId = activeTab.id;
          this.recordTabActivity(activeTab);
        }
      }
    } catch (error) {
      console.error('Error checking active tab:', error);
    }
  }

  // Backend Management System
  async startBackendManager() {
    console.log('Starting backend management system...');
    
    // Check if backend is already running
    const isRunning = await this.checkBackendHealth();
    if (isRunning) {
      console.log('Backend already running');
      this.backendStatus = 'running';
      return;
    }
    
    // Try to start backend automatically
    await this.startBackend();
  }

  async checkBackendHealth() {
    try {
      const response = await fetch(`http://localhost:${this.backendPort}/api/health`, {
        method: 'GET',
        timeout: 5000
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  async startBackend() {
    if (this.backendStatus === 'starting' || this.backendStatus === 'running') {
      return;
    }

    console.log('Attempting to start backend...');
    this.backendStatus = 'starting';
    
    try {
      // Method 1: Try Native Messaging (if available)
      if (await this.tryNativeMessaging()) {
        return;
      }
      
      // Method 2: Try subprocess via extension (Chrome limitations)
      if (await this.trySubprocess()) {
        return;
      }
      
      // Method 3: Show user instructions
      this.showBackendInstructions();
      
    } catch (error) {
      console.error('Failed to start backend:', error);
      this.backendStatus = 'error';
      this.showBackendError(error);
    }
  }

  async tryNativeMessaging() {
    try {
      // Check if native messaging is available
      if (!chrome.runtime.connectNative) {
        return false;
      }

      console.log('Trying native messaging approach...');
      
      // Connect to native host (would need to be implemented)
      const port = chrome.runtime.connectNative('com.historyplus.backend');
      
      port.onMessage.addListener((message) => {
        console.log('Native message received:', message);
        if (message.status === 'started') {
          this.backendStatus = 'running';
        }
      });
      
      port.onDisconnect.addListener(() => {
        console.log('Native messaging disconnected');
        this.backendStatus = 'stopped';
      });
      
      // Send start command
      port.postMessage({ action: 'start_backend' });
      
      // Wait for confirmation
      return new Promise((resolve) => {
        setTimeout(async () => {
          const isRunning = await this.checkBackendHealth();
          resolve(isRunning);
        }, 3000);
      });
      
    } catch (error) {
      console.log('Native messaging not available:', error);
      return false;
    }
  }

  async trySubprocess() {
    try {
      console.log('Trying subprocess approach...');
      
      // Chrome extensions can't directly spawn processes
      // But we can try to detect if Python is available and show instructions
      
      // For now, this is a placeholder for future Chrome API features
      return false;
      
    } catch (error) {
      console.log('Subprocess approach failed:', error);
      return false;
    }
  }

  showBackendInstructions() {
    console.log('Showing backend startup instructions to user');
    
    // Create a persistent notification
    chrome.notifications.create('backend-setup', {
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: 'History Plus Setup Required',
      message: 'Click to see backend setup instructions',
      buttons: [
        { title: 'Setup Instructions' },
        { title: 'Start Manually' }
      ],
      requireInteraction: true
    });
    
    // Handle notification clicks
    chrome.notifications.onButtonClicked.addListener((notificationId, buttonIndex) => {
      if (notificationId === 'backend-setup') {
        if (buttonIndex === 0) {
          this.openSetupGuide();
        } else {
          this.openManualSetup();
        }
        chrome.notifications.clear(notificationId);
      }
    });
  }

  async openSetupGuide() {
    // Create a setup instructions page
    const instructionsHtml = await this.generateSetupInstructions();
    
    // Create blob URL for instructions
    const blob = new Blob([instructionsHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    
    chrome.tabs.create({ url });
  }

  async generateSetupInstructions() {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>History Plus Backend Setup</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; }
            .header { text-align: center; margin-bottom: 40px; }
            .step { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .code { background: #1e1e1e; color: #ffffff; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; margin: 10px 0; }
            .button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px; text-decoration: none; display: inline-block; }
            .button:hover { background: #45a049; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 History Plus Backend Setup</h1>
            <p>Let's get your AI-powered history analysis running!</p>
        </div>

        <div class="step">
            <h3>📋 Step 1: Check Python Installation</h3>
            <p>Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:</p>
            <div class="code">python --version</div>
            <p>You should see Python 3.8 or higher. If not, <a href="https://python.org/downloads" target="_blank">download Python here</a>.</p>
        </div>

        <div class="step">
            <h3>📁 Step 2: Navigate to Extension Directory</h3>
            <p>Find your History Plus extension folder and navigate to the backend directory:</p>
            <div class="code">cd path/to/history-plus/backend</div>
        </div>

        <div class="step">
            <h3>📦 Step 3: Install Dependencies</h3>
            <p>Install the required Python packages:</p>
            <div class="code">pip install -r requirements.txt</div>
        </div>

        <div class="step">
            <h3>🚀 Step 4: Start the Backend</h3>
            <p>Start the Flask server:</p>
            <div class="code">python app.py</div>
            <p>You should see: "Running on http://localhost:5000"</p>
        </div>

        <div class="step">
            <h3>✅ Step 5: Verify Setup</h3>
            <button class="button" onclick="testBackend()">Test Backend Connection</button>
            <div id="status"></div>
        </div>

        <div class="step">
            <h3>🔧 Troubleshooting</h3>
            <ul>
                <li><strong>Permission Denied:</strong> Try running as administrator</li>
                <li><strong>Port 5000 in use:</strong> Close other applications using port 5000</li>
                <li><strong>Module not found:</strong> Ensure you're in the backend directory</li>
                <li><strong>Python not found:</strong> Add Python to your system PATH</li>
            </ul>
        </div>

        <script>
            async function testBackend() {
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '<div class="status">Testing connection...</div>';
                
                try {
                    const response = await fetch('http://localhost:5000/api/health');
                    if (response.ok) {
                        const data = await response.json();
                        statusDiv.innerHTML = '<div class="status success">✅ Backend is running successfully!</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="status error">❌ Backend responded but not healthy</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="status error">❌ Cannot connect to backend. Make sure it\'s running on port 5000.</div>';
                }
            }
            
            // Auto-test on page load
            setTimeout(testBackend, 1000);
        </script>
    </body>
    </html>
    `;
  }

  async openManualSetup() {
    // Open a new tab with the backend directory instructions
    chrome.tabs.create({
      url: chrome.runtime.getURL('dashboard/index.html')
    });
    
    // Show inline instructions in the dashboard
    setTimeout(() => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'showManualSetup'
        });
      });
    }, 1000);
  }

  showBackendError(error) {
    chrome.notifications.create('backend-error', {
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: 'History Plus Backend Error',
      message: `Failed to start backend: ${error.message}`,
      buttons: [{ title: 'Try Again' }, { title: 'Manual Setup' }]
    });
  }

  // Enhanced backend monitoring
  async monitorBackend() {
    if (this.backendStatus !== 'running') {
      return;
    }

    const isHealthy = await this.checkBackendHealth();
    
    if (!isHealthy) {
      console.log('Backend health check failed, attempting restart...');
      this.backendStatus = 'stopped';
      
      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        setTimeout(() => this.startBackend(), 5000); // Retry after 5 seconds
      } else {
        console.log('Max retries reached, manual intervention required');
        this.showBackendError(new Error('Backend stopped responding'));
      }
    } else {
      this.retryCount = 0; // Reset retry count on successful health check
    }
  }

  // Start periodic backend monitoring
  startBackendMonitoring() {
    setInterval(() => {
      this.monitorBackend();
    }, 30000); // Check every 30 seconds
  }

  async initializeDatabase() {
    // Initialize IndexedDB for local storage
    const db = await this.openDatabase();
    return db;
  }

  async openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('HistoryPlusDB', 1);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Create object stores
        if (!db.objectStoreNames.contains('historyEntries')) {
          const historyStore = db.createObjectStore('historyEntries', { keyPath: 'id', autoIncrement: true });
          historyStore.createIndex('url', 'url', { unique: false });
          historyStore.createIndex('timestamp', 'timestamp', { unique: false });
          historyStore.createIndex('domain', 'domain', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('sessions')) {
          const sessionStore = db.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
          sessionStore.createIndex('startTime', 'startTime', { unique: false });
          sessionStore.createIndex('endTime', 'endTime', { unique: false });
        }
        
        if (!db.objectStoreNames.contains('embeddings')) {
          const embeddingStore = db.createObjectStore('embeddings', { keyPath: 'url' });
        }

        if (!db.objectStoreNames.contains('enhancedData')) {
          const enhancedDataStore = db.createObjectStore('enhancedData', { keyPath: 'id' });
          enhancedDataStore.createIndex('url', 'url', { unique: false });
          enhancedDataStore.createIndex('timestamp', 'timestamp', { unique: false });
          enhancedDataStore.createIndex('domain', 'domain', { unique: false });
          enhancedDataStore.createIndex('pageType', 'pageType', { unique: false });
          enhancedDataStore.createIndex('timeOfDay', 'timeOfDay', { unique: false });
        }
      };
    });
  }

  async createObjectStores(db) {
    // Object stores are created in onupgradeneeded
    console.log('Database initialized with object stores');
  }

  setupEventListeners() {
    // Listen for tab changes
    chrome.tabs.onActivated.addListener(this.handleTabActivated.bind(this));
    chrome.tabs.onUpdated.addListener(this.handleTabUpdated.bind(this));
    
    // Listen for window focus changes
    chrome.windows.onFocusChanged.addListener(this.handleWindowFocusChanged.bind(this));
    
    // Listen for idle state changes
    chrome.idle.onStateChanged.addListener(this.handleIdleStateChanged.bind(this));
    
    // Listen for history changes
    chrome.history.onVisited.addListener(this.handleHistoryVisited.bind(this));
    
    // Listen for messages from popup/content scripts
    chrome.runtime.onMessage.addListener(this.handleMessage.bind(this));
    
    // Setup keyboard commands
    this.setupKeyboardCommands();
  }

  async handleTabActivated(activeInfo) {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    await this.recordTabActivity(tab);
  }

  async handleTabUpdated(tabId, changeInfo, tab) {
    if (changeInfo.status === 'complete' && tab.url) {
      await this.recordTabActivity(tab);
      
      // Update current tasks analysis when tabs change
      // TUNABLE: Debounce this to avoid too frequent updates
      if (!this.taskUpdateTimer) {
        this.taskUpdateTimer = setTimeout(() => {
          this.updateCurrentTasks();
          this.taskUpdateTimer = null;
        }, 2000); // Wait 2 seconds after last tab change
      }
    }
  }

  async handleWindowFocusChanged(windowId) {
    if (windowId === chrome.windows.WINDOW_ID_NONE) {
      // Window lost focus
      this.recordInactivity();
    } else {
      // Window gained focus
      this.recordActivity();
    }
  }

  handleIdleStateChanged(state) {
    this.isIdle = state === 'idle';
    if (this.isIdle) {
      this.recordInactivity();
    } else {
      this.recordActivity();
    }
  }

  async handleHistoryVisited(historyItem) {
    await this.recordHistoryEntry(historyItem);
  }

  async recordTabActivity(tab) {
    const timestamp = Date.now();
    const domain = this.extractDomain(tab.url);
    
    const activity = {
      type: 'tab_activity',
      tabId: tab.id,
      url: tab.url,
      title: tab.title,
      domain: domain,
      timestamp: timestamp,
      sessionId: this.currentSession?.id
    };
    
    await this.storeActivity(activity);
    this.recordActivity();
  }

  async recordHistoryEntry(historyItem) {
    const domain = this.extractDomain(historyItem.url);
    
    const entry = {
      type: 'history_entry',
      url: historyItem.url,
      title: historyItem.title,
      domain: domain,
      timestamp: historyItem.lastVisitTime,
      visitCount: historyItem.visitCount,
      sessionId: this.currentSession?.id
    };
    
    await this.storeHistoryEntry(entry);
  }

  recordActivity() {
    this.lastActivity = Date.now();
    this.isIdle = false;
  }

  recordInactivity() {
    // Mark current session as inactive
    if (this.currentSession) {
      this.currentSession.lastActivity = Date.now();
    }
  }

  extractDomain(url) {
    try {
      return new URL(url).hostname;
    } catch {
      return 'unknown';
    }
  }

  async storeActivity(activity) {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['historyEntries'], 'readwrite');
      const store = transaction.objectStore('historyEntries');
      const request = store.add(activity);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async storeHistoryEntry(entry) {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['historyEntries'], 'readwrite');
      const store = transaction.objectStore('historyEntries');
      const request = store.add(entry);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // MISSING METHOD - ADD THIS
  async storeEngagementData(engagementData) {
    console.log('Storing engagement data:', engagementData);
    
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['historyEntries'], 'readwrite');
      const store = transaction.objectStore('historyEntries');
      
      const entry = {
        type: 'engagement_data',
        url: engagementData.url,
        title: engagementData.title,
        domain: this.extractDomain(engagementData.url),
        timestamp: engagementData.timestamp,
        engagement: {
          // Raw metrics
          timeOnPage: engagementData.timeOnPage,
          focusTime: engagementData.focusTime || 0,
          activeTime: engagementData.activeTime || 0,
          idleTime: engagementData.idleTime || 0,
          backgroundTime: engagementData.backgroundTime || 0,
          scrollDepth: engagementData.scrollDepth || 0,
          mouseMovements: engagementData.mouseMovements || 0,
          keyPresses: engagementData.keyPresses || 0,
          
          // Context and scoring
          pageType: engagementData.pageType || 'webpage',
          contextualScore: engagementData.contextualScore || 0,
          confidence: engagementData.confidence || 0,
          
          // Time breakdown
          timeBreakdown: engagementData.timeBreakdown || {}
        },
        sessionId: this.currentSession?.id
      };
      
      const request = store.add(entry);
      
      request.onsuccess = () => {
        console.log('Engagement data stored successfully');
        resolve(request.result);
      };
      
      request.onerror = () => {
        console.error('Error storing engagement data:', request.error);
        reject(request.error);
      };
    });
  }

  async storeEnhancedEngagementData(pageData) {
    console.log('Storing enhanced engagement data:', pageData);
    
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['enhancedData'], 'readwrite');
      const store = transaction.objectStore('enhancedData');
      
      const entry = {
        id: `${pageData.url}_${pageData.timestamp}`,
        url: pageData.url,
        title: pageData.title,
        domain: pageData.domain,
        timestamp: pageData.timestamp,
        
        // Enhanced metadata for semantic search
        description: pageData.description || '',
        keywords: pageData.keywords || [],
        pageType: pageData.pageType || 'webpage',
        contentSummary: pageData.contentSummary || '',
        
        // Engagement metrics
        timeOnPage: pageData.timeOnPage || 0,
        focusTime: pageData.focusTime || 0,
        activeTime: pageData.activeTime || 0,
        scrollDepth: pageData.scrollDepth || 0,
        maxScrollDepth: pageData.maxScrollDepth || 0,
        
        // Interaction data
        mouseMovements: pageData.mouseMovements || 0,
        keyPresses: pageData.keyPresses || 0,
        clicks: pageData.clicks || 0,
        scrollEvents: pageData.scrollEvents || 0,
        
        // Content indicators
        hasVideo: pageData.hasVideo || false,
        hasAudio: pageData.hasAudio || false,
        hasImages: pageData.hasImages || false,
        hasCode: pageData.hasCode || false,
        textLength: pageData.textLength || 0,
        readingTime: pageData.readingTime || 0,
        
        // Session context
        sessionId: this.currentSession?.id,
        visitDuration: pageData.timeOnPage || 0,
        timeOfDay: new Date(pageData.timestamp).getHours(),
        dayOfWeek: new Date(pageData.timestamp).getDay()
      };
      
      const request = store.put(entry); // Use put to update existing entries
      
      request.onsuccess = () => {
        console.log('Enhanced engagement data stored successfully');
        resolve(request.result);
      };
      
      request.onerror = () => {
        console.error('Error storing enhanced engagement data:', request.error);
        reject(request.error);
      };
    });
  }

  startSessionTracking() {
    // Check for new sessions every 5 minutes
    setInterval(() => {
      this.checkForNewSession();
    }, 5 * 60 * 1000);
    
    // Start first session
    this.startNewSession();
  }

  checkForNewSession() {
    const now = Date.now();
    const timeSinceLastActivity = now - this.lastActivity;
    
    // Start new session if inactive for more than 15 minutes
    if (timeSinceLastActivity > 15 * 60 * 1000) {
      this.endCurrentSession();
      this.startNewSession();
    }
  }

  startNewSession() {
    this.currentSession = {
      id: Date.now(),
      startTime: Date.now(),
      lastActivity: Date.now(),
      entries: []
    };
    
    console.log('New session started:', this.currentSession.id);
  }

  async endCurrentSession() {
    if (this.currentSession) {
      this.currentSession.endTime = Date.now();
      await this.storeSession(this.currentSession);
      console.log('Session ended:', this.currentSession.id);
    }
  }

  async storeSession(session) {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['sessions'], 'readwrite');
      const store = transaction.objectStore('sessions');
      const request = store.add(session);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  handleMessage(request, sender, sendResponse) {
    console.log('=== BACKGROUND TEST 3 ===');
    console.log('Background script handling message:', request.action);
    
    try {
      switch (request.action) {
        case 'ping':
          console.log('Test 3A: Handling ping');
          const pingResponse = { success: true, message: 'pong', timestamp: Date.now() };
          console.log('Sending ping response:', pingResponse);
          sendResponse(pingResponse);
          break;
          
        case 'test':
          console.log('Test 3B: Handling test message');
          const testResponse = { 
            success: true, 
            message: 'Background script is working',
            receivedAt: Date.now(),
            originalMessage: request
          };
          console.log('Sending test response:', testResponse);
          sendResponse(testResponse);
          break;
          
        case 'getHistoryData':
          console.log('Test 3C: Handling getHistoryData');
          console.log('Filters received:', request.filters);
          
          // Handle async operation properly
          this.getHistoryData(request.filters)
            .then(data => {
              console.log('Retrieved data:', data ? data.length : 'null', 'entries');
              
              const historyResponse = {
                success: true,
                data: data || [], // Ensure it's always an array
                timestamp: Date.now(),
                filtersUsed: request.filters
              };
              
              console.log('Sending response:', historyResponse);
              sendResponse(historyResponse);
            })
            .catch(error => {
              console.error('Error getting history data:', error);
              sendResponse({
                success: false, 
                error: error.message,
                data: [], // Return empty array on error
                timestamp: Date.now() 
              });
            });
          break;
          
        case 'getCurrentTasks':
          console.log('Handling getCurrentTasks request');
          this.getCurrentTasks()
            .then(tasks => {
              sendResponse({
                success: true,
                data: tasks
              });
            })
            .catch(error => {
              console.error('Error getting current tasks:', error);
              this.getFallbackTasks().then(fallback => {
                sendResponse({
                  success: false,
                  error: error.message,
                  data: fallback
                });
              });
            });
          break;
          
        case 'updateCurrentTasks':
          console.log('Handling updateCurrentTasks request');
          this.updateCurrentTasks()
            .then(() => {
              sendResponse({
                success: true,
                message: 'Tasks updated'
              });
            })
            .catch(error => {
              console.error('Error updating current tasks:', error);
              sendResponse({
                success: false,
                error: error.message
              });
            });
          break;

        case 'recordEngagement':
          console.log('Recording engagement data');
          this.storeEngagementData(request.data)
            .then(() => {
              sendResponse({ success: true });
            })
            .catch(error => {
              console.error('Error storing engagement data:', error);
              sendResponse({ success: false, error: error.message });
            });
          break;
        case 'powerBackend':
          console.log('Power button pressed: starting/restarting backend');
          this.startBackend()
            .then(() => {
              sendResponse({ success: true, message: 'Backend start/restart triggered' });
            })
            .catch(error => {
              sendResponse({ success: false, error: error.message });
            });
          break;

        case 'recordEnhancedEngagement':
          console.log('Recording enhanced engagement data');
          this.storeEnhancedEngagementData(request.data)
            .then(() => {
              sendResponse({ success: true });
            })
            .catch(error => {
              console.error('Error storing enhanced engagement data:', error);
              sendResponse({ success: false, error: error.message });
            });
          break;

        case 'openDashboard':
          console.log('Opening dashboard');
          chrome.tabs.create({ url: chrome.runtime.getURL('dashboard/index.html') });
          sendResponse({ success: true });
          break;

        case 'openClassicHistory':
          console.log('Opening classic history');
          chrome.tabs.create({ url: 'chrome://history/' });
          sendResponse({ success: true });
          break;

        // Dashboard data request handlers
        case 'getCurrentTabs':
          console.log('Dashboard requesting current tabs data');
          chrome.tabs.query({})
            .then(tabs => {
              const tabData = tabs.map(tab => ({
                id: tab.id,
                title: tab.title || 'Untitled',
                url: tab.url || '',
                domain: this.extractDomain(tab.url || ''),
                active: tab.active,
                timestamp: Date.now()
              }));
              
              sendResponse({
                success: true,
                data: tabData
              });
            })
            .catch(error => {
              console.error('Error getting current tabs for dashboard:', error);
              sendResponse({
                success: false,
                error: error.message,
                data: []
              });
            });
          break;

        case 'getHistoricalData':
          console.log('Dashboard requesting historical data');
          const { startDate, endDate } = request;
          
          this.getHistoricalDataForPeriod(startDate, endDate)
            .then(data => {
              sendResponse({
                success: true,
                data: data || []
              });
            })
            .catch(error => {
              console.error('Error getting historical data for dashboard:', error);
              sendResponse({
                success: false,
                error: error.message,
                data: []
              });
            });
          break;

        case 'getEnhancedData':
          console.log('Dashboard requesting enhanced engagement data');
          const { startDate: enhancedStartDate, endDate: enhancedEndDate } = request;
          
          this.getEnhancedDataForPeriod(enhancedStartDate, enhancedEndDate)
            .then(data => {
              sendResponse({
                success: true,
                data: data || []
              });
            })
            .catch(error => {
              console.error('Error getting enhanced data for dashboard:', error);
              sendResponse({
                success: false,
                error: error.message,
                data: []
              });
            });
          break;

        default:
          console.log('Unknown action:', request.action);
          sendResponse({ 
            success: false, 
            error: `Unknown action: ${request.action}` 
          });
      }
    } catch (error) {
      console.error('=== BACKGROUND TEST 3 ERROR ===');
      console.error('Error in handleMessage:', error);
      sendResponse({
        success: false,
        error: error.message,
        data: []
      });
    }
    
    // CRITICAL: Return true for async response
    return true;
  }

  async getHistoryData(filters = {}) {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['historyEntries'], 'readonly');
      const store = transaction.objectStore('historyEntries');
      const request = store.getAll();
      
      request.onsuccess = () => {
        let data = request.result;
        
        // Apply filters
        if (filters.startDate) {
          data = data.filter(entry => entry.timestamp >= filters.startDate);
        }
        if (filters.endDate) {
          data = data.filter(entry => entry.timestamp <= filters.endDate);
        }
        if (filters.domain) {
          data = data.filter(entry => entry.domain === filters.domain);
        }
        
        resolve(data);
      };
      
      request.onerror = () => reject(request.error);
    });
  }

  async getSessions(filters = {}) {
    const db = await this.openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(['sessions'], 'readonly');
      const store = transaction.objectStore('sessions');
      const request = store.getAll();
      
      request.onsuccess = () => {
        let data = request.result;
        
        // Apply filters
        if (filters.startDate) {
          data = data.filter(session => session.startTime >= filters.startDate);
        }
        if (filters.endDate) {
          data = data.filter(session => session.endTime <= filters.endDate);
        }
        
        resolve(data);
      };
      
      request.onerror = () => reject(request.error);
    });
  }

  async openDashboard() {
    await chrome.tabs.create({
      url: chrome.runtime.getURL('dashboard/index.html')  // Use extension's dashboard
    });
  }

  setupKeyboardCommands() {
    chrome.commands.onCommand.addListener((command) => {
      switch (command) {
        case '_execute_action':
          this.openDashboard();
          break;
        case 'open_classic_history':
          this.openClassicHistory();
          break;
      }
    });
  }

  async openClassicHistory() {
    await chrome.tabs.create({
      url: 'chrome://history/'
    });
  }

  async getCurrentTasks() {
    try {
      console.log('Getting current tasks...');
      
      // Get all open tabs across all windows
      const tabs = await chrome.tabs.query({});
      console.log(`Found ${tabs.length} open tabs`);
      
      // Format tab data for backend analysis
      const tabData = tabs.map(tab => ({
        id: tab.id,
        title: tab.title || '',
        url: tab.url || '',
        domain: this.extractDomain(tab.url || ''),
        is_active: tab.active
      }));
      
      // Try to send to backend for AI analysis
      try {
        const response = await fetch('http://localhost:5000/api/analyze-current-tasks', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ tabs: tabData })
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('Current tasks analysis result:', result);
        
        if (result.success) {
          // Store current tasks for popup display
          await chrome.storage.local.set({
            currentTasks: result.data,
            lastTaskUpdate: Date.now()
          });
          
          return result.data;
        } else {
          throw new Error(result.error || 'Unknown analysis error');
        }
      } catch (fetchError) {
        console.log('Backend not available, using fallback tasks:', fetchError.message);
        // Fall back to local processing
        return await this.getFallbackTasks();
      }
      
    } catch (error) {
      console.error('Error getting current tasks:', error);
      return await this.getFallbackTasks();
    }
  }

  async getFallbackTasks() {
    try {
      const tabs = await chrome.tabs.query({});
      
      // Simple domain-based grouping as fallback
      const domainGroups = {};
      tabs.forEach(tab => {
        const domain = this.extractDomain(tab.url || '');
        if (!domainGroups[domain]) {
          domainGroups[domain] = {
            category: 'Web Browsing',
            subcategory: domain,
            confidence: 0.5,
            tab_count: 0,
            tabs: []
          };
        }
        domainGroups[domain].tab_count++;
        domainGroups[domain].tabs.push({
          id: tab.id,
          title: tab.title,
          url: tab.url,
          domain: domain,
          is_active: tab.active
        });
      });
      
      const currentTasks = Object.values(domainGroups);
      
      return {
        current_tasks: currentTasks,
        task_count: currentTasks.length,
        dominant_task: currentTasks.length > 0 ? currentTasks.reduce((a, b) => a.tab_count > b.tab_count ? a : b) : null,
        multitasking_score: currentTasks.length > 1 ? 0.5 : 0.0,
        summary: `${tabs.length} tabs across ${currentTasks.length} domains (offline mode)`,
        total_tabs_analyzed: tabs.length,
        categorization_confidence: 0.5
      };
    } catch (error) {
      console.error('Error in fallback tasks:', error);
      return {
        current_tasks: [],
        task_count: 0,
        dominant_task: null,
        multitasking_score: 0.0,
        summary: 'No tasks detected',
        total_tabs_analyzed: 0,
        categorization_confidence: 0.0
      };
    }
  }

  async getHistoricalDataForPeriod(startDate, endDate) {
    try {
      console.log(`Getting historical data from ${startDate} to ${endDate}`);
      
      const start = new Date(startDate);
      const end = new Date(endDate);
      
      // Get data from Chrome history API
      const historyItems = await chrome.history.search({
        text: '',
        startTime: start.getTime(),
        endTime: end.getTime(),
        maxResults: 1000 // Limit for performance
      });
      
      // Get data from IndexedDB
      const dbData = await this.getHistoryDataFromDB(start, end);
      
      // Combine and deduplicate
      const combinedData = [...historyItems, ...dbData];
      const uniqueData = combinedData.filter((item, index, array) => 
        array.findIndex(i => i.url === item.url && i.lastVisitTime === item.lastVisitTime) === index
      );
      
      // Enrich with domain and category information
      const enrichedData = uniqueData.map(item => ({
        ...item,
        domain: this.extractDomain(item.url),
        timestamp: item.lastVisitTime || Date.now(),
        visitCount: item.visitCount || 1,
        title: item.title || 'Untitled'
      }));
      
      console.log(`Retrieved ${enrichedData.length} historical entries`);
      return enrichedData;
      
    } catch (error) {
      console.error('Error getting historical data for period:', error);
      return [];
    }
  }

  async getEnhancedDataForPeriod(startDate, endDate) {
    try {
      console.log(`Getting enhanced data from ${startDate} to ${endDate}`);
      
      const start = new Date(startDate);
      const end = new Date(endDate);
      
      const db = await this.openDatabase();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(['enhancedData'], 'readonly');
        const store = transaction.objectStore('enhancedData');
        const index = store.index('timestamp');
        
        const range = IDBKeyRange.bound(start.getTime(), end.getTime());
        const getRequest = index.getAll(range);
        
        getRequest.onsuccess = () => {
          const enhancedData = getRequest.result || [];
          console.log(`Retrieved ${enhancedData.length} enhanced entries`);
          resolve(enhancedData);
        };
        
        getRequest.onerror = () => {
          console.error('Error reading enhanced data from IndexedDB:', getRequest.error);
          reject(getRequest.error);
        };
      });
      
    } catch (error) {
      console.error('Error getting enhanced data for period:', error);
      return [];
    }
  }

  async getHistoryDataFromDB(startDate, endDate) {
    return new Promise((resolve) => {
      try {
        const request = indexedDB.open('HistoryPlusDB', 1);
        
        request.onsuccess = (event) => {
          const db = event.target.result;
          
          if (!db.objectStoreNames.contains('historyEntries')) {
            resolve([]);
            return;
          }
          
          const transaction = db.transaction(['historyEntries'], 'readonly');
          const store = transaction.objectStore('historyEntries');
          const index = store.index('timestamp');
          
          const range = IDBKeyRange.bound(startDate.getTime(), endDate.getTime());
          const getRequest = index.getAll(range);
          
          getRequest.onsuccess = () => {
            resolve(getRequest.result || []);
          };
          
          getRequest.onerror = () => {
            console.error('Error reading from IndexedDB:', getRequest.error);
            resolve([]);
          };
        };
        
        request.onerror = () => {
          console.error('Error opening IndexedDB:', request.error);
          resolve([]);
        };
      } catch (error) {
        console.error('Error in getHistoryDataFromDB:', error);
        resolve([]);
      }
    });
  }

  async updateCurrentTasks() {
    try {
      console.log('Updating current tasks analysis...');
      this.currentTasks = await this.getCurrentTasks();
      
      // Notify popup and dashboard of task changes
      try {
        await chrome.runtime.sendMessage({
          action: 'currentTasksUpdated',
          tasks: this.currentTasks
        });
      } catch (e) {
        // Ignore if no listeners
      }
      
      console.log('Current tasks updated:', this.currentTasks);
    } catch (error) {
      console.error('Error updating current tasks:', error);
    }
  }
}

// Initialize the background script
new HistoryPlusBackground(); 