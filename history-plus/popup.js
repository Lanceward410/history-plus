// History Plus Popup Script
// Handles popup UI interactions and data display

// TEST 1: Basic Chrome API Detection
console.log('=== POPUP TEST 1: Chrome API Availability ===');
console.log('typeof chrome:', typeof chrome);
console.log('chrome exists:', typeof chrome !== 'undefined');
console.log('chrome.runtime exists:', typeof chrome?.runtime !== 'undefined');
console.log('chrome.runtime.sendMessage exists:', typeof chrome?.runtime?.sendMessage === 'function');
console.log('chrome.runtime.id:', chrome?.runtime?.id);
console.log('chrome.runtime.lastError:', chrome?.runtime?.lastError);
console.log('=== END TEST 1 ===\n');

class HistoryPlusPopup {
  constructor() {
    this.init();
  }

  async init() {
    this.setupEventListeners();
    await this.loadStats();
    console.log('History Plus popup initialized');
  }

  setupEventListeners() {
    document.getElementById('openDashboard').addEventListener('click', () => {
      this.openDashboard();
    });
    
    document.getElementById('viewHistory').addEventListener('click', () => {
      this.viewHistory();
    });
    
    document.getElementById('searchHistory').addEventListener('click', () => {
      this.searchHistory();
    });
    
    document.getElementById('settings').addEventListener('click', () => {
      this.openSettings();
    });
  }

  async loadStats() {
    console.log('=== POPUP TEST 2: Background Communication ===');
    
    try {
      // Test 2A: Simple ping test
      console.log('Test 2A: Sending ping message...');
      const pingResponse = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Ping timeout after 3 seconds'));
        }, 3000);

        chrome.runtime.sendMessage({ action: 'ping' }, (response) => {
          clearTimeout(timeout);
          console.log('Ping raw response:', response);
          console.log('Chrome last error after ping:', chrome.runtime.lastError);
          
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(response);
          }
        });
      });
      
      console.log('Test 2A Result - Ping successful:', pingResponse);
      
      // Test 2B: Background script existence test
      console.log('Test 2B: Testing background script response...');
      const testResponse = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Background test timeout after 5 seconds'));
        }, 5000);

        chrome.runtime.sendMessage({ 
          action: 'test',
          timestamp: Date.now()
        }, (response) => {
          clearTimeout(timeout);
          console.log('Test message raw response:', response);
          console.log('Chrome last error after test:', chrome.runtime.lastError);
          
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(response);
          }
        });
      });
      
      console.log('Test 2B Result - Background script responsive:', testResponse);
      
      // Test 2C: Actual getHistoryData test
      console.log('Test 2C: Testing actual getHistoryData...');
      const today = new Date();
      const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
      const endOfDay = startOfDay + (24 * 60 * 60 * 1000) - 1;
      
      const historyResponse = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('History data timeout after 10 seconds'));
        }, 10000);

        const message = {
          action: 'getHistoryData',
          filters: {
            startDate: startOfDay,
            endDate: endOfDay
          }
        };
        
        console.log('Sending history data request:', message);
        
        chrome.runtime.sendMessage(message, (response) => {
          clearTimeout(timeout);
          console.log('History data raw response:', response);
          console.log('Response type:', typeof response);
          console.log('Response is null:', response === null);
          console.log('Response is undefined:', response === undefined);
          console.log('Chrome last error after history:', chrome.runtime.lastError);
          
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(response);
          }
        });
      });
      
      console.log('Test 2C Result - History data response:', historyResponse);
      
      // Test 2D: Response structure validation
      console.log('Test 2D: Validating response structure...');
      if (historyResponse === undefined) {
        throw new Error('Response is undefined - background script not responding');
      }
      
      if (historyResponse === null) {
        throw new Error('Response is null - background script returned null');
      }
      
      if (typeof historyResponse !== 'object') {
        throw new Error(`Response is not an object - got ${typeof historyResponse}: ${historyResponse}`);
      }
      
      if (!historyResponse.hasOwnProperty('success')) {
        throw new Error(`Response missing 'success' property. Response keys: ${Object.keys(historyResponse)}`);
      }
      
      console.log('Test 2D Result - Response structure valid');
      
      // If we get here, try the original logic
      if (historyResponse.success) {
        console.log('SUCCESS: Updating stats with data:', historyResponse.data);
        this.updateStats(historyResponse.data);
      } else {
        console.error('FAILURE: Background script returned success=false:', historyResponse);
      }
      
    } catch (error) {
      console.error('=== TEST 2 FAILED ===');
      console.error('Error type:', error.constructor.name);
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
      console.error('Chrome runtime last error:', chrome?.runtime?.lastError);
      
      // Show fallback UI
      this.showTestErrorState(error.message);
    }
    
    console.log('=== END TEST 2 ===\n');
  }

  updateStats(data) {
    // Calculate today's visits
    const todayVisits = data.filter(entry => entry.type === 'history_entry').length;
    document.getElementById('todayVisits').textContent = todayVisits;
    
    // Calculate top domain
    const domainCounts = {};
    data.forEach(entry => {
      if (entry.domain) {
        domainCounts[entry.domain] = (domainCounts[entry.domain] || 0) + 1;
      }
    });
    
    const topDomain = Object.entries(domainCounts)
      .sort(([,a], [,b]) => b - a)[0];
    
    if (topDomain) {
      document.getElementById('topDomain').textContent = topDomain[0].substring(0, 10) + '...';
    } else {
      document.getElementById('topDomain').textContent = 'None';
    }
    
    // Calculate average time (placeholder)
    document.getElementById('avgTime').textContent = data.length > 0 ? '2.5m' : '0m';
    
    // BETTER METRIC: Today's sessions count
    // Count unique session IDs from today's data
    const sessionIds = new Set(data.map(entry => entry.sessionId).filter(id => id));
    document.getElementById('activeSessions').textContent = sessionIds.size;
  }

  async openDashboard() {
    try {
      await chrome.runtime.sendMessage({ action: 'openDashboard' });
      window.close();
    } catch (error) {
      console.error('Error opening dashboard:', error);
    }
  }

  viewHistory() {
    // Open Chrome's default history page
    chrome.tabs.create({ url: 'chrome://history/' });
    window.close();
  }

  searchHistory() {
    // Open a search interface (placeholder)
    chrome.tabs.create({ url: chrome.runtime.getURL('dashboard/search.html') });
    window.close();
  }

  openSettings() {
    // Open settings page
    chrome.tabs.create({ url: chrome.runtime.getURL('dashboard/settings.html') });
    window.close();
  }

  showTestErrorState(errorMessage) {
    console.log('Showing test error state:', errorMessage);
    
    // Update UI to show the specific error
    const elements = {
      'todayVisits': 'ERR',
      'topDomain': errorMessage.substring(0, 10),
      'avgTime': '0m',
      'activeSessions': '0'
    };
    
    Object.entries(elements).forEach(([id, value]) => {
      const element = document.getElementById(id);
      if (element) {
        element.textContent = value;
        element.style.color = 'red'; // Make errors visible
      }
    });
  }
}

// Initialize popup
new HistoryPlusPopup(); 