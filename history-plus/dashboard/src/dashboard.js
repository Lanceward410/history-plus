/**
 * History Plus Interactive Dashboard
 * Implements the dual pie chart system with intelligent categorization
 */

class HistoryPlusDashboard {
    constructor() {
        // Hard-coded colors for framework categories
        this.FRAMEWORK_COLORS = {
            'Social Media': '#FF6B9D',
            'Gaming': '#FF6B6B', 
            'Academic Research': '#4ECDC4',
            'Work/Professional': '#45B7D1',
            'Shopping': '#FFA07A',
            'Entertainment': '#98D8C8',
            'Learning/Tutorials': '#F7DC6F',
            'News/Information': '#BB8FCE',
            'AI/Development Tools': '#82E0AA',
            'Health/Wellness': '#F8C471'
        };

        // State management
        this.state = {
            currentPeriod: 'today',
            selectedCategory: null,
            selectedSubcategory: null,
            isBackendAvailable: false,
            isSplitView: false,
            aiMode: 'detecting',
            dataMode: this.loadDataModePreference(),
            focusOnlyMode: this.loadFocusOnlyPreference()
        };

        // Chart instances
        this.mainChart = null;
        this.subChart = null;
        this.hourlyChart = null;

        // Data
        this.rawData = [];
        this.processedData = {};
        this.timelineData = {};

        // Initialize
        this.init();
    }

    loadDataModePreference() {
        try {
            const savedMode = localStorage.getItem('historyplus-data-mode');
            if (savedMode && (savedMode === 'chrome-history' || savedMode === 'fused')) {
                console.log('Loaded saved data mode preference:', savedMode);
                return savedMode;
            }
        } catch (error) {
            console.warn('Could not load data mode preference from localStorage:', error);
        }
        // Default to fused mode (Enhanced data mode)
        return 'fused';
    }

    saveDataModePreference(mode) {
        try {
            localStorage.setItem('historyplus-data-mode', mode);
            console.log('Saved data mode preference:', mode);
        } catch (error) {
            console.warn('Could not save data mode preference to localStorage:', error);
        }
    }

    loadFocusOnlyPreference() {
        try {
            const savedFocusOnly = localStorage.getItem('historyplus-focus-only');
            if (savedFocusOnly !== null) {
                const isEnabled = savedFocusOnly === 'true';
                console.log('Loaded focus-only preference:', isEnabled);
                return isEnabled;
            }
        } catch (error) {
            console.warn('Could not load focus-only preference from localStorage:', error);
        }
        // Default to true (focus-only mode enabled by default)
        return true;
    }

    saveFocusOnlyPreference(enabled) {
        try {
            localStorage.setItem('historyplus-focus-only', enabled.toString());
            console.log('Saved focus-only preference:', enabled);
        } catch (error) {
            console.warn('Could not save focus-only preference to localStorage:', error);
        }
    }

    syncDataModeUI() {
        // Update the dropdown to match the current state
        const dataModeSelect = document.getElementById('dataModeSelect');
        if (dataModeSelect) {
            dataModeSelect.value = this.state.dataMode;
        }
        
        // Update the status display
        const statusDiv = document.getElementById('dataModeStatus');
        if (statusDiv) {
            if (this.state.dataMode === 'chrome-history') {
                statusDiv.textContent = 'Chrome data only';
                statusDiv.style.color = '#333';
            } else {
                statusDiv.textContent = 'Enhanced data mode';
                statusDiv.style.color = '#333';
            }
        }
        
        console.log('Synced data mode UI to:', this.state.dataMode);
    }

    syncFocusOnlyUI() {
        // Update the checkbox to match the current state
        const focusOnlyToggle = document.getElementById('focusOnlyToggle');
        if (focusOnlyToggle) {
            focusOnlyToggle.checked = this.state.focusOnlyMode;
        }
        
        // Update the status display
        const statusDiv = document.getElementById('focusToggleStatus');
        if (statusDiv) {
            if (this.state.focusOnlyMode) {
                statusDiv.textContent = 'Active sites only';
                statusDiv.style.color = '#4CAF50';
            } else {
                statusDiv.textContent = 'All sites';
                statusDiv.style.color = '#FF9800';
            }
        }
        
        console.log('Synced focus-only UI to:', this.state.focusOnlyMode);
    }

    async callBackendAPI(endpoint, method = 'GET', data = null) {
        /**
         * Mode-aware backend API call helper
         * Automatically includes current data mode in requests
         */
        try {
            const baseUrl = 'http://localhost:5000/api/enhanced';
            const url = new URL(`${baseUrl}${endpoint}`);
            
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };
            
            if (method === 'GET') {
                // Add mode as query parameter for GET requests
                url.searchParams.append('mode', this.state.dataMode);
            } else if (data) {
                // Add mode to request body for POST/PUT requests
                data.mode = this.state.dataMode;
                options.body = JSON.stringify(data);
            }
            
            console.log(`Making mode-aware API call: ${method} ${url} (mode: ${this.state.dataMode})`);
            
            const response = await fetch(url, options);
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.message || `API call failed: ${response.status}`);
            }
            
            return result;
            
        } catch (error) {
            console.error(`Backend API call failed: ${endpoint}`, error);
            throw error;
        }
    }

    async getAnalyticsData(type = 'categories', days = 30) {
        /**
         * Get analytics data from backend with mode awareness
         * Future-ready method for when analytics integration is expanded
         */
        try {
            const endpoint = `/analytics/${type}`;
            const data = await this.callBackendAPI(endpoint, 'GET');
            
            console.log(`Retrieved ${type} analytics for ${days} days in ${this.state.dataMode} mode`);
            return data;
            
        } catch (error) {
            console.warn(`Analytics ${type} not available:`, error.message);
            
            // Return fallback data structure for development
            return {
                success: false,
                message: `Analytics ${type} endpoint not available`,
                mode: this.state.dataMode,
                fallback: true
            };
        }
    }

    async init() {
        console.log('Initializing History Plus Dashboard...');
        
        this.setupEventListeners();
        this.updateResourceMonitor();
        
        // Apply initial feature gating based on loaded mode
        this.applyFeatureGating(this.state.dataMode);
        
        // Set the data mode dropdown to match the loaded preference
        this.syncDataModeUI();
        this.syncFocusOnlyUI();
        
        // Check if we're in extension context
        if (typeof chrome !== 'undefined' && chrome.runtime) {
            console.log('Running in extension context');
            await this.checkBackendStatus();
        } else {
            console.log('Running standalone, using mock data');
            this.showFallbackNotification();
            this.loadMockData();
        }
    }

    setupEventListeners() {
        // Time period buttons
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const period = e.target.dataset.period;
                this.state.currentPeriod = period;
                
                if (period === 'custom') {
                    document.getElementById('customRange').style.display = 'flex';
                } else {
                    document.getElementById('customRange').style.display = 'none';
                    this.loadData();
                }
            });
        });

        // Custom date range
        const applyBtn = document.getElementById('applyCustomRange');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                this.loadData();
            });
        }

        // Retry button
        const retryBtn = document.getElementById('retryBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                this.loadData();
            });
        }

        // Troubleshoot button
        const troubleshootBtn = document.getElementById('troubleshootBtn');
        if (troubleshootBtn) {
            troubleshootBtn.addEventListener('click', () => {
                this.showTroubleshootingGuide();
            });
        }

        // Reset button
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetToMainView();
            });
        }

        // Power button
        const powerBtn = document.getElementById('powerBtn');
        if (powerBtn) {
            powerBtn.addEventListener('click', () => {
                this.handlePowerButton();
            });
        }

        // Data mode toggle
        const dataModeSelect = document.getElementById('dataModeSelect');
        if (dataModeSelect) {
            dataModeSelect.addEventListener('change', (e) => {
                this.handleDataModeChange(e.target.value);
            });
        }

        // Focus-only toggle
        const focusOnlyToggle = document.getElementById('focusOnlyToggle');
        if (focusOnlyToggle) {
            focusOnlyToggle.addEventListener('change', (e) => {
                this.handleFocusOnlyChange(e.target.checked);
            });
        }

        // Filter event listeners
        this.setupFilterListeners();
        
        // Timeline event listeners
        this.setupTimelineListeners();
    }

    setupFilterListeners() {
        // Active Focus Time filter
        const durationFilter = document.getElementById('durationFilter');
        const durationValue = document.getElementById('durationValue');
        if (durationFilter && durationValue) {
            durationFilter.addEventListener('input', (e) => {
                durationValue.textContent = e.target.value;
                this.applyFilters();
            });
        }

        // Time range dual slider
        this.setupTimeRangeSlider();

        // Semantic search
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('semanticSearch');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', () => {
                this.performSemanticSearch();
            });
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.performSemanticSearch();
                }
            });
        }
    }

    setupTimeRangeSlider() {
        const minSlider = document.getElementById('timeRangeMin');
        const maxSlider = document.getElementById('timeRangeMax');
        const minDisplay = document.getElementById('timeMinDisplay');
        const maxDisplay = document.getElementById('timeMaxDisplay');
        const sliderRange = document.querySelector('.slider-range');

        if (!minSlider || !maxSlider || !minDisplay || !maxDisplay || !sliderRange) return;

        const updateTimeDisplay = () => {
            const minTime = this.minutesToTime(parseInt(minSlider.value));
            const maxTime = this.minutesToTime(parseInt(maxSlider.value));
            minDisplay.textContent = minTime;
            maxDisplay.textContent = maxTime;

            // Update slider range visual
            const minPercent = (minSlider.value / 1440) * 100;
            const maxPercent = (maxSlider.value / 1440) * 100;
            sliderRange.style.left = minPercent + '%';
            sliderRange.style.width = (maxPercent - minPercent) + '%';
        };

        const handleSliderChange = () => {
            // Ensure minimum 5-minute gap
            if (parseInt(maxSlider.value) - parseInt(minSlider.value) < 5) {
                if (parseInt(minSlider.value) >= parseInt(maxSlider.value)) {
                    maxSlider.value = parseInt(minSlider.value) + 5;
                }
            }
            updateTimeDisplay();
            this.applyFilters();
        };

        minSlider.addEventListener('input', handleSliderChange);
        maxSlider.addEventListener('input', handleSliderChange);

        // Initialize display
        updateTimeDisplay();
    }

    minutesToTime(minutes) {
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        const period = hours >= 12 ? 'PM' : 'AM';
        const displayHour = hours === 0 ? 12 : hours > 12 ? hours - 12 : hours;
        return `${displayHour}:${mins.toString().padStart(2, '0')} ${period}`;
    }

    applyFilters() {
        // Get filter values
        const minDuration = parseInt(document.getElementById('durationFilter')?.value || 0);
        const minTime = parseInt(document.getElementById('timeRangeMin')?.value || 0);
        const maxTime = parseInt(document.getElementById('timeRangeMax')?.value || 1440);

        // Apply filters and update website list
        this.updateWebsiteList({
            minDuration,
            timeRange: { min: minTime, max: maxTime }
        });
    }

    async performSemanticSearch() {
        const searchInput = document.getElementById('semanticSearch');
        const query = searchInput?.value?.trim();
        
        if (!query) {
            this.updateWebsiteList();
            return;
        }

        console.log('Performing semantic search:', query);
        
        // Show loading state
        const websiteList = document.getElementById('websiteList');
        if (websiteList) {
            websiteList.innerHTML = '<div class="loading-websites">🔍 Searching...</div>';
        }

        try {
            // Use the comprehensive semantic search implementation
            const results = await this.performComprehensiveSemanticSearch(query);
            this.displaySearchResults(results, query);
        } catch (error) {
            console.error('Semantic search error:', error);
            if (websiteList) {
                websiteList.innerHTML = '<div class="loading-websites">❌ Search failed. Please try again.</div>';
            }
        }
    }

    async performComprehensiveSemanticSearch(query) {
        console.log('🔍 Starting comprehensive semantic search...');
        
        // Try 1: Enhanced API service (preferred - has LLM backend)
        try {
            console.log('📡 Testing Enhanced API Service...');
            
            const data = await this.callBackendAPI('/semantic-search', 'POST', {
                query: query,
                filters: {},
                options: {
                    limit: 50,
                    min_similarity: 0.3
                }
            });

            if (data.success && data.data && data.data.results) {
                console.log('✅ Enhanced API Service successful:', data.data.results.length, 'results');
                return data.data.results;
            }
        } catch (error) {
            console.warn('⚠️ Enhanced API Service failed:', error.message);
            
            // Handle mode-specific errors
            if (error.message && error.message.includes('Chrome History mode')) {
                console.log('🔄 Semantic search not available in Chrome data only mode - feature properly gated');
                return []; // Return empty results for chrome-history mode
            }
        }

        // Try 2: Local semantic search as fallback
        try {
            console.log('🔍 Falling back to local semantic search...');
            const localResults = await this.performLocalSemanticSearch(query);
            
            if (localResults && localResults.length > 0) {
                console.log('✅ Local Semantic Search successful:', localResults.length, 'results');
                return localResults;
            }
        } catch (error) {
            console.warn('⚠️ Local Semantic Search failed:', error.message);
        }

        console.log('❌ All semantic search methods failed');
        return [];
    }

    async performLocalSemanticSearch(query) {
        console.log('Performing local semantic search fallback...');
        
        if (!this.rawData || this.rawData.length === 0) {
            console.warn('No data available for local search');
            return [];
        }

        const results = [];
        const queryLower = query.toLowerCase();
        const queryWords = queryLower.split(/\s+/).filter(word => word.length > 2);

        for (const item of this.rawData) {
            let relevance = 0;
            
            // Title relevance
            if (item.title) {
                relevance += this.calculateLocalTextRelevance(queryWords, item.title.toLowerCase()) * 0.4;
            }
            
            // URL relevance
            if (item.url) {
                relevance += this.calculateLocalTextRelevance(queryWords, item.url.toLowerCase()) * 0.3;
            }
            
            // Domain relevance
            if (item.domain) {
                relevance += this.calculateLocalTextRelevance(queryWords, item.domain.toLowerCase()) * 0.2;
            }
            
            // Category relevance
            if (item.category) {
                relevance += this.calculateLocalTextRelevance(queryWords, item.category.toLowerCase()) * 0.1;
            }

            if (relevance > 0.1) { // Minimum relevance threshold
                results.push({
                    ...item,
                    similarity: Math.min(relevance, 1.0),
                    relevance_score: relevance
                });
            }
        }

        // Sort by relevance
        results.sort((a, b) => b.relevance_score - a.relevance_score);
        
        console.log('Local search found:', results.length, 'results');
        return results.slice(0, 20); // Limit results
    }

    calculateLocalTextRelevance(queryWords, text) {
        if (!queryWords || queryWords.length === 0 || !text) return 0;
        
        let matches = 0;
        for (const queryWord of queryWords) {
            if (text.includes(queryWord)) {
                matches++;
                // Boost for exact phrase matches
                if (text.includes(queryWords.join(' '))) {
                    matches += 0.5;
                }
            }
        }
        
        return queryWords.length > 0 ? matches / queryWords.length : 0;
    }

    setupTimelineListeners() {
        // Timeline view toggle buttons
        document.querySelectorAll('.timeline-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Update active button
                document.querySelectorAll('.timeline-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const view = e.target.dataset.view;
                this.switchTimelineView(view);
            });
        });
    }

    switchTimelineView(view) {
        // Hide all timeline views
        document.querySelectorAll('.timeline-view').forEach(viewEl => {
            viewEl.style.display = 'none';
        });
        
        // Show selected view
        const targetView = document.getElementById(`${view}View`);
        if (targetView) {
            targetView.style.display = 'block';
            
            // Initialize the view with data
            switch (view) {
                case 'hourly':
                    this.renderHourlyChart();
                    break;
                case 'categories':
                    this.renderCategoryTimeline();
                    break;
                case 'focus':
                    this.renderFocusHeatmap();
                    break;
            }
        }
    }

    generateTimelineData() {
        if (!this.rawData || this.rawData.length === 0) return;
        
        // Process data for timeline visualizations
        this.timelineData = {
            hourlyActivity: this.generateHourlyActivity(),
            categoryTimeline: this.generateCategoryTimeline(),
            focusHeatmap: this.generateFocusHeatmap(),
            insights: this.generateTimelineInsights()
        };
        
        // Update timeline stats
        this.updateTimelineStats();
    }

    generateHourlyActivity() {
        const hourlyData = Array.from({ length: 24 }, (_, hour) => ({
            hour,
            activeTime: 0,
            totalTime: 0,
            backgroundTime: 0,
            sessions: 0
        }));
        
        this.rawData.forEach(item => {
            const hour = new Date(item.timestamp).getHours();
            const activeTime = item.activeTime || 0;
            const totalTime = item.timeSpent || 0;
            
            hourlyData[hour].activeTime += activeTime;
            hourlyData[hour].totalTime += totalTime;
            hourlyData[hour].backgroundTime += Math.max(0, totalTime - activeTime);
            hourlyData[hour].sessions += 1;
        });
        
        return hourlyData;
    }

    generateCategoryTimeline() {
        const categories = Object.keys(this.processedData.categories || {});
        const timelineCategories = {};
        
        categories.forEach(category => {
            timelineCategories[category] = Array.from({ length: 24 }, () => ({
                activeTime: 0,
                totalTime: 0,
                sessions: []
            }));
        });
        
        this.rawData.forEach(item => {
            const hour = new Date(item.timestamp).getHours();
            const category = item.category || this.classifyDomain(item.domain || item.url);
            const activeTime = item.activeTime || 0;
            const totalTime = item.timeSpent || 0;
            
            if (timelineCategories[category]) {
                timelineCategories[category][hour].activeTime += activeTime;
                timelineCategories[category][hour].totalTime += totalTime;
                timelineCategories[category][hour].sessions.push({
                    url: item.url,
                    title: item.title,
                    activeTime: activeTime
                });
            }
        });
        
        return timelineCategories;
    }

    generateFocusHeatmap() {
        const heatmapData = Array.from({ length: 24 }, (_, hour) => ({
            hour,
            focusIntensity: 0,
            sessions: 0,
            totalActiveTime: 0
        }));
        
        this.rawData.forEach(item => {
            const hour = new Date(item.timestamp).getHours();
            const activeTime = item.activeTime || 0;
            const totalTime = item.timeSpent || 0;
            const focusRatio = totalTime > 0 ? activeTime / totalTime : 0;
            
            heatmapData[hour].focusIntensity += focusRatio;
            heatmapData[hour].sessions += 1;
            heatmapData[hour].totalActiveTime += activeTime;
        });
        
        // Calculate average focus intensity per hour
        heatmapData.forEach(hourData => {
            if (hourData.sessions > 0) {
                hourData.focusIntensity = hourData.focusIntensity / hourData.sessions;
            }
        });
        
        return heatmapData;
    }

    generateTimelineInsights() {
        const hourlyData = this.timelineData.hourlyActivity || this.generateHourlyActivity();
        
        // Find peak hour
        const peakHour = hourlyData.reduce((max, current) => 
            current.activeTime > max.activeTime ? current : max
        );
        
        // Calculate focus efficiency by time periods
        const morningHours = hourlyData.slice(6, 12);
        const afternoonHours = hourlyData.slice(12, 18);
        const eveningHours = hourlyData.slice(18, 24);
        
        const calculateEfficiency = (hours) => {
            const totalActive = hours.reduce((sum, hour) => sum + hour.activeTime, 0);
            const totalTime = hours.reduce((sum, hour) => sum + hour.totalTime, 0);
            return totalTime > 0 ? (totalActive / totalTime) * 100 : 0;
        };
        
        // Calculate average session length
        const totalSessions = hourlyData.reduce((sum, hour) => sum + hour.sessions, 0);
        const totalActiveTime = hourlyData.reduce((sum, hour) => sum + hour.activeTime, 0);
        const avgSession = totalSessions > 0 ? totalActiveTime / totalSessions / 60 : 0; // in minutes
        
        return {
            peakHour: this.formatHour(peakHour.hour),
            avgSession: Math.round(avgSession),
            focusScore: Math.round(calculateEfficiency(hourlyData)),
            morningEfficiency: Math.round(calculateEfficiency(morningHours)),
            afternoonEfficiency: Math.round(calculateEfficiency(afternoonHours)),
            eveningEfficiency: Math.round(calculateEfficiency(eveningHours))
        };
    }

    formatHour(hour) {
        const period = hour >= 12 ? 'PM' : 'AM';
        const displayHour = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;
        return `${displayHour}:00 ${period}`;
    }

    renderHourlyChart() {
        const canvas = document.getElementById('hourlyChart');
        if (!canvas || !this.timelineData.hourlyActivity) return;
        
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        if (this.hourlyChart) {
            this.hourlyChart.destroy();
        }
        
        const hourlyData = this.timelineData.hourlyActivity;
        const labels = hourlyData.map(data => this.formatHour(data.hour));
        const activeTimeData = hourlyData.map(data => data.activeTime / 60); // Convert to minutes
        const totalTimeData = hourlyData.map(data => data.totalTime / 60);
        const backgroundTimeData = hourlyData.map(data => data.backgroundTime / 60);
        
        this.hourlyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Active Focus Time',
                        data: activeTimeData,
                        backgroundColor: 'rgba(76, 175, 80, 0.8)',
                        borderColor: '#4CAF50',
                        borderWidth: 1,
                        order: 1
                    },
                    {
                        label: 'Total Browse Time',
                        data: totalTimeData,
                        backgroundColor: 'rgba(33, 150, 243, 0.6)',
                        borderColor: '#2196F3',
                        borderWidth: 1,
                        order: 2
                    },
                    {
                        label: 'Background Time',
                        data: backgroundTimeData,
                        backgroundColor: 'rgba(255, 152, 0, 0.5)',
                        borderColor: '#FF9800',
                        borderWidth: 1,
                        order: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false // We have custom legend
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = Math.round(context.parsed.y);
                                return `${context.dataset.label}: ${value} min`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: 'rgba(255, 255, 255, 0.8)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { 
                            color: 'rgba(255, 255, 255, 0.8)',
                            callback: (value) => `${value} min`
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    renderCategoryTimeline() {
        const categoryRows = document.getElementById('categoryRows');
        if (!categoryRows || !this.timelineData.categoryTimeline) return;
        
        const categories = Object.keys(this.timelineData.categoryTimeline);
        categoryRows.innerHTML = '';
        
        categories.forEach(category => {
            const categoryData = this.timelineData.categoryTimeline[category];
            const color = this.FRAMEWORK_COLORS[category] || this.generateRandomColor();
            
            const row = document.createElement('div');
            row.className = 'category-row';
            
            const maxActiveTime = Math.max(...categoryData.map(hour => hour.activeTime));
            
            row.innerHTML = `
                <div class="category-label">${category}</div>
                <div class="category-timeline-bar">
                    ${categoryData.map((hourData, hour) => {
                        const width = maxActiveTime > 0 ? (hourData.activeTime / maxActiveTime) * 100 : 0;
                        const left = (hour / 24) * 100;
                        return width > 0 ? `
                            <div class="timeline-segment" 
                                 style="left: ${left}%; width: ${width/24}%; background-color: ${color};"
                                 title="${this.formatHour(hour)}: ${Math.round(hourData.activeTime/60)} min active">
                            </div>
                        ` : '';
                    }).join('')}
                </div>
            `;
            
            categoryRows.appendChild(row);
        });
    }

    renderFocusHeatmap() {
        const heatmapGrid = document.getElementById('focusHeatmap');
        if (!heatmapGrid || !this.timelineData.focusHeatmap) return;
        
        const heatmapData = this.timelineData.focusHeatmap;
        heatmapGrid.innerHTML = '';
        
        // Find max focus intensity for scaling
        const maxIntensity = Math.max(...heatmapData.map(hour => hour.focusIntensity));
        
        heatmapData.forEach((hourData, hour) => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            
            // Calculate intensity level (0-4)
            const intensity = maxIntensity > 0 ? Math.floor((hourData.focusIntensity / maxIntensity) * 4) : 0;
            const colors = [
                'rgba(227, 242, 253, 0.3)',
                'rgba(144, 202, 249, 0.5)',
                'rgba(66, 165, 245, 0.7)',
                'rgba(30, 136, 229, 0.8)',
                'rgba(21, 101, 192, 1)'
            ];
            
            cell.style.backgroundColor = colors[intensity];
            cell.textContent = hour.toString().padStart(2, '0');
            cell.title = `${this.formatHour(hour)}: ${Math.round(hourData.focusIntensity * 100)}% focus intensity`;
            
            heatmapGrid.appendChild(cell);
        });
    }

    updateTimelineStats() {
        if (!this.timelineData.insights) return;
        
        const insights = this.timelineData.insights;
        
        // Update timeline stats
        const peakHourEl = document.getElementById('peakHour');
        const avgSessionEl = document.getElementById('avgSession');
        const focusScoreEl = document.getElementById('focusScore');
        
        if (peakHourEl) peakHourEl.textContent = insights.peakHour;
        if (avgSessionEl) avgSessionEl.textContent = `${insights.avgSession} min`;
        if (focusScoreEl) focusScoreEl.textContent = `${insights.focusScore}%`;
        
        // Update activity insights
        const morningFocusEl = document.getElementById('morningFocus');
        const afternoonFocusEl = document.getElementById('afternoonFocus');
        const eveningFocusEl = document.getElementById('eveningFocus');
        
        if (morningFocusEl) morningFocusEl.textContent = `${insights.morningEfficiency}%`;
        if (afternoonFocusEl) afternoonFocusEl.textContent = `${insights.afternoonEfficiency}%`;
        if (eveningFocusEl) eveningFocusEl.textContent = `${insights.eveningEfficiency}%`;
        
        // Update insight details
        this.updateInsightDetails(insights);
    }

    updateInsightDetails(insights) {
        const morningDetailEl = document.getElementById('morningDetail');
        const afternoonDetailEl = document.getElementById('afternoonDetail');
        const eveningDetailEl = document.getElementById('eveningDetail');
        const weeklyDetailEl = document.getElementById('weeklyDetail');
        
        if (morningDetailEl) {
            if (insights.morningEfficiency > 70) {
                morningDetailEl.textContent = 'Excellent morning focus!';
            } else if (insights.morningEfficiency > 50) {
                morningDetailEl.textContent = 'Good morning productivity';
            } else {
                morningDetailEl.textContent = 'Room for morning improvement';
            }
        }
        
        if (afternoonDetailEl) {
            if (insights.afternoonEfficiency > 60) {
                afternoonDetailEl.textContent = 'Strong afternoon performance';
            } else if (insights.afternoonEfficiency > 40) {
                afternoonDetailEl.textContent = 'Typical afternoon dip';
            } else {
                afternoonDetailEl.textContent = 'Consider afternoon breaks';
            }
        }
        
        if (eveningDetailEl) {
            if (insights.eveningEfficiency > 65) {
                eveningDetailEl.textContent = 'Peak evening focus time';
            } else if (insights.eveningEfficiency > 45) {
                eveningDetailEl.textContent = 'Moderate evening activity';
            } else {
                eveningDetailEl.textContent = 'Light evening browsing';
            }
        }
        
        if (weeklyDetailEl) {
            weeklyDetailEl.textContent = `${insights.focusScore}% overall focus efficiency`;
        }
    }

    extractQueryConcepts(query) {
        const concepts = [];
        
        // Education-related concepts
        const educationTerms = ['college', 'course', 'university', 'class', 'school', 'academic', 'education', 'learning', 'study', 'lesson', 'tutorial'];
        const educationWeight = educationTerms.reduce((weight, term) => {
            return query.includes(term) ? weight + 1 : weight;
        }, 0);
        if (educationWeight > 0) {
            concepts.push({ type: 'education', weight: educationWeight });
        }
        
        // Technology-related concepts
        const techTerms = ['code', 'programming', 'development', 'software', 'tech', 'computer', 'algorithm', 'github', 'api'];
        const techWeight = techTerms.reduce((weight, term) => {
            return query.includes(term) ? weight + 1 : weight;
        }, 0);
        if (techWeight > 0) {
            concepts.push({ type: 'technology', weight: techWeight });
        }
        
        // Extension-related concepts
        const extensionTerms = ['extension', 'plugin', 'addon', 'browser', 'chrome', 'firefox'];
        const extensionWeight = extensionTerms.reduce((weight, term) => {
            return query.includes(term) ? weight + 1 : weight;
        }, 0);
        if (extensionWeight > 0) {
            concepts.push({ type: 'extension', weight: extensionWeight });
        }
        
        return concepts;
    }

    calculateTextRelevance(query, text) {
        if (!text) return 0;
        
        const queryWords = query.split(/\s+/).filter(word => word.length > 2);
        let score = 0;
        
        queryWords.forEach(word => {
            if (text.includes(word)) {
                score += 1;
                // Boost for exact phrase matches
                if (text.includes(query)) {
                    score += 2;
                }
            }
        });
        
        return Math.min(score, 10); // Cap the score
    }

    calculateUrlRelevance(query, url) {
        if (!url) return 0;
        
        let score = 0;
        const urlLower = url.toLowerCase();
        const queryWords = query.split(/\s+/).filter(word => word.length > 2);
        
        // 1. Direct keyword matching in URL
        queryWords.forEach(word => {
            if (urlLower.includes(word)) {
                score += 3; // URLs are highly specific and relevant
            }
        });
        
        // 2. Semantic URL analysis - extract meaningful parts
        const urlParts = this.extractUrlSemantics(urlLower);
        
        // 3. Educational URL patterns
        if (this.containsEducationalTerms(query)) {
            if (urlParts.domain.includes('edu') || 
                urlParts.domain.includes('course') ||
                urlParts.domain.includes('learn') ||
                urlParts.path.includes('course') ||
                urlParts.path.includes('lesson') ||
                urlParts.path.includes('tutorial') ||
                urlParts.subdomain.includes('course')) {
                score += 8;
            }
            
            // Specific educational platforms
            const educationalDomains = [
                'coursera', 'udemy', 'edx', 'khanacademy', 'skillshare',
                'pluralsight', 'lynda', 'codecademy', 'freecodecamp',
                'fast.ai', 'deeplearning.ai', 'brilliant', 'masterclass'
            ];
            
            if (educationalDomains.some(domain => urlParts.domain.includes(domain))) {
                score += 10;
            }
        }
        
        // 4. Technology/Development URL patterns
        if (this.containsTechTerms(query)) {
            if (urlParts.domain.includes('github') ||
                urlParts.domain.includes('stackoverflow') ||
                urlParts.domain.includes('dev') ||
                urlParts.path.includes('api') ||
                urlParts.path.includes('docs') ||
                urlParts.path.includes('tutorial')) {
                score += 8;
            }
        }
        
        // 5. Extension-specific URL patterns
        if (query.includes('extension') || query.includes('plugin')) {
            if (urlLower.includes('extension') || 
                urlLower.includes('chrome.google.com') ||
                urlLower.includes('addons.mozilla.org') ||
                urlLower.includes('marketplace.visualstudio.com')) {
                score += 10;
            }
        }
        
        // 6. Research/Academic URL patterns
        if (this.containsResearchTerms(query)) {
            if (urlParts.domain.includes('scholar') ||
                urlParts.domain.includes('arxiv') ||
                urlParts.domain.includes('pubmed') ||
                urlParts.domain.includes('jstor') ||
                urlParts.domain.includes('researchgate')) {
                score += 8;
            }
        }
        
        // 7. Gaming URL patterns (for gaming queries)
        if (this.containsGamingTerms(query)) {
            if (urlParts.domain.includes('wiki') && 
                (urlParts.domain.includes('game') || 
                 urlParts.domain.includes('runescape') ||
                 urlParts.domain.includes('minecraft'))) {
                score += 8;
            }
        }
        
        return score;
    }

    extractUrlSemantics(url) {
        try {
            const urlObj = new URL(url);
            const pathParts = urlObj.pathname.split('/').filter(part => part.length > 0);
            const subdomainParts = urlObj.hostname.split('.');
            
            return {
                domain: urlObj.hostname,
                subdomain: subdomainParts.length > 2 ? subdomainParts[0] : '',
                path: urlObj.pathname,
                pathParts: pathParts,
                query: urlObj.search,
                fragment: urlObj.hash
            };
        } catch (error) {
            // Fallback for invalid URLs
            return {
                domain: url,
                subdomain: '',
                path: '',
                pathParts: [],
                query: '',
                fragment: ''
            };
        }
    }

    containsEducationalTerms(query) {
        const educationalTerms = [
            'education', 'educational', 'learn', 'learning', 'course', 'courses',
            'tutorial', 'tutorials', 'lesson', 'lessons', 'study', 'studying',
            'class', 'classes', 'school', 'university', 'college', 'academic',
            'teach', 'teaching', 'training', 'certification', 'degree'
        ];
        return educationalTerms.some(term => query.includes(term));
    }

    containsResearchTerms(query) {
        const researchTerms = [
            'research', 'paper', 'papers', 'study', 'studies', 'analysis',
            'investigation', 'academic', 'scholar', 'scholarly', 'journal',
            'publication', 'thesis', 'dissertation', 'article', 'scientific'
        ];
        return researchTerms.some(term => query.includes(term));
    }

    containsGamingTerms(query) {
        const gamingTerms = [
            'game', 'gaming', 'play', 'player', 'strategy', 'guide',
            'walkthrough', 'tips', 'tricks', 'build', 'character',
            'level', 'quest', 'achievement', 'boss', 'item'
        ];
        return gamingTerms.some(term => query.includes(term));
    }

    isEducationContent(item) {
        const educationIndicators = [
            item.domain?.includes('edu'),
            item.domain?.includes('coursera'),
            item.domain?.includes('udemy'),
            item.domain?.includes('khan'),
            item.pageType === 'coursera',
            item.pageType === 'udemy',
            item.contentSummary?.toLowerCase().includes('course'),
            item.title?.toLowerCase().includes('course'),
            item.title?.toLowerCase().includes('class')
        ];
        
        return educationIndicators.some(indicator => indicator);
    }

    isTechContent(item) {
        const techIndicators = [
            item.domain?.includes('github'),
            item.domain?.includes('stackoverflow'),
            item.pageType === 'github',
            item.pageType === 'stackoverflow',
            item.hasCode,
            item.contentSummary?.toLowerCase().includes('code'),
            item.contentSummary?.toLowerCase().includes('programming')
        ];
        
        return techIndicators.some(indicator => indicator);
    }

    isExtensionContent(item) {
        const extensionIndicators = [
            item.url?.toLowerCase().includes('extension'),
            item.title?.toLowerCase().includes('extension'),
            item.contentSummary?.toLowerCase().includes('extension'),
            item.domain?.includes('chrome.google.com'),
            item.domain?.includes('addons.mozilla.org')
        ];
        
        return extensionIndicators.some(indicator => indicator);
    }

    getPageTypeRelevance(pageType, query) {
        const pageTypeScores = {
            'youtube_video': query.includes('video') ? 5 : 0,
            'github': this.containsTechTerms(query) ? 5 : 0,
            'stackoverflow': this.containsTechTerms(query) ? 5 : 0,
            'coursera': query.includes('course') || query.includes('education') ? 5 : 0,
            'wikipedia': query.includes('information') || query.includes('research') ? 3 : 0
        };
        
        return pageTypeScores[pageType] || 0;
    }

    containsTechTerms(query) {
        const techTerms = ['code', 'programming', 'development', 'software', 'tech', 'computer', 'algorithm'];
        return techTerms.some(term => query.includes(term));
    }

    getFilteredWebsites() {
        // Get current filter context
        let sourceData = this.rawData;
        
        // Filter by selected category/subcategory
        if (this.state.selectedSubcategory) {
            sourceData = sourceData.filter(item => {
                const category = item.category || this.classifyDomain(item.domain || item.url);
                const subcategory = item.subcategory || this.generateIntelligentSubcategory(category, item);
                return category === this.state.selectedCategory && subcategory === this.state.selectedSubcategory;
            });
        } else if (this.state.selectedCategory) {
            sourceData = sourceData.filter(item => {
                const category = item.category || this.classifyDomain(item.domain || item.url);
                return category === this.state.selectedCategory;
            });
        }

        return sourceData;
    }

    updateWebsiteList(filters = {}) {
        const websiteList = document.getElementById('websiteList');
        const listHeader = document.getElementById('listHeader');
        
        if (!websiteList || !listHeader) return;

        // Update header based on selection
        let headerText = '📋 All Websites';
        if (this.state.selectedSubcategory) {
            headerText = `📋 ${this.state.selectedCategory} - ${this.state.selectedSubcategory}`;
        } else if (this.state.selectedCategory) {
            headerText = `📋 ${this.state.selectedCategory} Websites`;
        }
        listHeader.textContent = headerText;

        // Get filtered data
        let websites = this.getFilteredWebsites();

        // Apply duration filter
        if (filters.minDuration) {
            websites = websites.filter(item => 
                (item.activeTime || 0) >= filters.minDuration
            );
        }

        // Apply time range filter (would need timestamp data)
        if (filters.timeRange) {
            // This would filter by time of day when we have proper timestamp data
        }

        // Generate website items
        if (websites.length === 0) {
            websiteList.innerHTML = '<div class="loading-websites">No websites found matching the current filters.</div>';
            return;
        }

        const websiteItems = websites.map(item => {
            const nickname = this.generateWebsiteNickname(item);
            const domain = item.domain || this.extractDomain(item.url);
            
            return `
                <div class="website-item">
                    <div class="website-nickname">${nickname}</div>
                    <a href="${item.url}" class="website-link" target="_blank">${domain}</a>
                </div>
            `;
        }).join('');

        websiteList.innerHTML = websiteItems;
    }

    generateWebsiteNickname(item) {
        const domain = item.domain || this.extractDomain(item.url);
        const title = item.title || '';
        
        // Generate meaningful nicknames based on domain and content
        const nicknameMap = {
            'github.com': '💻 Code Repository',
            'stackoverflow.com': '❓ Programming Q&A',
            'youtube.com': '🎥 Video Content',
            'twitter.com': '🐦 Social Updates',
            'linkedin.com': '💼 Professional Network',
            'google.com': '🔍 Search Engine',
            'wikipedia.org': '📚 Encyclopedia',
            'medium.com': '📖 Article Reading',
            'netflix.com': '🎬 Streaming Video',
            'spotify.com': '🎵 Music Streaming'
        };

        // Check for specific domain nicknames
        for (const [domainKey, nickname] of Object.entries(nicknameMap)) {
            if (domain.includes(domainKey)) {
                return nickname;
            }
        }

        // Generate based on category
        const category = item.category || this.classifyDomain(domain);
        const categoryIcons = {
            'Social Media': '💬',
            'Gaming': '🎮',
            'Work/Professional': '💼',
            'Entertainment': '🎭',
            'Shopping': '🛒',
            'Learning/Tutorials': '📚',
            'News/Information': '📰',
            'AI/Development Tools': '⚙️',
            'Health/Wellness': '🏥',
            'Finance': '💰',
            'Travel': '✈️',
            'Technology': '🔧'
        };

        const icon = categoryIcons[category] || '🌐';
        const shortTitle = title.length > 30 ? title.substring(0, 30) + '...' : title;
        return `${icon} ${shortTitle || domain}`;
    }

    displaySearchResults(results, query) {
        const websiteList = document.getElementById('websiteList');
        const listHeader = document.getElementById('listHeader');
        
        if (!websiteList || !listHeader) return;

        listHeader.textContent = `🔍 Search Results for "${query}"`;
        
        if (results.length === 0) {
            websiteList.innerHTML = '<div class="loading-websites">No results found for your search query.</div>';
            return;
        }

        const searchItems = results.map(item => {
            const nickname = this.generateWebsiteNickname(item);
            const domain = item.domain || this.extractDomain(item.url);
            
            return `
                <div class="website-item">
                    <div class="website-nickname">${nickname}</div>
                    <a href="${item.url}" class="website-link" target="_blank">${domain}</a>
                </div>
            `;
        }).join('');

        websiteList.innerHTML = searchItems;
    }

    async checkBackendStatus() {
        try {
            const response = await fetch('http://localhost:5000/api/v1/health');
            this.state.isBackendAvailable = response.ok;
            this.state.aiMode = 'backend';
            console.log('Backend available:', this.state.isBackendAvailable);
        } catch (error) {
            console.log('Backend not available, checking built-in AI...');
            this.state.isBackendAvailable = false;
            
            // Check for Chrome built-in AI
            if (typeof window.ai !== 'undefined') {
                this.state.aiMode = 'chrome-ai';
                console.log('Chrome built-in AI available');
            } else {
                this.state.aiMode = 'fallback';
                this.showFallbackNotification();
                console.log('Using fallback mode');
            }
        }
        
        // Update power button color based on backend status
        const powerBtn = document.getElementById('powerBtn');
        if (powerBtn) {
            powerBtn.style.color = this.state.isBackendAvailable ? '#4CAF50' : '#888';
        }
        
        this.updateResourceMonitor();
        this.loadData();
    }

    async loadMockData() {
        console.log('Loading mock data...');
        const mockData = this.getMockData();
        this.rawData = mockData;
        this.processData(mockData);
        this.renderCharts();
        this.updateInsights();
        this.updateWebsiteList(); // Initialize website list
        
        // Generate and render timeline data
        this.generateTimelineData();
        this.renderHourlyChart(); // Initialize with hourly view
        
        this.hideLoading();
    }

    showFallbackNotification() {
        const notification = document.getElementById('fallbackNotification');
        if (notification) {
            notification.style.display = 'block';
        }
    }

    async loadData() {
        console.log('Loading data for period:', this.state.currentPeriod);
        
        this.showLoading();
        
        try {
            let data;
            
            if (typeof chrome !== 'undefined' && chrome.runtime) {
                // Extension context - get real data
                data = await this.getExtensionData();
            } else {
                // Standalone context - use mock data
                data = this.getMockData();
            }
            
            this.rawData = data;
            this.processData(data);
            this.renderCharts();
            this.updateInsights();
            this.updateWebsiteList(); // Initialize website list
            
            // Generate and render timeline data
            this.generateTimelineData();
            this.renderHourlyChart(); // Initialize with hourly view
            
            this.hideLoading();
            
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError();
        }
    }

    async getExtensionData() {
        try {
            console.log(`Loading data in ${this.state.dataMode} mode`);
            
            // Get date range
            const { startDate, endDate } = this.getDateRange();
            
            // Get current tabs
            const currentTabsResponse = await chrome.runtime.sendMessage({
                action: 'getCurrentTabs'
            });
            
            // Get historical data from Chrome API
            const historicalResponse = await chrome.runtime.sendMessage({
                action: 'getHistoricalData',
                startDate: startDate.toISOString(),
                endDate: endDate.toISOString()
            });

            // Combine basic data
            const currentData = currentTabsResponse.success ? currentTabsResponse.data : [];
            const historicalData = historicalResponse.success ? historicalResponse.data : [];
            
            if (this.state.dataMode === 'chrome-history') {
                // Chrome History Only Mode - basic data with simple categorization
                console.log('Using chrome-history mode - basic data only');
                console.log(`Retrieved ${currentData.length} current tabs, ${historicalData.length} historical entries`);
                
                // Validate IndexedDB data access
                if (historicalData.length === 0) {
                    console.warn('No historical data from IndexedDB - this may indicate IndexedDB access issues');
                }
                
                return [...currentData, ...this.addBasicCategorizationToData(historicalData)];
            
            } else if (this.state.dataMode === 'fused') {
                // Fused Mode - enhanced data with AI processing
                console.log('Using fused mode - enhanced data with AI insights');
                
                // Get enhanced engagement data from our database
                const enhancedDataResponse = await chrome.runtime.sendMessage({
                    action: 'getEnhancedData',
                    startDate: startDate.toISOString(),
                    endDate: endDate.toISOString()
                });
                
                const enhancedData = enhancedDataResponse.success ? enhancedDataResponse.data : [];
                
                // Merge enhanced data with historical data
                const enrichedHistoricalData = this.mergeEnhancedData(historicalData, enhancedData);
                
                return [...currentData, ...enrichedHistoricalData];
            }
            
        } catch (error) {
            console.error('Error getting extension data:', error);
            throw error;
        }
    }

    addBasicCategorizationToData(historicalData) {
        // Simple domain-based categorization for chrome-history mode
        return historicalData.map(item => {
            const domain = item.domain || this.extractDomain(item.url);
            const url = item.url.toLowerCase();
            const title = (item.title || '').toLowerCase();
            
            // Basic categorization based on domain patterns and URL analysis
            let category = 'Other';
            
            if (domain.includes('youtube.com') || domain.includes('netflix.com') || domain.includes('twitch.tv') || 
                domain.includes('hulu.com') || url.includes('video') || url.includes('watch')) {
                category = 'Entertainment';
            } else if (domain.includes('github.com') || domain.includes('stackoverflow.com') || 
                      domain.includes('docs.') || url.includes('documentation') || url.includes('api')) {
                category = 'AI/Development Tools';
            } else if (domain.includes('coursera.org') || domain.includes('udemy.com') || domain.includes('edx.org') ||
                      domain.includes('khanacademy.org') || url.includes('tutorial') || url.includes('course')) {
                category = 'Learning/Tutorials';
            } else if (domain.includes('facebook.com') || domain.includes('twitter.com') || domain.includes('instagram.com') ||
                      domain.includes('linkedin.com') || domain.includes('reddit.com') || domain.includes('discord.com')) {
                category = 'Social Media';
            } else if (domain.includes('amazon.com') || domain.includes('ebay.com') || domain.includes('shop') ||
                      url.includes('cart') || url.includes('buy') || url.includes('product')) {
                category = 'Shopping';
            } else if (domain.includes('news.') || domain.includes('cnn.com') || domain.includes('bbc.com') ||
                      domain.includes('reuters.com') || domain.includes('techcrunch.com')) {
                category = 'News/Information';
            } else if (domain.includes('gmail.com') || domain.includes('outlook.com') || domain.includes('office.com') ||
                      domain.includes('slack.com') || domain.includes('teams.microsoft.com')) {
                category = 'Work/Professional';
            }
            
            return {
                ...item,
                category,
                domain,
                description: `Basic categorization: ${category}`,
                pageType: 'webpage'
            };
        });
    }

    mergeEnhancedData(historicalData, enhancedData) {
        // Create a map of enhanced data by URL for quick lookup
        const enhancedMap = new Map();
        enhancedData.forEach(item => {
            enhancedMap.set(item.url, item);
        });

        // Enrich historical data with enhanced metadata
        return historicalData.map(item => {
            const enhanced = enhancedMap.get(item.url);
            if (enhanced) {
                return {
                    ...item,
                    description: enhanced.description,
                    keywords: enhanced.keywords,
                    pageType: enhanced.pageType,
                    contentSummary: enhanced.contentSummary,
                    hasVideo: enhanced.hasVideo,
                    hasAudio: enhanced.hasAudio,
                    hasImages: enhanced.hasImages,
                    hasCode: enhanced.hasCode,
                    textLength: enhanced.textLength,
                    readingTime: enhanced.readingTime,
                    activeTime: enhanced.activeTime
                };
            }
            return item;
        });
    }

    getMockData() {
        // Generate realistic mock data with semantically meaningful URLs
        const mockEntries = [
            // Educational content
            { 
                url: 'https://course.fast.ai/Lessons/lesson1.html', 
                title: 'Practical Deep Learning for Coders - Lesson 1',
                domain: 'course.fast.ai',
                category: 'Learning/Tutorials',
                description: 'Learn practical deep learning with fast.ai',
                contentSummary: 'Practical Deep Learning for Coders: Introduction to deep learning concepts using the fast.ai library. This lesson covers the basics of neural networks and how to build your first image classifier.',
                pageType: 'course',
                keywords: ['deep learning', 'machine learning', 'AI', 'course', 'tutorial'],
                hasVideo: true,
                hasCode: true
            },
            { 
                url: 'https://www.coursera.org/learn/machine-learning', 
                title: 'Machine Learning Course by Andrew Ng',
                domain: 'coursera.org',
                category: 'Learning/Tutorials',
                description: 'Stanford Machine Learning Course',
                contentSummary: 'Machine Learning Course: Comprehensive introduction to machine learning, datamining, and statistical pattern recognition taught by Andrew Ng from Stanford.',
                pageType: 'coursera',
                keywords: ['machine learning', 'stanford', 'andrew ng', 'course'],
                hasVideo: true
            },
            { 
                url: 'https://github.com/microsoft/vscode-extension-samples', 
                title: 'VS Code Extension Samples',
                domain: 'github.com',
                category: 'AI/Development Tools',
                description: 'Sample extensions for VS Code',
                contentSummary: 'GitHub Repository: VS Code Extension Samples. This repository contains sample code and tutorials for building Visual Studio Code extensions.',
                pageType: 'github',
                keywords: ['vscode', 'extension', 'samples', 'microsoft'],
                hasCode: true
            },
            { 
                url: 'https://chrome.google.com/webstore/detail/history-plus/abcdef123456', 
                title: 'History Plus - Chrome Extension',
                domain: 'chrome.google.com',
                category: 'AI/Development Tools',
                description: 'AI-powered browsing history extension',
                contentSummary: 'Chrome Web Store: History Plus Extension. An AI-powered extension that provides intelligent analysis of your browsing history with semantic search capabilities.',
                pageType: 'chrome_extension',
                keywords: ['chrome', 'extension', 'history', 'AI'],
                hasImages: true
            },
            {
                url: 'https://runescape.wiki/w/Ironman_Mode',
                title: 'Ironman Mode - Old School RuneScape Wiki',
                domain: 'runescape.wiki',
                category: 'Gaming',
                description: 'Ironman game mode guide',
                contentSummary: 'RuneScape Wiki: Ironman Mode. Ironman Mode is an account type that restricts players from most forms of player interaction that could be seen as cheating.',
                pageType: 'runescape_wiki',
                keywords: ['runescape', 'ironman', 'game mode', 'guide']
            },
            {
                url: 'https://stackoverflow.com/questions/12345/how-to-build-chrome-extension',
                title: 'How to build a Chrome extension?',
                domain: 'stackoverflow.com',
                category: 'AI/Development Tools',
                description: 'Chrome extension development question',
                contentSummary: 'Stack Overflow: How to build a Chrome extension? Question about developing Chrome extensions with manifest v3, content scripts, and background workers.',
                pageType: 'stackoverflow',
                keywords: ['chrome', 'extension', 'development', 'javascript'],
                hasCode: true
            },
            {
                url: 'https://scholar.google.com/scholar?q=machine+learning+research',
                title: 'Machine Learning Research Papers',
                domain: 'scholar.google.com',
                category: 'Academic Research',
                description: 'Academic papers on machine learning',
                contentSummary: 'Google Scholar: Machine Learning Research. Collection of peer-reviewed academic papers and citations related to machine learning research.',
                pageType: 'scholar',
                keywords: ['research', 'machine learning', 'academic', 'papers']
            },
            {
                url: 'https://www.youtube.com/watch?v=abc123',
                title: 'Introduction to Web Extensions - Tutorial',
                domain: 'youtube.com',
                category: 'Learning/Tutorials',
                description: 'Web extension development tutorial',
                contentSummary: 'YouTube Video: Introduction to Web Extensions - Tutorial. Learn how to build browser extensions from scratch with this comprehensive tutorial covering manifest files, content scripts, and API usage.',
                pageType: 'youtube_video',
                keywords: ['extension', 'tutorial', 'web development', 'javascript'],
                hasVideo: true
            },
            // Test entries for extension search
            {
                url: 'chrome-extension://abcdef123456/popup.html',
                title: 'Untitled Extension1',
                domain: 'chrome-extension',
                category: 'AI/Development Tools',
                description: 'Chrome extension popup page',
                contentSummary: 'Chrome Extension: Untitled Extension1. A browser extension popup page with various controls and settings for managing extension functionality.',
                pageType: 'chrome_extension',
                keywords: ['chrome', 'extension', 'popup', 'browser'],
                hasImages: true
            },
            {
                url: 'chrome-extension://fedcba654321/options.html',
                title: 'Extension2 Options',
                domain: 'chrome-extension',
                category: 'AI/Development Tools',
                description: 'Extension options and settings page',
                contentSummary: 'Chrome Extension: Extension2 Options. Configuration page for a browser extension allowing users to customize settings and preferences.',
                pageType: 'chrome_extension',
                keywords: ['extension', 'options', 'settings', 'configuration'],
                hasImages: true
            },
            {
                url: 'file:///C:/Users/dev/Documents/untitled-extension-project/index.html',
                title: 'Untitled Extension Project',
                domain: 'local-file',
                category: 'AI/Development Tools',
                description: 'Local extension development project',
                contentSummary: 'Local File: Untitled Extension Project. A local development project for creating a browser extension, containing HTML, CSS, and JavaScript files.',
                pageType: 'local_development',
                keywords: ['extension', 'development', 'project', 'local'],
                hasCode: true
            }
        ];

        // Add some regular mock data to fill out the dataset
        const categories = Object.keys(this.FRAMEWORK_COLORS);
        const regularSites = [
            'google.com', 'twitter.com', 'facebook.com', 'reddit.com',
            'amazon.com', 'netflix.com', 'spotify.com', 'discord.com'
        ];
        
        const now = new Date();
        
        // Add semantic entries
        const mockData = mockEntries.map((entry, index) => {
            const timestamp = new Date(now.getTime() - (index * 60 * 60 * 1000)); // Space them out by hours
            const timeSpentOptions = [120, 180, 300, 420, 600, 900, 1200, 1800];
            const timeSpent = timeSpentOptions[Math.floor(Math.random() * timeSpentOptions.length)];
            
            // Generate realistic active time based on content type
            let activeTime;
            if (entry.hasVideo) {
                // Video content typically has high engagement
                activeTime = Math.floor(timeSpent * (0.7 + Math.random() * 0.25)); // 70-95% of time spent
            } else if (entry.hasCode || entry.pageType === 'github') {
                // Code/development content has moderate engagement
                activeTime = Math.floor(timeSpent * (0.5 + Math.random() * 0.3)); // 50-80% of time spent
            } else if (entry.category === 'Learning/Tutorials') {
                // Educational content has good engagement
                activeTime = Math.floor(timeSpent * (0.6 + Math.random() * 0.25)); // 60-85% of time spent
            } else {
                // General content has variable engagement
                activeTime = Math.floor(timeSpent * (0.3 + Math.random() * 0.4)); // 30-70% of time spent
            }
            
            return {
                ...entry,
                subcategory: this.generateIntelligentSubcategory(entry.category, entry),
                timestamp: timestamp.getTime(),
                timeSpent: timeSpent,
                activeTime: activeTime, // Add precise active time
                focusTime: Math.floor(activeTime * 1.1), // Focus time is slightly higher than active time
                confidence: Math.random() * 0.3 + 0.7, // High confidence for curated data
                textLength: Math.floor(Math.random() * 3000) + 1000,
                readingTime: Math.floor(Math.random() * 8) + 2
            };
        });

        // Add some regular entries
        for (let i = 0; i < 25; i++) {
            const category = categories[Math.floor(Math.random() * categories.length)];
            const site = regularSites[Math.floor(Math.random() * regularSites.length)];
            const timestamp = new Date(now.getTime() - Math.random() * 24 * 60 * 60 * 1000);
            
            const timeSpentOptions = [30, 45, 60, 90, 120, 180, 240, 300, 420, 600];
            const timeSpent = timeSpentOptions[Math.floor(Math.random() * timeSpentOptions.length)];
            
            // Generate realistic active time for regular sites
            const engagementFactor = 0.2 + Math.random() * 0.5; // 20-70% engagement
            const activeTime = Math.floor(timeSpent * engagementFactor);
            
            mockData.push({
                url: `https://${site}`,
                title: `Page on ${site}`,
                domain: site,
                category,
                subcategory: `${category} - Subcategory ${Math.floor(Math.random() * 3) + 1}`,
                timestamp: timestamp.getTime(),
                timeSpent: timeSpent,
                activeTime: activeTime, // Add active time for regular entries
                focusTime: Math.floor(activeTime * 1.1),
                confidence: Math.random() * 0.4 + 0.6,
                
                // Enhanced mock data
                description: `This is a page about ${category.toLowerCase()} on ${site}`,
                keywords: [category.toLowerCase(), site.split('.')[0], 'web', 'content'],
                pageType: site.includes('youtube') ? 'youtube_video' : 'webpage',
                contentSummary: `${category} content from ${site}: A page about ${category.toLowerCase()} topics and related information.`,
                hasVideo: site.includes('youtube') || site.includes('netflix'),
                hasCode: site.includes('github') || site.includes('stackoverflow'),
                textLength: Math.floor(Math.random() * 5000) + 500,
                readingTime: Math.floor(Math.random() * 10) + 1
            });
        }
        
        return mockData;
    }

    processData(data) {
        console.log('Processing data:', data.length, 'items');
        
        if (!Array.isArray(data)) {
            console.error('Data is not an array:', data);
            data = [];
        }
        
        // Process by categories
        const categoryData = {};
        const subcategoryData = {};
        
        data.forEach(item => {
            const category = item.category || this.classifyDomain(item.domain || item.url);
            const subcategory = item.subcategory || this.generateIntelligentSubcategory(category, item);
            
            // Use activeTime if available, fallback to timeSpent, then generate reasonable default
            const activeTime = item.activeTime || item.timeSpent || Math.floor(Math.random() * 300) + 60;
            
            // NEW: Filter out items with no active time when focus-only mode is enabled
            if (this.state.focusOnlyMode && (!item.activeTime || item.activeTime === 0)) {
                console.log('Filtering out item with no active time:', item.url || item.domain);
                return; // Skip this item
            }
            
            // Main categories
            if (!categoryData[category]) {
                categoryData[category] = {
                    name: category,
                    value: 0,
                    count: 0,
                    sites: new Set(),
                    subcategories: {}
                };
            }
            
            categoryData[category].value += activeTime;
            categoryData[category].count += 1;
            categoryData[category].sites.add(item.domain || this.extractDomain(item.url));
            
            // Subcategories
            if (!categoryData[category].subcategories[subcategory]) {
                categoryData[category].subcategories[subcategory] = {
                    name: subcategory,
                    value: 0,
                    count: 0,
                    sites: new Set()
                };
            }
            
            categoryData[category].subcategories[subcategory].value += activeTime;
            categoryData[category].subcategories[subcategory].count += 1;
            categoryData[category].subcategories[subcategory].sites.add(item.domain || this.extractDomain(item.url));
        });
        
        // Convert sets to arrays and sort
        Object.values(categoryData).forEach(cat => {
            cat.sites = Array.from(cat.sites);
            Object.values(cat.subcategories).forEach(sub => {
                sub.sites = Array.from(sub.sites);
            });
        });
        
        this.processedData = {
            categories: categoryData,
            totalTime: Object.values(categoryData).reduce((sum, cat) => sum + cat.value, 0),
            totalSites: new Set(data.map(item => item.domain || this.extractDomain(item.url))).size,
            totalSessions: data.length
        };
        
        console.log('Processed data:', this.processedData);
    }

    classifyDomain(domain) {
        if (!domain) return 'Web Browsing';
        
        const classifications = {
            'Social Media': ['facebook', 'twitter', 'instagram', 'linkedin', 'snapchat', 'tiktok', 'reddit', 'discord'],
            'Gaming': ['steam', 'twitch', 'ign', 'gamespot', 'minecraft', 'roblox', 'epic', 'origin'],
            'Academic Research': ['scholar.google', 'jstor', 'pubmed', 'arxiv', 'researchgate', 'academia', 'mendeley'],
            'Work/Professional': ['slack', 'zoom', 'asana', 'trello', 'notion', 'teams', 'office', 'docs.google'],
            'Shopping': ['amazon', 'ebay', 'etsy', 'walmart', 'target', 'shopify', 'alibaba', 'wish'],
            'Entertainment': ['youtube', 'netflix', 'hulu', 'spotify', 'disney', 'twitch', 'crunchyroll', 'prime'],
            'Learning/Tutorials': ['coursera', 'udemy', 'khan', 'codecademy', 'duolingo', 'skillshare', 'edx'],
            'News/Information': ['cnn', 'bbc', 'reuters', 'wikipedia', 'medium', 'substack', 'npr', 'guardian'],
            'AI/Development Tools': ['github', 'stackoverflow', 'openai', 'huggingface', 'colab', 'replit', 'codepen'],
            'Health/Wellness': ['webmd', 'healthline', 'mayo', 'fitbit', 'myfitnesspal', 'headspace', 'calm'],
            'Finance': ['paypal', 'venmo', 'mint', 'robinhood', 'coinbase', 'banking', 'credit'],
            'Travel': ['booking', 'airbnb', 'expedia', 'uber', 'lyft', 'maps', 'tripadvisor'],
            'Technology': ['apple', 'microsoft', 'google', 'tech', 'software', 'hardware', 'android'],
            'Creative/Design': ['behance', 'dribbble', 'figma', 'canva', 'adobe', 'sketch', 'photoshop']
        };
        
        // Check for exact matches first
        for (const [category, keywords] of Object.entries(classifications)) {
            if (keywords.some(keyword => domain.toLowerCase().includes(keyword))) {
                return category;
            }
        }
        
        // Semantic analysis for unknown domains
        return this.intelligentDomainClassification(domain);
    }

    intelligentDomainClassification(domain) {
        // Extract meaningful parts from domain
        const domainParts = domain.toLowerCase().replace(/\.(com|org|net|edu|gov|io|co|ai)$/, '').split('.');
        const mainDomain = domainParts[domainParts.length - 1];
        
        // Common patterns and semantic indicators
        const semanticPatterns = {
            'Media/Content': ['blog', 'news', 'media', 'post', 'article', 'story', 'press'],
            'Business/Corporate': ['corp', 'company', 'business', 'enterprise', 'official', 'inc'],
            'Education': ['edu', 'school', 'university', 'college', 'learn', 'study', 'course'],
            'Communication': ['mail', 'chat', 'message', 'talk', 'connect', 'meet', 'call'],
            'Tools/Utilities': ['tool', 'app', 'service', 'utility', 'helper', 'calc', 'convert'],
            'Reference/Documentation': ['docs', 'wiki', 'guide', 'manual', 'help', 'reference', 'api'],
            'Marketplace/Commerce': ['shop', 'store', 'buy', 'sell', 'market', 'trade', 'deal'],
            'Creative/Art': ['art', 'design', 'creative', 'photo', 'image', 'gallery', 'portfolio']
        };
        
        // Check semantic patterns
        for (const [category, patterns] of Object.entries(semanticPatterns)) {
            if (patterns.some(pattern => mainDomain.includes(pattern) || domain.includes(pattern))) {
                return category;
            }
        }
        
        // TLD-based classification
        if (domain.includes('.edu')) return 'Education';
        if (domain.includes('.gov')) return 'Government/Public';
        if (domain.includes('.org')) return 'Organizations/Non-profit';
        if (domain.includes('.io') || domain.includes('.dev')) return 'Technology/Development';
        
        // Final fallback - categorize by domain structure
        if (mainDomain.length <= 3) return 'Web Services';
        if (mainDomain.includes('app') || mainDomain.includes('web')) return 'Web Applications';
        
        return 'Online Resources';
    }

    generateIntelligentSubcategory(category, item) {
        const domain = item.domain || this.extractDomain(item.url);
        const title = item.title || '';
        const domainName = domain.replace(/\.(com|org|net|edu|gov|io|co|ai)$/, '').split('.').pop();
        
        // PRIORITY 1: Iconic Brand Recognition - Use brand names when they're recognizable
        const iconicBrands = {
            // Google Services
            'drive.google': 'Google Drive',
            'docs.google': 'Google Docs',
            'sheets.google': 'Google Sheets', 
            'slides.google': 'Google Slides',
            'gmail': 'Gmail',
            'youtube': 'YouTube',
            'google': 'Google Search',
            
            // Microsoft Services  
            'office.com': 'Microsoft Office',
            'outlook': 'Outlook',
            'teams': 'Microsoft Teams',
            'onedrive': 'OneDrive',
            
            // Social Media Brands
            'facebook': 'Facebook',
            'instagram': 'Instagram', 
            'twitter': 'Twitter/X',
            'linkedin': 'LinkedIn',
            'tiktok': 'TikTok',
            'snapchat': 'Snapchat',
            'discord': 'Discord',
            'reddit': 'Reddit',
            'whatsapp': 'WhatsApp',
            
            // Development & Tech
            'github': 'GitHub',
            'stackoverflow': 'Stack Overflow',
            'aws.amazon': 'Amazon AWS',
            'firebase': 'Firebase',
            'vercel': 'Vercel',
            'heroku': 'Heroku',
            'netlify': 'Netlify',
            
            // Entertainment & Media
            'netflix': 'Netflix',
            'spotify': 'Spotify',
            'hulu': 'Hulu',
            'disney': 'Disney+',
            'twitch': 'Twitch',
            'amazon.com': 'Amazon',
            'prime': 'Amazon Prime',
            
            // Education & Learning
            'coursera': 'Coursera',
            'udemy': 'Udemy',
            'khanacademy': 'Khan Academy',
            'codecademy': 'Codecademy',
            'duolingo': 'Duolingo',
            'edx': 'edX',
            
            // Productivity & Work
            'notion': 'Notion',
            'slack': 'Slack',
            'zoom': 'Zoom',
            'trello': 'Trello',
            'asana': 'Asana',
            'figma': 'Figma',
            'canva': 'Canva',
            
            // News & Information
            'wikipedia': 'Wikipedia',
            'medium': 'Medium',
            'substack': 'Substack',
            
            // Shopping & E-commerce
            'ebay': 'eBay',
            'etsy': 'Etsy',
            'shopify': 'Shopify',
            'paypal': 'PayPal',
            
            // AI & Modern Tools
            'openai': 'OpenAI',
            'chatgpt': 'ChatGPT',
            'anthropic': 'Claude AI',
            'huggingface': 'Hugging Face',
            'colab': 'Google Colab',
            'replit': 'Replit'
        };
        
        // Check for iconic brand matches first
        for (const [brandPattern, brandName] of Object.entries(iconicBrands)) {
            if (domain.toLowerCase().includes(brandPattern.toLowerCase())) {
                return brandName;
            }
        }
        
        // PRIORITY 2: Category-specific brand mappings for less obvious cases
        const categoryBrandMappings = {
            'Work/Professional': {
                'office365': 'Microsoft 365',
                'gsuite': 'Google Workspace', 
                'workspace.google': 'Google Workspace',
                'sharepoint': 'SharePoint',
                'monday': 'Monday.com',
                'basecamp': 'Basecamp',
                'jira': 'Jira'
            },
            'Entertainment': {
                'paramount': 'Paramount+',
                'hbo': 'HBO Max',
                'crunchyroll': 'Crunchyroll',
                'funimation': 'Funimation',
                'soundcloud': 'SoundCloud',
                'bandcamp': 'Bandcamp'
            },
            'Shopping': {
                'walmart': 'Walmart',
                'target': 'Target',
                'bestbuy': 'Best Buy',
                'newegg': 'Newegg',
                'alibaba': 'Alibaba'
            },
            'Gaming': {
                'steam': 'Steam',
                'epicgames': 'Epic Games',
                'origin': 'EA Origin',
                'battle.net': 'Battle.net',
                'xbox': 'Xbox Live',
                'playstation': 'PlayStation'
            }
        };

        // Check category-specific brand mappings
        const categoryMappings = categoryBrandMappings[category];
        if (categoryMappings) {
            for (const [brandPattern, brandName] of Object.entries(categoryMappings)) {
                if (domain.toLowerCase().includes(brandPattern.toLowerCase())) {
                    return brandName;
                }
            }
        }
        
        // PRIORITY 3: Generic semantic patterns (only if no brand detected)
        const semanticPatterns = {
            'mail': 'Email Service',
            'maps': 'Maps & Navigation', 
            'calendar': 'Calendar App',
            'blog': 'Blog Platform',
            'wiki': 'Wiki Platform',
            'news': 'News Source',
            'shop': 'Online Store',
            'store': 'Retail Store',
            'app': 'Web Application',
            'api': 'API Service',
            'docs': 'Documentation',
            'help': 'Help Center',
            'support': 'Support Portal'
        };
        
        // Check semantic patterns
        for (const [pattern, description] of Object.entries(semanticPatterns)) {
            if (domain.includes(pattern) || title.toLowerCase().includes(pattern)) {
                return description;
            }
        }
        
        // PRIORITY 4: TLD-based classification
        if (domain.includes('.edu')) return 'Educational Institution';
        if (domain.includes('.gov')) return 'Government Service';
        if (domain.includes('.org')) return 'Organization Site';
        
        // PRIORITY 5: Fallback to meaningful subcategories based on category
        const fallbackSubcategories = {
            'Social Media': 'Social Platform',
            'Gaming': 'Gaming Platform',
            'Work/Professional': 'Professional Tool', 
            'Entertainment': 'Entertainment Platform',
            'Shopping': 'Shopping Site',
            'Learning/Tutorials': 'Learning Platform',
            'News/Information': 'Information Source',
            'AI/Development Tools': 'Development Tool',
            'Health/Wellness': 'Health Resource',
            'Finance': 'Financial Service',
            'Travel': 'Travel Service',
            'Technology': 'Tech Platform',
            'Creative/Design': 'Creative Tool',
            'Education': 'Educational Resource',
            'Communication': 'Communication Tool',
            'Tools/Utilities': 'Web Utility',
            'Reference/Documentation': 'Reference Material',
            'Media/Content': 'Content Platform',
            'Business/Corporate': 'Business Resource'
        };
        
        return fallbackSubcategories[category] || 'Web Service';
    }

    extractDomain(url) {
        try {
            return new URL(url).hostname;
        } catch {
            return url;
        }
    }

    getDateRange() {
        const now = new Date();
        let startDate, endDate;
        
        switch (this.state.currentPeriod) {
            case 'today':
                startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                endDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
                break;
            case 'week':
                const dayOfWeek = now.getDay();
                startDate = new Date(now.getTime() - dayOfWeek * 24 * 60 * 60 * 1000);
                startDate.setHours(0, 0, 0, 0);
                endDate = new Date(startDate.getTime() + 7 * 24 * 60 * 60 * 1000);
                break;
            case 'month':
                startDate = new Date(now.getFullYear(), now.getMonth(), 1);
                endDate = new Date(now.getFullYear(), now.getMonth() + 1, 1);
                break;
            case 'custom':
                const startInput = document.getElementById('startDate');
                const endInput = document.getElementById('endDate');
                startDate = startInput ? new Date(startInput.value) : new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                endDate = endInput ? new Date(endInput.value) : now;
                break;
            default:
                startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                endDate = now;
        }
        
        return { startDate, endDate };
    }

    renderCharts() {
        this.renderMainChart();
        if (this.state.selectedCategory) {
            this.renderSubChart();
        }
    }

    renderMainChart() {
        const canvas = document.getElementById('mainChart');
        const container = document.getElementById('mainChartContainer');
        if (!canvas || !container) return;
        
        // Check if Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded');
            this.showError('Chart library not loaded');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (this.mainChart) {
            this.mainChart.destroy();
        }
        
        const categories = Object.values(this.processedData.categories);
        const labels = categories.map(cat => cat.name);
        const data = categories.map(cat => cat.value);
        const colors = labels.map(label => this.FRAMEWORK_COLORS[label] || this.generateRandomColor());
        
        this.mainChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 20
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.label;
                                const value = Math.round(context.parsed / 60); // Convert to minutes
                                const percentage = ((context.parsed / this.processedData.totalTime) * 100).toFixed(1);
                                return `${label}: ${value} min active focus (${percentage}%)`;
                            }
                        }
                    }
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        const index = elements[0].index;
                        const category = labels[index];
                        this.selectCategory(category);
                    }
                }
            }
        });
        
        container.style.display = 'flex';
    }

    renderSubChart() {
        const canvas = document.getElementById('subChart');
        const container = document.getElementById('subChartContainer');
        const title = document.getElementById('subChartTitle');
        if (!canvas || !container || !this.state.selectedCategory) return;
        
        // Check if Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (this.subChart) {
            this.subChart.destroy();
        }
        
        const category = this.processedData.categories[this.state.selectedCategory];
        const subcategories = Object.values(category.subcategories);
        const labels = subcategories.map(sub => sub.name);
        const data = subcategories.map(sub => sub.value);
        const colors = labels.map(() => this.generateRandomColor());
        
        // Update title
        if (title) {
            title.textContent = `🎯 ${this.state.selectedCategory} - Sub Categories`;
        }
        
        this.subChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#333333',
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.label;
                                const value = Math.round(context.parsed / 60);
                                const percentage = ((context.parsed / category.value) * 100).toFixed(1);
                                return `${label}: ${value} min active focus (${percentage}%)`;
                            }
                        }
                    }
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        const index = elements[0].index;
                        const subcategory = labels[index];
                        this.selectSubcategory(subcategory);
                    }
                }
            }
        });
        
        container.style.display = 'flex';
    }

    selectCategory(category) {
        console.log('Selected category:', category);
        this.state.selectedCategory = category;
        this.state.selectedSubcategory = null;
        
        // Switch to split view
        const chartArea = document.getElementById('pieChartArea');
        if (chartArea) {
            chartArea.classList.add('split-view');
        }
        
        this.renderSubChart();
        this.updateInsights();
        this.updateWebsiteList(); // Update website list for selected category
        this.updateResetButton(); // Update reset button state
    }

    selectSubcategory(subcategory) {
        console.log('Selected subcategory:', subcategory);
        this.state.selectedSubcategory = subcategory;
        this.updateInsights();
        this.updateWebsiteList(); // Update website list for selected subcategory
        this.updateResetButton(); // Update reset button state
    }

    resetToMainView() {
        if (!this.state.selectedCategory) return;
        
        console.log('Resetting to main view');
        this.state.selectedCategory = null;
        this.state.selectedSubcategory = null;
        
        // Remove split view
        const chartArea = document.getElementById('pieChartArea');
        if (chartArea) {
            chartArea.classList.remove('split-view');
        }
        
        // Hide sub chart container
        const subContainer = document.getElementById('subChartContainer');
        if (subContainer) {
            subContainer.style.display = 'none';
        }
        
        if (this.subChart) {
            this.subChart.destroy();
            this.subChart = null;
        }
        
        this.updateInsights();
        this.updateWebsiteList(); // Update website list back to all websites
        this.updateResetButton(); // Update reset button state
    }

    updateResetButton() {
        const resetBtn = document.getElementById('resetBtn');
        if (!resetBtn) return;

        if (this.state.selectedCategory || this.state.selectedSubcategory) {
            resetBtn.style.display = 'block';
            resetBtn.disabled = false;
            
            // Update button text based on selection
            if (this.state.selectedSubcategory) {
                resetBtn.textContent = '← Back to Categories';
            } else if (this.state.selectedCategory) {
                resetBtn.textContent = '← Back to Categories';
            }
        } else {
            resetBtn.style.display = 'none';
            resetBtn.disabled = true;
        }
    }

    updateInsights() {
        const { categories, totalTime, totalSites, totalSessions } = this.processedData;
        
        // Update basic stats - now showing active focus time
        const totalTimeEl = document.getElementById('totalTime');
        if (totalTimeEl) {
            totalTimeEl.textContent = `${Math.round(totalTime / 60)} min active focus`;
        }
        
        const sitesCountEl = document.getElementById('sitesCount');
        if (sitesCountEl) {
            sitesCountEl.textContent = totalSites;
        }
        
        const sessionCountEl = document.getElementById('sessionCount');
        if (sessionCountEl) {
            sessionCountEl.textContent = totalSessions;
        }
        
        // Top category by active focus time
        const topCategory = Object.values(categories).sort((a, b) => b.value - a.value)[0];
        const topCategoryEl = document.getElementById('topCategory');
        const topCategoryTimeEl = document.getElementById('topCategoryTime');
        
        if (topCategory && topCategoryEl && topCategoryTimeEl) {
            topCategoryEl.textContent = topCategory.name;
            topCategoryTimeEl.textContent = `${Math.round(topCategory.value / 60)} minutes active focus`;
        }
    }

    generateRandomColor() {
        const colors = [
            '#FF6B9D', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
            '#98D8C8', '#F7DC6F', '#BB8FCE', '#82E0AA', '#F8C471',
            '#AED6F1', '#F9E79F', '#D7BDE2', '#A9DFBF', '#FAD7A0'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    showLoading() {
        const loading = document.getElementById('chartLoading');
        const error = document.getElementById('errorState');
        const mainContainer = document.getElementById('mainChartContainer');
        const subContainer = document.getElementById('subChartContainer');
        
        if (loading) loading.style.display = 'flex';
        if (error) error.style.display = 'none';
        if (mainContainer) mainContainer.style.display = 'none';
        if (subContainer) subContainer.style.display = 'none';
    }

    hideLoading() {
        const loading = document.getElementById('chartLoading');
        if (loading) loading.style.display = 'none';
    }

    showError(message = 'Failed to load data') {
        const loading = document.getElementById('chartLoading');
        const error = document.getElementById('errorState');
        const mainContainer = document.getElementById('mainChartContainer');
        const subContainer = document.getElementById('subChartContainer');
        
        if (loading) loading.style.display = 'none';
        if (error) {
            error.style.display = 'flex';
            // Update error message if available
            const errorMsg = error.querySelector('.error-message');
            if (errorMsg) {
                errorMsg.textContent = message;
            }
        }
        if (mainContainer) mainContainer.style.display = 'none';
        if (subContainer) subContainer.style.display = 'none';
    }

    updateResourceMonitor() {
        // Update AI mode
        const aiModeEl = document.getElementById('aiMode');
        if (aiModeEl) {
            const modes = {
                'backend': 'Full AI',
                'chrome-ai': 'Chrome AI',
                'fallback': 'Basic',
                'detecting': 'Detecting...'
            };
            aiModeEl.textContent = modes[this.state.aiMode] || 'Unknown';
        }
        
        // Mock storage and RAM usage
        const storageEl = document.getElementById('storageUsage');
        const ramEl = document.getElementById('ramUsage');
        
        if (storageEl) {
            storageEl.textContent = '12.5 MB';
        }
        
        if (ramEl) {
            ramEl.textContent = '8.2 MB';
        }
    }

    showTroubleshootingGuide() {
        alert(`Troubleshooting Guide:

1. ✅ Check Backend Status
   - Ensure Flask server is running on localhost:5000
   - Run: python backend/app.py

2. 🔧 Install Dependencies
   - Run: pip install -r backend/requirements.txt

3. 🔄 Restart Extension
   - Go to chrome://extensions/
   - Click reload on History Plus

4. 🌐 Check Network
   - Ensure no firewall blocking localhost:5000
   - Try accessing http://localhost:5000/ directly

5. 📱 Extension Permissions
   - Verify all permissions are granted
   - Check for any console errors

Still having issues? The extension works in basic mode without the backend!`);
    }

    // Public API methods
    refreshData() {
        this.loadData();
    }

    exportData() {
        const dataStr = JSON.stringify(this.processedData, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `history-plus-data-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    async handlePowerButton() {
        console.log('Power button clicked: attempting to start/restart backend');
        
        const powerBtn = document.getElementById('powerBtn');
        if (powerBtn) {
            powerBtn.style.color = '#FFA500'; // Orange for loading
            powerBtn.disabled = true;
        }
        
        try {
            if (typeof chrome !== 'undefined' && chrome.runtime) {
                // Send message to background script to start/restart backend
                chrome.runtime.sendMessage({ action: 'powerBackend' }, (response) => {
                    console.log('Power backend response:', response);
                    // Check backend status after a delay
                    setTimeout(() => {
                        this.checkBackendStatus();
                    }, 3000);
                });
            }
        } catch (error) {
            console.error('Error triggering power backend:', error);
        } finally {
            if (powerBtn) {
                powerBtn.disabled = false;
            }
        }
    }

    handleDataModeChange(mode) {
        console.log('Data mode changed to:', mode);
        
        // Update state and save preference
        this.state.dataMode = mode;
        this.saveDataModePreference(mode);
        
        // Update status display
        const statusDiv = document.getElementById('dataModeStatus');
        if (statusDiv) {
            if (mode === 'chrome-history') {
                statusDiv.textContent = 'Chrome data only';
                statusDiv.style.color = '#333';
            } else {
                statusDiv.textContent = 'Enhanced data mode';
                statusDiv.style.color = '#333';
            }
        }
        
        // Apply feature gating based on mode
        this.applyFeatureGating(mode);
        
        // Reload data with new mode
        this.loadData();
    }

    handleFocusOnlyChange(enabled) {
        console.log('Focus-only mode changed to:', enabled);
        
        // Update state and save preference
        this.state.focusOnlyMode = enabled;
        this.saveFocusOnlyPreference(enabled);
        
        // Update UI status
        this.syncFocusOnlyUI();
        
        // Reload data with new filter
        this.loadData();
    }

    applyFeatureGating(mode) {
        console.log('Applying feature gating for mode:', mode);
        
        // Get UI elements that need to be gated
        const semanticSearchRow = document.getElementById('semanticSearch')?.parentElement?.parentElement;
        const semanticSearchInput = document.getElementById('semanticSearch');
        const searchBtn = document.getElementById('searchBtn');
        const sessionCountElement = document.getElementById('sessionCount');
        const avgSessionElement = document.getElementById('avgSession');
        const activityInsights = document.querySelector('.activity-insights');
        const insightsPanel = document.querySelector('.insights-panel');
        
        if (mode === 'chrome-history') {
            // Chrome History Only Mode - disable advanced features
            
            // Disable semantic search
            if (semanticSearchInput) {
                semanticSearchInput.disabled = true;
                semanticSearchInput.placeholder = 'Semantic search unavailable in Chrome data only mode. Switch to Enhanced data mode for AI-powered search.';
                semanticSearchInput.style.backgroundColor = '#f5f5f5';
                semanticSearchInput.style.color = '#999';
            }
            if (searchBtn) {
                searchBtn.disabled = true;
                searchBtn.style.backgroundColor = '#ccc';
                searchBtn.style.cursor = 'not-allowed';
            }
            
            // Show limited session data with disclaimers
            if (sessionCountElement) {
                sessionCountElement.textContent = 'Estimated';
                sessionCountElement.title = 'Session tracking is heuristic-based in Chrome data only mode';
            }
            if (avgSessionElement) {
                avgSessionElement.textContent = 'Estimated';
                avgSessionElement.title = 'Average session time is estimated based on basic patterns';
            }
            
            // Show limited activity insights
            if (activityInsights) {
                activityInsights.innerHTML = `
                    <h3>📊 Basic Activity Overview</h3>
                    <div class="mode-limitation-notice">
                        <p>⚠️ Limited insights available in Chrome data only mode</p>
                        <p>For detailed session tracking, behavioral insights, and real-time analytics, switch to <strong>Enhanced data mode</strong>.</p>
                    </div>
                `;
            }
            
            // Update insights panel
            if (insightsPanel) {
                const insightsContent = insightsPanel.querySelector('.insights-content') || insightsPanel;
                insightsContent.innerHTML = `
                    <p><strong>Chrome data only mode</strong></p>
                    <p>✅ Basic categorization and charts available</p>
                    <p>⚠️ Limited to browser history data only</p>
                    <p>🔄 Switch to Enhanced data mode for advanced AI insights</p>
                `;
            }
            
        } else if (mode === 'fused') {
            // Fused Mode - enable all features
            
            // Enable semantic search
            if (semanticSearchInput) {
                semanticSearchInput.disabled = false;
                semanticSearchInput.placeholder = 'Search for websites by content, topic, or activity... e.g., \'research papers about machine learning\'';
                semanticSearchInput.style.backgroundColor = '';
                semanticSearchInput.style.color = '';
            }
            if (searchBtn) {
                searchBtn.disabled = false;
                searchBtn.style.backgroundColor = '';
                searchBtn.style.cursor = 'pointer';
            }
            
            // Show accurate session data
            if (sessionCountElement) {
                sessionCountElement.title = 'Real-time session tracking with accurate timing';
            }
            if (avgSessionElement) {
                avgSessionElement.title = 'Precise average session time based on engagement tracking';
            }
            
            // Restore full activity insights (will be populated by loadData)
            if (activityInsights) {
                activityInsights.innerHTML = `
                    <h3>📊 Advanced Activity Insights</h3>
                    <div class="mode-enhancement-notice">
                        <p>✅ Full analytics available in Enhanced data mode</p>
                        <p>Real-time session tracking, behavioral insights, and AI-powered analysis active.</p>
                    </div>
                `;
            }
            
            // Update insights panel
            if (insightsPanel) {
                const insightsContent = insightsPanel.querySelector('.insights-content') || insightsPanel;
                insightsContent.innerHTML = `
                    <p><strong>Enhanced data mode active</strong></p>
                    <p>✅ Advanced AI categorization</p>
                    <p>✅ Real-time session tracking</p>
                    <p>✅ Behavioral insights & semantic search</p>
                `;
            }
        }
    }
}

// Auto-initialize if not already done
// Removed - now handled by init.js 