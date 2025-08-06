// History Plus Content Script - Enhanced Data Collection
// Collects rich contextual data for semantic search and LLM analysis

class HistoryPlusContentCollector {
  constructor() {
        this.pageData = {
            url: window.location.href,
            title: document.title,
            domain: window.location.hostname,
            timestamp: Date.now(),
            startTime: Date.now(),
            lastActiveTime: Date.now(),
            
            // Enhanced metadata
            description: '',
            keywords: [],
            pageType: 'webpage',
            contentSummary: '',
            
            // Precise engagement tracking
            timeOnPage: 0,
            focusTime: 0,           // Time when tab is visible and window focused
            activeTime: 0,          // Time when user is actively interacting (3-min idle threshold)
            idleTime: 0,            // Time when user is idle but page is in focus
            backgroundTime: 0,      // Time when tab/window is not focused
      scrollDepth: 0,
            maxScrollDepth: 0,
            
            // Interaction tracking
      mouseMovements: 0,
      keyPresses: 0,
            clicks: 0,
            scrollEvents: 0,
            
            // Context indicators
            hasVideo: false,
            hasAudio: false,
            hasImages: false,
            hasCode: false,
            textLength: 0,
            readingTime: 0,
            
            // Media engagement tracking
            videoPlayTime: 0,
            audioPlayTime: 0,
            mediaEngagementTime: 0
        };
        
        this.isActive = true;           // User is actively interacting
        this.isVisible = !document.hidden; // Tab is visible
        this.isFocused = document.hasFocus(); // Window has focus
        this.lastScrollY = window.scrollY;
        this.activeIdleThreshold = 180000; // 3 minutes = 180,000ms
        this.activeIdleTimer = null;
        this.lastInteractionTime = Date.now();
        
        // Media tracking
        this.videoElements = [];
        this.audioElements = [];
        this.mediaTrackingInterval = null;
    
    this.init();
  }

    init() {
        console.log('History Plus content collector initialized');
        
        // Collect initial page data
        this.collectPageMetadata();
        this.analyzePageContent();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start tracking loops
        this.startTrackingLoops();
        
        // Send initial data
        this.sendPageData();
    }

    collectPageMetadata() {
        // Basic metadata
        this.pageData.title = document.title || '';
        this.pageData.description = this.getMetaDescription();
        this.pageData.keywords = this.getMetaKeywords();
        
        // Detect page type and collect specific data
        this.pageData.pageType = this.detectPageType();
        
        // Collect content summary based on page type
        this.pageData.contentSummary = this.generateContentSummary();
    }

    getMetaDescription() {
        const metaDesc = document.querySelector('meta[name="description"]');
        return metaDesc ? metaDesc.content : '';
    }

    getMetaKeywords() {
        const metaKeywords = document.querySelector('meta[name="keywords"]');
        if (metaKeywords) {
            return metaKeywords.content.split(',').map(k => k.trim());
        }
        
        // Extract keywords from title and headings
        const titleWords = this.pageData.title.toLowerCase().split(/\s+/);
        const headings = Array.from(document.querySelectorAll('h1, h2, h3')).map(h => h.textContent);
        const headingWords = headings.join(' ').toLowerCase().split(/\s+/);
        
        return [...new Set([...titleWords, ...headingWords])].filter(word => word.length > 3);
    }

    detectPageType() {
        const url = window.location.href.toLowerCase();
        const domain = window.location.hostname.toLowerCase();
        
        // YouTube detection
        if (domain.includes('youtube.com')) {
            if (url.includes('/watch')) return 'youtube_video';
            if (url.includes('/channel') || url.includes('/c/')) return 'youtube_channel';
            return 'youtube';
        }
        
        // Social media detection
        if (domain.includes('twitter.com') || domain.includes('x.com')) return 'twitter';
        if (domain.includes('facebook.com')) return 'facebook';
        if (domain.includes('linkedin.com')) return 'linkedin';
        if (domain.includes('instagram.com')) return 'instagram';
        if (domain.includes('reddit.com')) return 'reddit';
        
        // Development platforms
        if (domain.includes('github.com')) return 'github';
        if (domain.includes('stackoverflow.com')) return 'stackoverflow';
        
        // Education platforms
        if (domain.includes('coursera.org')) return 'coursera';
        if (domain.includes('udemy.com')) return 'udemy';
        if (domain.includes('khanacademy.org')) return 'khan_academy';
        
        // News and articles
        if (domain.includes('medium.com')) return 'medium_article';
        if (domain.includes('substack.com')) return 'substack';
        if (domain.includes('wikipedia.org')) return 'wikipedia';
        
        // Gaming
        if (domain.includes('runescape.wiki')) return 'runescape_wiki';
        if (domain.includes('steam')) return 'steam';
        
        // Shopping
        if (domain.includes('amazon.com')) return 'amazon';
        if (domain.includes('ebay.com')) return 'ebay';
        
        // Generic types based on content
        if (document.querySelector('video')) return 'video_content';
        if (document.querySelector('article')) return 'article';
        if (document.querySelector('form')) return 'form_page';
        if (document.querySelector('code, pre')) return 'code_documentation';
    
    return 'webpage';
  }

    generateContentSummary() {
        const pageType = this.pageData.pageType;
        
        try {
            switch (pageType) {
                case 'youtube_video':
                    return this.summarizeYouTubeVideo();
                
                case 'runescape_wiki':
                    return this.summarizeRunescapeWiki();
                
                case 'wikipedia':
                    return this.summarizeWikipedia();
                
                case 'github':
                    return this.summarizeGitHub();
                
                case 'stackoverflow':
                    return this.summarizeStackOverflow();
                
                case 'medium_article':
                case 'article':
                    return this.summarizeArticle();
                
                case 'reddit':
                    return this.summarizeReddit();
                
                default:
                    return this.summarizeGenericPage();
            }
        } catch (error) {
            console.warn('Error generating content summary:', error);
            return this.summarizeGenericPage();
        }
    }

    summarizeYouTubeVideo() {
        const titleElement = document.querySelector('#title h1, #container h1');
        const title = titleElement ? titleElement.textContent.trim() : '';
        
        const descriptionElement = document.querySelector('#description-text, #meta-contents #description');
        let description = descriptionElement ? descriptionElement.textContent.trim() : '';
        
        // Clean up description - take first meaningful sentences
        if (description.length > 200) {
            const sentences = description.split('.').slice(0, 3);
            description = sentences.join('.') + (sentences.length === 3 ? '.' : '');
        }
        
        const channelElement = document.querySelector('#channel-name a, #owner-name a');
        const channel = channelElement ? channelElement.textContent.trim() : '';
        
        let summary = `YouTube Video: ${title}`;
        if (channel) summary += ` by ${channel}`;
        if (description && !this.isJunkDescription(description)) {
            summary += `. ${description}`;
        }
        
        return summary;
    }

    summarizeRunescapeWiki() {
        const title = document.querySelector('h1.firstHeading, #firstHeading')?.textContent?.trim() || '';
        const introText = document.querySelector('.mw-parser-output > p:first-of-type')?.textContent?.trim() || '';
        
        let summary = `RuneScape Wiki: ${title}`;
        if (introText && introText.length > 20) {
            const cleanIntro = introText.substring(0, 150).trim();
            summary += `. ${cleanIntro}`;
            if (introText.length > 150) summary += '...';
        }
        
        return summary;
    }

    summarizeWikipedia() {
        const title = document.querySelector('h1.firstHeading')?.textContent?.trim() || '';
        const firstParagraph = document.querySelector('.mw-parser-output > p:not(.mw-empty-elt)')?.textContent?.trim() || '';
        
        let summary = `Wikipedia: ${title}`;
        if (firstParagraph) {
            const cleanText = firstParagraph.substring(0, 200).trim();
            summary += `. ${cleanText}`;
            if (firstParagraph.length > 200) summary += '...';
        }
        
        return summary;
    }

    summarizeGitHub() {
        const repoName = document.querySelector('[data-testid="AppHeader-context-item-label"], .js-current-repository')?.textContent?.trim() || '';
        const description = document.querySelector('[data-testid="repository-description"], .repository-meta .f4')?.textContent?.trim() || '';
        const readmeContent = document.querySelector('#readme .Box-body p:first-of-type')?.textContent?.trim() || '';
        
        let summary = `GitHub Repository: ${repoName}`;
        if (description) summary += `. ${description}`;
        else if (readmeContent) summary += `. ${readmeContent.substring(0, 100)}`;
        
        return summary;
    }

    summarizeStackOverflow() {
        const questionTitle = document.querySelector('h1 a, #question-header h1')?.textContent?.trim() || '';
        const questionBody = document.querySelector('.js-post-body p:first-of-type')?.textContent?.trim() || '';
        
        let summary = `Stack Overflow: ${questionTitle}`;
        if (questionBody) {
            summary += `. ${questionBody.substring(0, 150)}`;
            if (questionBody.length > 150) summary += '...';
        }
        
        return summary;
    }

    summarizeArticle() {
        const title = document.querySelector('h1, .entry-title, [role="heading"]')?.textContent?.trim() || '';
        const firstParagraph = document.querySelector('article p:first-of-type, .content p:first-of-type, .post-content p:first-of-type')?.textContent?.trim() || '';
        
        let summary = `Article: ${title}`;
        if (firstParagraph) {
            summary += `. ${firstParagraph.substring(0, 150)}`;
            if (firstParagraph.length > 150) summary += '...';
        }
        
        return summary;
    }

    summarizeReddit() {
        const postTitle = document.querySelector('[data-testid="post-content"] h1, .title a')?.textContent?.trim() || '';
        const postText = document.querySelector('[data-testid="post-content"] div[data-testid="post-content"] > div > p')?.textContent?.trim() || '';
        
        let summary = `Reddit Post: ${postTitle}`;
        if (postText && postText.length > 20) {
            summary += `. ${postText.substring(0, 100)}`;
            if (postText.length > 100) summary += '...';
        }
        
        return summary;
    }

    summarizeGenericPage() {
        const headings = Array.from(document.querySelectorAll('h1, h2')).map(h => h.textContent.trim()).filter(h => h.length > 0);
        const firstParagraph = document.querySelector('p')?.textContent?.trim() || '';
        
        let summary = this.pageData.title;
        if (headings.length > 0 && headings[0] !== this.pageData.title) {
            summary += `. ${headings[0]}`;
        }
        if (firstParagraph && firstParagraph.length > 20) {
            summary += `. ${firstParagraph.substring(0, 100)}`;
            if (firstParagraph.length > 100) summary += '...';
        }
        
        return summary;
    }

    isJunkDescription(text) {
        const junkPatterns = [
            /^(like|subscribe|comment|share|follow)/i,
            /^(social media|links|contact)/i,
            /^(music|sound effects|background)/i,
            /^\s*\d+\s*$/,
            /^(▼|►|★|♪|♫|♪♫)/
        ];
        
        return junkPatterns.some(pattern => pattern.test(text)) || text.length < 20;
    }

    analyzePageContent() {
        // Content analysis
        this.pageData.hasVideo = document.querySelectorAll('video').length > 0;
        this.pageData.hasAudio = document.querySelectorAll('audio').length > 0;
        this.pageData.hasImages = document.querySelectorAll('img').length > 0;
        this.pageData.hasCode = document.querySelectorAll('code, pre').length > 0;
        
        // Text analysis
        const bodyText = document.body.textContent || '';
        this.pageData.textLength = bodyText.length;
        this.pageData.readingTime = Math.ceil(bodyText.split(' ').length / 200); // 200 WPM average
    }

    setupEventListeners() {
        // Focus and visibility tracking
        document.addEventListener('visibilitychange', () => {
            this.isVisible = !document.hidden;
            this.isFocused = document.hasFocus();
            this.updateBackgroundTime();
            this.resetActiveIdleTimer();
        });

        window.addEventListener('focus', () => {
            this.isActive = true;
            this.isFocused = true;
            this.pageData.lastActiveTime = Date.now();
            this.resetActiveIdleTimer();
        });

        window.addEventListener('blur', () => {
            this.isActive = false;
            this.isFocused = false;
        });

        // Interaction tracking
        document.addEventListener('mousemove', () => {
            this.pageData.mouseMovements++;
            this.pageData.lastInteractionTime = Date.now();
            this.resetActiveIdleTimer();
        });

        document.addEventListener('keypress', () => {
            this.pageData.keyPresses++;
            this.pageData.lastInteractionTime = Date.now();
            this.resetActiveIdleTimer();
        });

        document.addEventListener('click', () => {
            this.pageData.clicks++;
            this.pageData.lastInteractionTime = Date.now();
            this.resetActiveIdleTimer();
        });

        // Scroll tracking
        window.addEventListener('scroll', () => {
            this.pageData.scrollEvents++;
            this.updateScrollDepth();
            this.resetActiveIdleTimer();
        });

        // Media engagement tracking
        this.setupMediaTracking();

        // Page unload
        window.addEventListener('beforeunload', () => {
            this.finalizePageData();
            this.sendPageData();
        });
    }

    updateScrollDepth() {
        const scrollTop = window.scrollY;
        const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = documentHeight > 0 ? (scrollTop / documentHeight) : 0;
        
        this.pageData.scrollDepth = scrollPercent;
        if (scrollPercent > this.pageData.maxScrollDepth) {
            this.pageData.maxScrollDepth = scrollPercent;
        }
    }

    updateBackgroundTime() {
        if (!this.isVisible || !this.isFocused) {
            this.pageData.backgroundTime += 1000; // Increment by 1 second
        }
    }

    startTrackingLoops() {
        // Update time tracking every second with precise focus calculation
        setInterval(() => {
            const now = Date.now();
            const timeSinceLastInteraction = now - this.lastInteractionTime;
            
            this.pageData.timeOnPage = now - this.pageData.startTime;
            
            // Focus time: when tab is visible AND window is focused
            if (this.isVisible && this.isFocused) {
                this.pageData.focusTime += 1000;
                
                // Active time: focus time + within 3-minute interaction threshold OR media playing
                if (timeSinceLastInteraction < this.activeIdleThreshold || this.isMediaPlaying()) {
                    this.pageData.activeTime += 1000;
      } else {
                    // User is idle but page is still in focus
                    this.pageData.idleTime += 1000;
      }
    } else {
                // Page is not focused or visible
                this.pageData.backgroundTime += 1000;
            }
            
            // Track media engagement separately
            this.updateMediaPlayTime();
            
        }, 1000);

        // Send periodic updates every 30 seconds
        setInterval(() => {
            this.sendPageData();
        }, 30000);
        
        // Refresh media elements every 10 seconds (for dynamic content)
        setInterval(() => {
            this.refreshMediaElements();
        }, 10000);
    }

    isMediaPlaying() {
        // Check if any video or audio is currently playing
        const playingVideo = this.videoElements.some(video => !video.paused && !video.ended);
        const playingAudio = this.audioElements.some(audio => !audio.paused && !audio.ended);
        return playingVideo || playingAudio;
    }

    updateMediaPlayTime() {
        if (this.isVisible && this.isFocused) {
            // Track video play time
            this.videoElements.forEach(video => {
                if (!video.paused && !video.ended) {
                    this.pageData.videoPlayTime += 1000;
                }
            });
            
            // Track audio play time  
            this.audioElements.forEach(audio => {
                if (!audio.paused && !audio.ended) {
                    this.pageData.audioPlayTime += 1000;
                }
            });
        }
        
        // Calculate total media engagement time
        this.pageData.mediaEngagementTime = this.pageData.videoPlayTime + this.pageData.audioPlayTime;
    }

    refreshMediaElements() {
        // Refresh media elements for dynamically loaded content (like YouTube)
        this.videoElements = Array.from(document.querySelectorAll('video'));
        this.audioElements = Array.from(document.querySelectorAll('audio'));
        
        // Update has media flags
        this.pageData.hasVideo = this.videoElements.length > 0;
        this.pageData.hasAudio = this.audioElements.length > 0;
    }

    setupMediaTracking() {
        this.refreshMediaElements();
        
        // Add event listeners for media events
        document.addEventListener('play', (e) => {
            if (e.target.tagName === 'VIDEO' || e.target.tagName === 'AUDIO') {
                this.lastInteractionTime = Date.now();
                this.resetActiveIdleTimer();
            }
        }, true);
        
        document.addEventListener('pause', (e) => {
            if (e.target.tagName === 'VIDEO' || e.target.tagName === 'AUDIO') {
                this.lastInteractionTime = Date.now();
            }
        }, true);
    }

    resetActiveIdleTimer() {
        if (this.activeIdleTimer) {
            clearTimeout(this.activeIdleTimer);
        }
        
        this.lastInteractionTime = Date.now();
        this.isActive = true;
        
        // Set timer for when user becomes idle (3 minutes)
        this.activeIdleTimer = setTimeout(() => {
            this.isActive = false;
        }, this.activeIdleThreshold);
    }

    finalizePageData() {
        this.pageData.timeOnPage = Date.now() - this.pageData.startTime;
        this.updateScrollDepth();
    }

    sendPageData() {
        try {
            chrome.runtime.sendMessage({
                action: 'recordEnhancedEngagement',
                data: { ...this.pageData }
            });
        } catch (error) {
            console.warn('Failed to send page data:', error);
        }
    }
}

// Initialize the content collector
if (typeof window !== 'undefined' && window.location) {
    new HistoryPlusContentCollector();
} 