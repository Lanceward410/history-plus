"""
Hierarchical category framework for consistent tab classification
"""

CATEGORY_FRAMEWORK = {
    "Social Media": {
        "description": "Social networking, communication, and content sharing platforms",
        "keywords": ["social", "chat", "messaging", "networking", "community", "friend", "follow"],
        "domains": ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "discord.com", "snapchat.com", "tiktok.com"],
        "common_subcategories": [
            "Personal Browsing", "Professional Networking", "Community Discussion", 
            "Content Creation", "Job Searching", "News Feed Scrolling"
        ]
    },
    "Gaming": {
        "description": "Video games, game guides, gaming communities, and gaming content",
        "keywords": ["game", "gaming", "play", "guide", "wiki", "strategy", "quest", "level", "character"],
        "domains": ["steam.com", "runescape.com", "minecraft.net", "twitch.tv", "ign.com", "gamespot.com", "gamefaqs.com"],
        "common_subcategories": [
            "Playing", "Quest Research", "Strategy Guides", "Community Discussion",
            "Game Reviews", "Streaming", "Mod Research", "Achievement Hunting"
        ]
    },
    "Academic Research": {
        "description": "Educational content, research papers, academic institutions, scholarly resources",
        "keywords": ["research", "paper", "study", "academic", "university", "journal", "publication", "thesis"],
        "domains": ["ieee.org", "arxiv.org", "scholar.google.com", "researchgate.net", "pubmed.ncbi.nlm.nih.gov", "jstor.org"],
        "common_subcategories": [
            "Literature Review", "Paper Research", "Course Materials", "Reference Lookup",
            "Data Analysis", "Citation Research", "Conference Information"
        ]
    },
    "Work/Professional": {
        "description": "Work-related tasks, professional tools, business activities, career development",
        "keywords": ["work", "professional", "business", "career", "job", "corporate", "office", "meeting"],
        "domains": ["slack.com", "teams.microsoft.com", "jira.atlassian.com", "github.com", "google.com/workspace"],
        "common_subcategories": [
            "Team Communication", "Code Development", "Project Management", "Documentation",
            "Email", "Calendar Management", "File Sharing", "Video Conferencing"
        ]
    },
    "Shopping": {
        "description": "E-commerce, price comparison, product research, online purchasing",
        "keywords": ["buy", "shop", "price", "product", "review", "deal", "cart", "purchase"],
        "domains": ["amazon.com", "ebay.com", "walmart.com", "target.com", "bestbuy.com", "etsy.com"],
        "common_subcategories": [
            "Product Research", "Price Comparison", "Purchase Process", "Reviews Reading",
            "Deal Hunting", "Wishlist Management", "Return Process"
        ]
    },
    "Entertainment": {
        "description": "Videos, music, movies, TV shows, general entertainment content",
        "keywords": ["watch", "video", "music", "movie", "tv", "entertainment", "show", "series"],
        "domains": ["youtube.com", "netflix.com", "spotify.com", "hulu.com", "disney.com", "twitch.tv"],
        "common_subcategories": [
            "Video Watching", "Music Listening", "Movies/TV Shows", "Content Discovery",
            "Playlist Creation", "Artist Research", "Episode Tracking"
        ]
    },
    "Learning/Tutorials": {
        "description": "Educational content, how-to guides, skill development, online courses",
        "keywords": ["tutorial", "learn", "course", "how-to", "guide", "education", "training", "skill"],
        "domains": ["coursera.org", "udemy.com", "khanacademy.org", "codecademy.com", "stackoverflow.com"],
        "common_subcategories": [
            "Programming Tutorials", "Creative Skills", "Technical Documentation", "Online Courses",
            "Skill Development", "Certification Study", "Problem Solving"
        ]
    },
    "News/Information": {
        "description": "News websites, current events, general information gathering, fact-checking",
        "keywords": ["news", "information", "current", "events", "politics", "world", "breaking", "update"],
        "domains": ["cnn.com", "bbc.com", "reuters.com", "wikipedia.org", "ap.org", "nytimes.com"],
        "common_subcategories": [
            "Breaking News", "Research/Fact-Checking", "General Reading", "Political News",
            "Weather", "Local News", "International News"
        ]
    },
    "AI/Development Tools": {
        "description": "AI assistants, development environments, technical tools, coding resources",
        "keywords": ["ai", "development", "code", "programming", "tool", "assistant", "api", "documentation"],
        "domains": ["chat.openai.com", "github.com", "stackoverflow.com", "replit.com", "codepen.io"],
        "common_subcategories": [
            "Code Assistance", "Problem Solving", "API Documentation", "Tool Configuration",
            "AI Chat", "Code Review", "Development Environment"
        ]
    },
    "Health/Wellness": {
        "description": "Health information, fitness, medical research, wellness activities",
        "keywords": ["health", "fitness", "medical", "wellness", "exercise", "nutrition", "doctor", "medicine"],
        "domains": ["webmd.com", "mayoclinic.org", "healthline.com", "myfitnesspal.com"],
        "common_subcategories": [
            "Medical Research", "Fitness Planning", "Nutrition Information", "Mental Health",
            "Symptom Research", "Wellness Tips"
        ]
    }
}

class CategoryFramework:
    """Manages the category framework and provides matching utilities"""
    
    def __init__(self):
        self.framework = CATEGORY_FRAMEWORK
        
    def get_all_categories(self):
        """Get list of all available categories"""
        return list(self.framework.keys())
    
    def get_category_info(self, category_name):
        """Get detailed information about a specific category"""
        return self.framework.get(category_name, {})
    
    def get_common_subcategories(self, category_name):
        """Get common subcategories for a given category"""
        return self.framework.get(category_name, {}).get('common_subcategories', [])
    
    def get_category_domains(self, category_name):
        """Get typical domains for a category"""
        return self.framework.get(category_name, {}).get('domains', [])
    
    def get_category_keywords(self, category_name):
        """Get keywords associated with a category"""
        return self.framework.get(category_name, {}).get('keywords', [])
    
    def find_category_by_domain(self, domain):
        """Find the most likely category for a given domain"""
        domain_lower = domain.lower()
        
        for category_name, category_info in self.framework.items():
            for cat_domain in category_info.get('domains', []):
                if cat_domain.lower() in domain_lower:
                    return category_name
        
        return None
    
    def get_framework_stats(self):
        """Get statistics about the framework"""
        return {
            "total_categories": len(self.framework),
            "total_domains": sum(len(cat.get('domains', [])) for cat in self.framework.values()),
            "total_keywords": sum(len(cat.get('keywords', [])) for cat in self.framework.values()),
            "categories": list(self.framework.keys())
        } 