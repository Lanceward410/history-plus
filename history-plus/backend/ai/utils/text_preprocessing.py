"""
Text Preprocessing Utilities for AI Services

Provides comprehensive text cleaning and normalization functions optimized
for browser history data including:
- URL and domain extraction and cleaning
- HTML tag removal and decoding
- Unicode normalization
- Noise reduction for better embedding quality

Key Features:
- Tunable preprocessing intensity
- Domain-specific cleaning rules
- Performance optimization for batch processing
- Preservation of semantic meaning while removing noise
"""

import re
import html
import unicodedata
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, unquote
import string

class TextPreprocessor:
    """
    Comprehensive text preprocessing for browser history analysis
    
    Optimized for cleaning browser tab titles, URLs, and related text
    while preserving semantic meaning for AI processing.
    """
    
    def __init__(self, aggressive_cleaning: bool = False):
        """
        Initialize text preprocessor
        
        Args:
            aggressive_cleaning: If True, apply more aggressive cleaning that may
                               remove semantic information but improves embedding quality
        """
        self.aggressive_cleaning = aggressive_cleaning
        
        # TUNABLE: Preprocessing parameters
        self.max_text_length = 500  # TUNE: Maximum text length to prevent huge inputs
        self.min_text_length = 3   # TUNE: Minimum meaningful text length
        self.preserve_case = False  # TUNE: Whether to preserve original casing
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Common noise words to remove in aggressive mode
        # TUNABLE: Add/remove noise words based on your use case
        self.noise_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        } if aggressive_cleaning else set()
        
        # Domain-specific cleaning rules
        self.domain_rules = self._load_domain_rules()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing"""
        
        # HTML patterns
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.html_entity_pattern = re.compile(r'&[a-zA-Z0-9]+;')
        
        # URL patterns
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Whitespace patterns
        self.multiple_spaces = re.compile(r'\s+')
        self.leading_trailing_spaces = re.compile(r'^\s+|\s+$')
        
        # Special character patterns
        self.special_chars = re.compile(r'[^\w\s\-.,!?:;()"\']')  # TUNABLE: Adjust allowed chars
        self.repeated_punctuation = re.compile(r'([.!?]){2,}')
        
        # Browser-specific patterns
        self.browser_noise = re.compile(r'\s*[-|•]\s*(Google Chrome|Mozilla Firefox|Safari|Edge|Browser).*$', re.IGNORECASE)
        self.tab_counters = re.compile(r'\s*\(\d+\)\s*$')  # Remove (1), (2) etc.
        
        # Social media patterns
        self.twitter_handles = re.compile(r'@\w+')
        self.hashtags = re.compile(r'#\w+')
    
    def _load_domain_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Load domain-specific cleaning rules
        
        Returns dict mapping domain patterns to cleaning configurations
        """
        return {
            # YouTube specific cleaning
            'youtube.com': {
                'remove_patterns': [r'\s*-\s*YouTube\s*$'],
                'preserve_patterns': [r'\b\d+[KMB]?\s*views?\b'],  # View counts
                'title_separators': [' - ', ' | ']
            },
            
            # Reddit specific cleaning
            'reddit.com': {
                'remove_patterns': [r'\s*:\s*reddit\s*$', r'\s*•\s*r/\w+\s*$'],
                'preserve_patterns': [r'\br/\w+\b'],  # Subreddit names
                'title_separators': [' : ']
            },
            
            # Wikipedia specific cleaning
            'wikipedia.org': {
                'remove_patterns': [r'\s*-\s*Wikipedia\s*$'],
                'preserve_patterns': [r'\b\d{4}\b'],  # Years
                'title_separators': [' - ']
            },
            
            # News sites
            'cnn.com': {
                'remove_patterns': [r'\s*-\s*CNN\s*$'],
                'title_separators': [' - ']
            },
            
            # GitHub
            'github.com': {
                'preserve_patterns': [r'\b[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+\b'],  # repo names
                'remove_patterns': [r'\s*·\s*GitHub\s*$']
            },
            
            # Shopping sites
            'amazon.com': {
                'remove_patterns': [r'\s*-\s*Amazon\.com\s*$'],
                'preserve_patterns': [r'\$[\d,]+\.?\d*', r'\b\d+\.\d+\s*out\s*of\s*5\b']  # Prices, ratings
            }
        }
    
    def clean_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Clean and normalize text for AI processing
        
        Args:
            text: Input text to clean
            context: Optional context ('title', 'url', 'domain') for specialized cleaning
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Apply context-specific preprocessing
        if context == 'url':
            text = self._clean_url(text)
        elif context == 'title':
            text = self._clean_title(text)
        elif context == 'domain':
            text = self._clean_domain(text)
        
        # Basic cleaning steps
        text = self._remove_html(text)
        text = self._normalize_unicode(text)
        text = self._remove_urls_emails(text)
        text = self._normalize_whitespace(text)
        text = self._remove_special_characters(text)
        
        # Aggressive cleaning if enabled
        if self.aggressive_cleaning:
            text = self._aggressive_clean(text)
        
        # Length constraints
        text = self._apply_length_constraints(text)
        
        return text.strip()
    
    def clean_batch(self, texts: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        """
        Clean multiple texts efficiently
        
        Args:
            texts: List of texts to clean
            contexts: Optional list of contexts for each text
            
        Returns:
            List of cleaned texts
        """
        if contexts is None:
            contexts = [None] * len(texts)
        
        return [self.clean_text(text, context) for text, context in zip(texts, contexts)]
    
    def _clean_url(self, url: str) -> str:
        """Clean and extract meaningful information from URLs"""
        try:
            parsed = urlparse(url)
            
            # Extract domain
            domain = parsed.netloc.lower()
            
            # Clean path - decode URL encoding
            path = unquote(parsed.path)
            
            # Extract meaningful parts
            path_parts = [part for part in path.split('/') if part and len(part) > 1]
            
            # Combine domain and meaningful path parts
            meaningful_parts = [domain] + path_parts[:3]  # TUNABLE: Limit path depth
            
            return ' '.join(meaningful_parts).replace('-', ' ').replace('_', ' ')
            
        except Exception:
            # Fallback to simple cleaning
            return re.sub(r'https?://', '', url).replace('/', ' ').replace('-', ' ')
    
    def _clean_title(self, title: str) -> str:
        """Clean browser tab titles with domain-specific rules"""
        
        # Remove browser-specific noise
        title = self.browser_noise.sub('', title)
        title = self.tab_counters.sub('', title)
        
        # Apply domain-specific rules if available
        # Note: Would need domain context to apply these rules
        
        return title
    
    def _clean_domain(self, domain: str) -> str:
        """Clean and normalize domain names"""
        domain = domain.lower()
        
        # Remove www prefix
        domain = re.sub(r'^www\.', '', domain)
        
        # Remove common subdomains that add no semantic value
        domain = re.sub(r'^(m|mobile|app|api|cdn)\.', '', domain)
        
        # Split domain parts and keep meaningful ones
        parts = domain.split('.')
        if len(parts) >= 2:
            # Keep main domain name and TLD
            return f"{parts[-2]} {parts[-1]}"
        
        return domain.replace('.', ' ')
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and decode entities"""
        # Remove HTML tags
        text = self.html_tag_pattern.sub(' ', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        # Normalize to NFC form (canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Remove or replace problematic Unicode categories
        # TUNABLE: Adjust based on your text requirements
        cleaned_chars = []
        for char in text:
            category = unicodedata.category(char)
            if category.startswith('C'):  # Control characters
                cleaned_chars.append(' ')
            elif category in ['Mn', 'Mc']:  # Combining marks - keep them
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)
    
    def _remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses"""
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple whitespace with single space
        text = self.multiple_spaces.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove or normalize special characters"""
        # Normalize repeated punctuation
        text = self.repeated_punctuation.sub(r'\1', text)
        
        # Remove excessive special characters (but keep some for meaning)
        if self.aggressive_cleaning:
            text = self.special_chars.sub(' ', text)
        
        return text
    
    def _aggressive_clean(self, text: str) -> str:
        """Apply aggressive cleaning that may remove semantic information"""
        
        # Remove noise words
        if self.noise_words:
            words = text.lower().split()
            words = [word for word in words if word not in self.noise_words]
            text = ' '.join(words)
        
        # Remove social media artifacts
        text = self.twitter_handles.sub('', text)
        text = self.hashtags.sub('', text)
        
        # Remove numbers in aggressive mode (TUNABLE)
        # text = re.sub(r'\b\d+\b', '', text)  # Uncomment to remove all numbers
        
        return text
    
    def _apply_length_constraints(self, text: str) -> str:
        """Apply minimum and maximum length constraints"""
        
        # Remove if too short
        if len(text) < self.min_text_length:
            return ""
        
        # Truncate if too long
        if len(text) > self.max_text_length:
            # Try to break at word boundary
            truncated = text[:self.max_text_length].rsplit(' ', 1)[0]
            if len(truncated) >= self.min_text_length:
                return truncated + '...'
            else:
                return text[:self.max_text_length] + '...'
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract meaningful keywords from cleaned text
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords
        """
        # Clean text first
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return []
        
        # Split into words
        words = cleaned.lower().split()
        
        # Filter words
        keywords = []
        for word in words:
            # Skip short words and noise
            if (len(word) >= 3 and 
                word not in self.noise_words and
                not word.isdigit() and
                word.isalpha()):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
                if len(unique_keywords) >= max_keywords:
                    break
        
        return unique_keywords
    
    def get_preprocessing_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing operation
        
        Args:
            original_text: Original input text
            cleaned_text: Processed output text
            
        Returns:
            Dictionary with preprocessing statistics
        """
        return {
            "original_length": len(original_text),
            "cleaned_length": len(cleaned_text),
            "reduction_ratio": 1 - (len(cleaned_text) / len(original_text)) if original_text else 0,
            "original_words": len(original_text.split()) if original_text else 0,
            "cleaned_words": len(cleaned_text.split()) if cleaned_text else 0,
            "aggressive_cleaning": self.aggressive_cleaning,
            "empty_result": len(cleaned_text) == 0
        }

# Global preprocessor instances
_standard_preprocessor: Optional[TextPreprocessor] = None
_aggressive_preprocessor: Optional[TextPreprocessor] = None

def get_text_preprocessor(aggressive: bool = False) -> TextPreprocessor:
    """
    Get global text preprocessor instance
    
    Args:
        aggressive: Whether to use aggressive cleaning
        
    Returns:
        TextPreprocessor instance
    """
    global _standard_preprocessor, _aggressive_preprocessor
    
    if aggressive:
        if _aggressive_preprocessor is None:
            _aggressive_preprocessor = TextPreprocessor(aggressive_cleaning=True)
        return _aggressive_preprocessor
    else:
        if _standard_preprocessor is None:
            _standard_preprocessor = TextPreprocessor(aggressive_cleaning=False)
        return _standard_preprocessor

def clean_browser_data(titles: List[str], urls: List[str], domains: List[str], 
                      aggressive: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """
    Clean browser history data in batch
    
    Args:
        titles: List of tab titles
        urls: List of URLs
        domains: List of domain names
        aggressive: Whether to use aggressive cleaning
        
    Returns:
        Tuple of (cleaned_titles, cleaned_urls, cleaned_domains)
    """
    preprocessor = get_text_preprocessor(aggressive=aggressive)
    
    # Clean each type with appropriate context
    cleaned_titles = preprocessor.clean_batch(titles, ['title'] * len(titles))
    cleaned_urls = preprocessor.clean_batch(urls, ['url'] * len(urls))
    cleaned_domains = preprocessor.clean_batch(domains, ['domain'] * len(domains))
    
    return cleaned_titles, cleaned_urls, cleaned_domains 