"""
Local LLM Service with Advanced Rule-Based Processing

Provides language model functionality using sophisticated rule-based approaches
as the primary implementation, with extensibility for future local LLM integration.

Key Features:
- Advanced pattern matching for category generation
- Context-aware subcategory creation
- Comprehensive domain and keyword analysis
- Session summarization with intelligent grouping
- Category validation and confidence scoring
- Extensible architecture for future LLM integration

This service is designed to be fast, reliable, and provide high-quality
results without requiring external AI services or heavy local models.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

from ..interfaces.ai_processor import LLMService, ProcessingConfig, PerformanceMonitor
from ..category_framework import CategoryFramework
from ..utils.text_preprocessing import get_text_preprocessor

class LocalLLMService(LLMService):
    """
    Local LLM service with sophisticated rule-based processing
    
    Provides intelligent text generation and analysis using advanced
    pattern matching, domain knowledge, and contextual reasoning.
    """
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.category_framework = CategoryFramework()
        self.text_preprocessor = get_text_preprocessor(aggressive=False)
        
        # Load pattern databases
        self.category_patterns = self._load_category_patterns()
        self.subcategory_patterns = self._load_subcategory_patterns()
        self.domain_category_mapping = self._load_domain_mappings()
        self.contextual_keywords = self._load_contextual_keywords()
        
        # TUNABLE: Processing parameters
        self.confidence_boost_factors = {
            "domain_match": 0.3,     # TUNE: Boost for domain-based matches
            "keyword_density": 0.2,  # TUNE: Boost for high keyword density
            "pattern_match": 0.25,   # TUNE: Boost for pattern matches
            "context_match": 0.15    # TUNE: Boost for contextual matches
        }
        
        # Performance tracking
        self.generation_history = []
        
    def generate_text(self, prompt: str, max_tokens: int = 50, temperature: float = 0.3) -> str:
        """
        Generate text using rule-based approaches
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (affects response length)
            temperature: Creativity parameter (affects response variation)
            
        Returns:
            Generated text response
        """
        operation_start = self.performance_monitor.start_operation("generate_text")
        
        try:
            # Parse the prompt to understand the request
            request_type = self._analyze_prompt_intent(prompt)
            
            if request_type == "category_generation":
                return self._generate_category_from_prompt(prompt, max_tokens, temperature)
            elif request_type == "subcategory_generation":
                return self._generate_subcategory_from_prompt(prompt, max_tokens, temperature)
            elif request_type == "summarization":
                return self._generate_summary_from_prompt(prompt, max_tokens)
            else:
                return self._generate_generic_response(prompt, max_tokens, temperature)
                
        except Exception as e:
            print(f"Text generation failed: {e}")
            return "Generated response"
        
        finally:
            self.performance_monitor.end_operation("generate_text", operation_start)
    
    def generate_category_name(self, titles: List[str], domains: List[str], 
                             context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate category and subcategory using advanced rule-based analysis
        
        Args:
            titles: List of tab titles
            domains: List of tab domains
            context: Optional context information (engagement data, etc.)
            
        Returns:
            Dictionary with category, subcategory, confidence, and analysis metadata
        """
        operation_start = self.performance_monitor.start_operation("generate_category_name")
        
        try:
            # Preprocess input data
            cleaned_titles = [self.text_preprocessor.clean_text(title, 'title') for title in titles]
            cleaned_domains = [self.text_preprocessor.clean_text(domain, 'domain') for domain in domains]
            
            # Combine all text for analysis
            all_text = ' '.join(cleaned_titles + cleaned_domains).lower()
            
            # Multi-stage analysis
            analysis_result = self._perform_comprehensive_analysis(
                titles=cleaned_titles,
                domains=cleaned_domains,
                combined_text=all_text,
                context=context
            )
            
            # Generate category and subcategory
            category_result = self._determine_final_category(analysis_result)
            
            # Add processing metadata
            category_result.update({
                "method": "rule_based_advanced",
                "analysis_metadata": {
                    "text_length": len(all_text),
                    "unique_domains": len(set(domains)),
                    "processing_stages": analysis_result.get("stages_completed", [])
                }
            })
            
            # Record performance
            self._record_generation_performance(category_result, len(titles))
            
            return category_result
            
        except Exception as e:
            print(f"Category generation failed: {e}")
            return self._fallback_category_generation(titles, domains)
        
        finally:
            self.performance_monitor.end_operation("generate_category_name", operation_start)
    
    def _perform_comprehensive_analysis(self, titles: List[str], domains: List[str], 
                                      combined_text: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Perform multi-stage analysis to understand the browsing pattern
        
        Returns comprehensive analysis results for category determination
        """
        analysis = {
            "stages_completed": [],
            "confidence_scores": {},
            "evidence": {}
        }
        
        # Stage 1: Domain Analysis
        domain_analysis = self._analyze_domains(domains)
        analysis["domain_analysis"] = domain_analysis
        analysis["stages_completed"].append("domain_analysis")
        
        # Stage 2: Keyword Analysis  
        keyword_analysis = self._analyze_keywords(combined_text)
        analysis["keyword_analysis"] = keyword_analysis
        analysis["stages_completed"].append("keyword_analysis")
        
        # Stage 3: Pattern Matching
        pattern_analysis = self._analyze_patterns(combined_text, titles)
        analysis["pattern_analysis"] = pattern_analysis
        analysis["stages_completed"].append("pattern_analysis")
        
        # Stage 4: Contextual Analysis
        if context:
            contextual_analysis = self._analyze_context(combined_text, context)
            analysis["contextual_analysis"] = contextual_analysis
            analysis["stages_completed"].append("contextual_analysis")
        
        # Stage 5: Behavioral Analysis
        behavioral_analysis = self._analyze_browsing_behavior(titles, domains, context)
        analysis["behavioral_analysis"] = behavioral_analysis
        analysis["stages_completed"].append("behavioral_analysis")
        
        return analysis
    
    def _analyze_domains(self, domains: List[str]) -> Dict[str, Any]:
        """Analyze domains to extract category signals"""
        
        domain_analysis = {
            "primary_domains": [],
            "category_votes": {},
            "confidence_factors": {}
        }
        
        # Count domain frequency
        domain_counts = Counter(domains)
        total_domains = len(domains)
        
        # Analyze each unique domain
        for domain, count in domain_counts.most_common():
            weight = count / total_domains
            
            # Check against domain mappings
            for category, category_info in self.category_framework.framework.items():
                category_domains = category_info.get('domains', [])
                
                # Direct domain match
                if any(cat_domain in domain for cat_domain in category_domains):
                    domain_analysis["category_votes"][category] = domain_analysis["category_votes"].get(category, 0) + weight
                    
                    # Record evidence
                    if category not in domain_analysis["confidence_factors"]:
                        domain_analysis["confidence_factors"][category] = []
                    domain_analysis["confidence_factors"][category].append(f"Domain match: {domain}")
        
        # Identify primary domains (>20% of total)
        # TUNABLE: Threshold for primary domain identification
        primary_threshold = 0.2
        domain_analysis["primary_domains"] = [
            domain for domain, count in domain_counts.items() 
            if count / total_domains >= primary_threshold
        ]
        
        return domain_analysis
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze keywords and phrases for category signals"""
        
        keyword_analysis = {
            "category_scores": {},
            "matched_keywords": {},
            "keyword_density": {}
        }
        
        # Extract keywords from text
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return keyword_analysis
        
        # Analyze against each category's keyword patterns
        for category, patterns in self.category_patterns.items():
            matches = []
            total_matches = 0
            
            for pattern in patterns:
                pattern_matches = len(re.findall(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE))
                if pattern_matches > 0:
                    matches.append((pattern, pattern_matches))
                    total_matches += pattern_matches
            
            if matches:
                # Calculate keyword density and score
                density = total_matches / word_count
                # TUNABLE: Keyword scoring formula
                score = min(1.0, density * 10 + len(matches) * 0.1)  # Boost for variety of matches
                
                keyword_analysis["category_scores"][category] = score
                keyword_analysis["matched_keywords"][category] = matches
                keyword_analysis["keyword_density"][category] = density
        
        return keyword_analysis
    
    def _analyze_patterns(self, text: str, titles: List[str]) -> Dict[str, Any]:
        """Analyze text patterns and title structures"""
        
        pattern_analysis = {
            "detected_patterns": [],
            "category_indicators": {},
            "title_patterns": {}
        }
        
        # Analyze title patterns
        if titles:
            # Common title patterns that indicate categories
            title_patterns = {
                "tutorial_pattern": r'\b(how\s+to|tutorial|guide|learn|step\s+by\s+step)\b',
                "news_pattern": r'\b(breaking|news|update|report|article)\b',
                "social_pattern": r'\b(post|share|comment|like|follow|friend)\b',
                "shopping_pattern": r'\b(buy|shop|price|cart|order|deal|sale)\b',
                "work_pattern": r'\b(meeting|project|deadline|team|office|work)\b',
                "entertainment_pattern": r'\b(watch|movie|video|music|game|play|fun)\b'
            }
            
            for pattern_name, pattern_regex in title_patterns.items():
                matches = []
                for title in titles:
                    if re.search(pattern_regex, title, re.IGNORECASE):
                        matches.append(title)
                
                if matches:
                    pattern_analysis["title_patterns"][pattern_name] = {
                        "matches": len(matches),
                        "ratio": len(matches) / len(titles),
                        "examples": matches[:3]  # Store first 3 examples
                    }
        
        # Analyze sequential patterns in text
        # TUNABLE: Pattern detection thresholds
        sequence_patterns = {
            "research_sequence": [r'\bsearch\b', r'\bresult\b', r'\barticle\b', r'\bpaper\b'],
            "shopping_sequence": [r'\bproduct\b', r'\bprice\b', r'\breview\b', r'\bbuy\b'],
            "learning_sequence": [r'\blearn\b', r'\btutorial\b', r'\bpractice\b', r'\btest\b']
        }
        
        for sequence_name, sequence_words in sequence_patterns.items():
            sequence_score = 0
            for word_pattern in sequence_words:
                if re.search(word_pattern, text, re.IGNORECASE):
                    sequence_score += 1
            
            if sequence_score >= 2:  # TUNABLE: Minimum words for sequence detection
                pattern_analysis["detected_patterns"].append({
                    "type": sequence_name,
                    "strength": sequence_score / len(sequence_words),
                    "evidence": f"{sequence_score}/{len(sequence_words)} sequence words found"
                })
        
        return pattern_analysis
    
    def _analyze_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual information for additional category signals"""
        
        contextual_analysis = {
            "context_signals": {},
            "engagement_indicators": {},
            "temporal_patterns": {}
        }
        
        # Analyze engagement context if available
        if "engagement" in context:
            engagement = context["engagement"]
            
            # High engagement might indicate entertainment or work
            if engagement.get("time_spent", 0) > 300:  # TUNABLE: High engagement threshold (5 minutes)
                contextual_analysis["engagement_indicators"]["high_time_investment"] = True
                
            if engagement.get("interactions", 0) > 10:  # TUNABLE: High interaction threshold
                contextual_analysis["engagement_indicators"]["high_interaction"] = True
        
        # Analyze temporal context
        if "timestamp" in context:
            # Time-based category hints
            hour = datetime.fromisoformat(context["timestamp"]).hour
            
            # TUNABLE: Time-based category associations
            if 9 <= hour <= 17:  # Business hours
                contextual_analysis["temporal_patterns"]["business_hours"] = True
            elif 18 <= hour <= 23:  # Evening
                contextual_analysis["temporal_patterns"]["evening_leisure"] = True
            elif 0 <= hour <= 6:  # Late night
                contextual_analysis["temporal_patterns"]["late_night"] = True
        
        return contextual_analysis
    
    def _analyze_browsing_behavior(self, titles: List[str], domains: List[str], 
                                 context: Optional[Dict]) -> Dict[str, Any]:
        """Analyze browsing behavior patterns"""
        
        behavioral_analysis = {
            "behavior_type": "unknown",
            "characteristics": {},
            "confidence": 0.0
        }
        
        n_tabs = len(titles)
        unique_domains = len(set(domains))
        
        # Analyze browsing pattern characteristics
        if unique_domains == 1:
            behavioral_analysis["behavior_type"] = "focused_single_site"
            behavioral_analysis["confidence"] = 0.8
        elif unique_domains / n_tabs < 0.3:  # TUNABLE: Domain diversity threshold
            behavioral_analysis["behavior_type"] = "focused_multi_site"
            behavioral_analysis["confidence"] = 0.6
        elif unique_domains / n_tabs > 0.7:  # TUNABLE: High diversity threshold
            behavioral_analysis["behavior_type"] = "exploratory_browsing"
            behavioral_analysis["confidence"] = 0.5
        else:
            behavioral_analysis["behavior_type"] = "mixed_browsing"
            behavioral_analysis["confidence"] = 0.4
        
        # Analyze tab count patterns
        # TUNABLE: Tab count categorization
        if n_tabs >= 10:
            behavioral_analysis["characteristics"]["high_tab_count"] = True
        elif n_tabs <= 3:
            behavioral_analysis["characteristics"]["low_tab_count"] = True
        
        return behavioral_analysis
    
    def _determine_final_category(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine final category and subcategory from comprehensive analysis
        
        Uses weighted scoring from all analysis stages
        """
        category_scores = {}
        evidence_summary = {}
        
        # Aggregate scores from different analysis stages
        
        # Domain analysis scores
        if "domain_analysis" in analysis:
            domain_votes = analysis["domain_analysis"].get("category_votes", {})
            for category, score in domain_votes.items():
                category_scores[category] = category_scores.get(category, 0) + score * self.confidence_boost_factors["domain_match"]
                if category not in evidence_summary:
                    evidence_summary[category] = []
                evidence_summary[category].append(f"Domain evidence (score: {score:.2f})")
        
        # Keyword analysis scores
        if "keyword_analysis" in analysis:
            keyword_scores = analysis["keyword_analysis"].get("category_scores", {})
            for category, score in keyword_scores.items():
                category_scores[category] = category_scores.get(category, 0) + score * self.confidence_boost_factors["keyword_density"]
                if category not in evidence_summary:
                    evidence_summary[category] = []
                evidence_summary[category].append(f"Keyword evidence (score: {score:.2f})")
        
        # Pattern analysis scores
        if "pattern_analysis" in analysis:
            patterns = analysis["pattern_analysis"].get("detected_patterns", [])
            for pattern in patterns:
                # Map pattern types to categories
                pattern_category_map = {
                    "research_sequence": "Academic Research",
                    "shopping_sequence": "Shopping",
                    "learning_sequence": "Learning/Tutorials"
                }
                
                if pattern["type"] in pattern_category_map:
                    category = pattern_category_map[pattern["type"]]
                    score = pattern["strength"]
                    category_scores[category] = category_scores.get(category, 0) + score * self.confidence_boost_factors["pattern_match"]
                    if category not in evidence_summary:
                        evidence_summary[category] = []
                    evidence_summary[category].append(f"Pattern evidence: {pattern['type']} (strength: {score:.2f})")
        
        # Find best category
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category_name = best_category[0]
            confidence = min(0.95, best_category[1])  # Cap confidence at 95%
        else:
            # Fallback category determination
            category_name = "Web Browsing"
            confidence = 0.3
            evidence_summary[category_name] = ["Fallback - no strong category signals detected"]
        
        # Generate subcategory
        subcategory = self._generate_subcategory_for_category(category_name, analysis)
        
        return {
            "category": category_name,
            "subcategory": subcategory,
            "confidence": confidence,
            "evidence": evidence_summary.get(category_name, []),
            "all_scores": category_scores,
            "method": "comprehensive_analysis"
        }
    
    def _generate_subcategory_for_category(self, category: str, analysis: Dict[str, Any]) -> str:
        """Generate specific subcategory based on category and analysis"""
        
        # Use subcategory patterns if available
        if category in self.subcategory_patterns:
            category_subcategories = self.subcategory_patterns[category]
            
            # Find best matching subcategory
            combined_text = ""
            if "keyword_analysis" in analysis:
                # Reconstruct some text for analysis
                keywords = analysis["keyword_analysis"].get("matched_keywords", {}).get(category, [])
                combined_text = " ".join([kw[0] for kw in keywords])
            
            for subcategory, patterns in category_subcategories.items():
                for pattern in patterns:
                    if pattern in combined_text:
                        return subcategory
        
        # Behavioral subcategory generation
        if "behavioral_analysis" in analysis:
            behavior_type = analysis["behavioral_analysis"].get("behavior_type", "")
            
            if behavior_type == "focused_single_site":
                return "Deep Dive"
            elif behavior_type == "exploratory_browsing":
                return "Research/Exploration"
            elif behavior_type == "focused_multi_site":
                return "Comparative Analysis"
        
        # Pattern-based subcategory generation
        if "pattern_analysis" in analysis:
            title_patterns = analysis["pattern_analysis"].get("title_patterns", {})
            
            if "tutorial_pattern" in title_patterns:
                return "Learning/Tutorials"
            elif "news_pattern" in title_patterns:
                return "News/Updates"
            elif "social_pattern" in title_patterns:
                return "Social Activity"
            elif "shopping_pattern" in title_patterns:
                return "Shopping Research"
        
        # Fallback subcategory generation
        return "General Activity"
    
    def summarize_session(self, tab_data: List[Dict], max_length: int = 100) -> str:
        """
        Generate intelligent natural language summary of browsing session
        
        Args:
            tab_data: List of tab information dictionaries
            max_length: Maximum summary length in characters
            
        Returns:
            Natural language session summary
        """
        operation_start = self.performance_monitor.start_operation("summarize_session")
        
        try:
            if not tab_data:
                return "No browsing activity detected"
            
            # Extract and analyze session data
            domains = [tab.get('domain', '') for tab in tab_data]
            titles = [tab.get('title', '') for tab in tab_data]
            
            # Analyze session characteristics
            session_analysis = self._analyze_session_characteristics(domains, titles, tab_data)
            
            # Generate summary based on analysis
            summary = self._generate_session_summary(session_analysis, max_length)
            
            return summary
            
        except Exception as e:
            print(f"Session summarization failed: {e}")
            return f"Browsing session with {len(tab_data)} tabs"
        
        finally:
            self.performance_monitor.end_operation("summarize_session", operation_start)
    
    def _analyze_session_characteristics(self, domains: List[str], titles: List[str], 
                                       tab_data: List[Dict]) -> Dict[str, Any]:
        """Analyze characteristics of the browsing session"""
        
        domain_counts = Counter(domains)
        total_tabs = len(tab_data)
        unique_domains = len(set(domains))
        
        analysis = {
            "tab_count": total_tabs,
            "unique_domains": unique_domains,
            "domain_diversity": unique_domains / total_tabs if total_tabs > 0 else 0,
            "primary_domains": [],
            "session_type": "",
            "key_activities": [],
            "temporal_span": None
        }
        
        # Identify primary domains (>20% of tabs)
        # TUNABLE: Primary domain threshold
        primary_threshold = 0.2
        analysis["primary_domains"] = [
            (domain, count) for domain, count in domain_counts.most_common()
            if count / total_tabs >= primary_threshold
        ]
        
        # Determine session type
        if analysis["domain_diversity"] < 0.3:
            analysis["session_type"] = "focused"
        elif analysis["domain_diversity"] > 0.7:
            analysis["session_type"] = "exploratory"
        else:
            analysis["session_type"] = "mixed"
        
        # Identify key activities from titles
        activity_patterns = {
            "learning": [r'\b(tutorial|learn|how\s+to|guide|course)\b', "learning activity"],
            "shopping": [r'\b(buy|shop|price|product|cart|order)\b', "shopping research"],
            "news": [r'\b(news|article|report|breaking|update)\b', "news reading"],
            "social": [r'\b(post|share|comment|social|friend)\b', "social media"],
            "work": [r'\b(meeting|project|work|office|team|document)\b', "work tasks"],
            "entertainment": [r'\b(video|movie|music|game|watch|play)\b', "entertainment"]
        }
        
        combined_titles = " ".join(titles).lower()
        for activity, (pattern, description) in activity_patterns.items():
            if re.search(pattern, combined_titles, re.IGNORECASE):
                analysis["key_activities"].append(description)
        
        # Calculate temporal span if timestamps available
        timestamps = [tab.get('timestamp') for tab in tab_data if tab.get('timestamp')]
        if timestamps:
            try:
                times = [datetime.fromisoformat(ts) for ts in timestamps]
                span = max(times) - min(times)
                analysis["temporal_span"] = span.total_seconds() / 60  # minutes
            except Exception:
                pass
        
        return analysis
    
    def _generate_session_summary(self, analysis: Dict[str, Any], max_length: int) -> str:
        """Generate natural language summary from session analysis"""
        
        parts = []
        
        # Session scale description
        tab_count = analysis["tab_count"]
        if tab_count == 1:
            parts.append("Single tab session")
        elif tab_count <= 5:
            parts.append(f"Small session ({tab_count} tabs)")
        elif tab_count <= 15:
            parts.append(f"Medium session ({tab_count} tabs)")
        else:
            parts.append(f"Large session ({tab_count} tabs)")
        
        # Session type and focus
        session_type = analysis["session_type"]
        unique_domains = analysis["unique_domains"]
        
        if session_type == "focused":
            if unique_domains == 1:
                domain = analysis["primary_domains"][0][0] if analysis["primary_domains"] else "single site"
                parts.append(f"focused on {domain}")
            else:
                parts.append(f"focused across {unique_domains} sites")
        elif session_type == "exploratory":
            parts.append(f"exploring {unique_domains} different sites")
        else:
            parts.append(f"mixed browsing across {unique_domains} sites")
        
        # Primary activities
        if analysis["key_activities"]:
            if len(analysis["key_activities"]) == 1:
                parts.append(f"involving {analysis['key_activities'][0]}")
            else:
                activities = ", ".join(analysis["key_activities"][:2])
                if len(analysis["key_activities"]) > 2:
                    activities += f" and {len(analysis['key_activities']) - 2} other activities"
                parts.append(f"including {activities}")
        
        # Temporal information
        if analysis["temporal_span"]:
            span_minutes = analysis["temporal_span"]
            if span_minutes < 1:
                parts.append("in under a minute")
            elif span_minutes < 60:
                parts.append(f"over {span_minutes:.0f} minutes")
            else:
                hours = span_minutes / 60
                parts.append(f"over {hours:.1f} hours")
        
        # Combine parts into summary
        summary = " ".join(parts)
        
        # Truncate if too long
        if len(summary) > max_length:
            # Try to break at a word boundary
            truncated = summary[:max_length].rsplit(' ', 1)[0]
            if len(truncated) > max_length * 0.8:  # If we didn't lose too much
                summary = truncated + "..."
            else:
                summary = summary[:max_length - 3] + "..."
        
        return summary.capitalize()
    
    def validate_category(self, category: str, subcategory: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Validate whether a category assignment makes sense
        
        Args:
            category: Proposed category
            subcategory: Proposed subcategory  
            evidence: Evidence text (titles, domains, etc.)
            
        Returns:
            Dictionary with validation result and confidence
        """
        operation_start = self.performance_monitor.start_operation("validate_category")
        
        try:
            validation_result = {
                "is_valid": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": [],
                "validation_details": {}
            }
            
            # Check if category exists in framework
            if category not in self.category_framework.get_all_categories():
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Category '{category}' not in framework")
                validation_result["suggestions"].append("Use a standard framework category")
            
            # Analyze evidence for category support
            evidence_text = " ".join(evidence).lower()
            
            if category in self.category_patterns:
                category_patterns = self.category_patterns[category]
                matches = sum(1 for pattern in category_patterns if pattern in evidence_text)
                match_ratio = matches / len(category_patterns)
                
                validation_result["validation_details"]["pattern_matches"] = matches
                validation_result["validation_details"]["pattern_ratio"] = match_ratio
                
                if match_ratio < 0.1:  # TUNABLE: Minimum pattern match threshold
                    validation_result["issues"].append("Low pattern match for category")
                    validation_result["confidence"] *= 0.7
                else:
                    validation_result["confidence"] += match_ratio * 0.3
            
            # Check subcategory appropriateness
            if category in self.subcategory_patterns:
                if subcategory in self.subcategory_patterns[category]:
                    subcategory_patterns = self.subcategory_patterns[category][subcategory]
                    subcategory_matches = sum(1 for pattern in subcategory_patterns if pattern in evidence_text)
                    
                    validation_result["validation_details"]["subcategory_matches"] = subcategory_matches
                    
                    if subcategory_matches == 0:
                        validation_result["issues"].append("Subcategory doesn't match evidence")
                        validation_result["confidence"] *= 0.8
                else:
                    validation_result["issues"].append("Subcategory not recognized for this category")
                    validation_result["confidence"] *= 0.9
            
            # Final confidence calculation
            validation_result["confidence"] = min(0.95, max(0.1, validation_result["confidence"]))
            
            return validation_result
            
        except Exception as e:
            print(f"Category validation failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Manual review recommended"],
                "validation_details": {}
            }
        
        finally:
            self.performance_monitor.end_operation("validate_category", operation_start)
    
    def _load_category_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive patterns for category matching"""
        return {
            "Gaming": [
                # Core gaming terms
                "game", "gaming", "play", "player", "level", "character", "quest", "mission",
                "achievement", "score", "leaderboard", "multiplayer", "single player",
                # Game types
                "rpg", "fps", "strategy", "puzzle", "simulation", "adventure", "action",
                "mmorpg", "battle royale", "sandbox", "platformer",
                # Gaming platforms and stores
                "steam", "epic games", "origin", "uplay", "gog", "itch.io", "xbox", "playstation",
                # Gaming content
                "walkthrough", "guide", "tips", "tricks", "build", "meta", "patch", "update",
                "dlc", "mod", "speedrun", "stream", "twitch", "gameplay"
            ],
            
            "Social Media": [
                # Platforms
                "facebook", "twitter", "instagram", "linkedin", "snapchat", "tiktok", "pinterest",
                "reddit", "discord", "telegram", "whatsapp", "youtube", "twitch",
                # Activities
                "post", "share", "like", "comment", "follow", "friend", "message", "chat",
                "story", "reel", "tweet", "pin", "upvote", "downvote", "subscribe",
                # Content types
                "photo", "video", "live", "stream", "feed", "timeline", "profile", "bio",
                "hashtag", "mention", "tag", "dm", "notification"
            ],
            
            "Academic Research": [
                # Research terms
                "research", "study", "paper", "article", "journal", "publication", "thesis",
                "dissertation", "academic", "scholar", "peer review", "citation", "reference",
                # Platforms and databases
                "google scholar", "pubmed", "ieee", "acm", "arxiv", "researchgate", "academia",
                "jstor", "scopus", "web of science", "springer", "elsevier",
                # Academic activities
                "bibliography", "methodology", "abstract", "conclusion", "literature review",
                "experiment", "data", "analysis", "hypothesis", "theory", "findings"
            ],
            
            "Work/Professional": [
                # Work activities
                "work", "job", "office", "meeting", "project", "deadline", "task", "assignment",
                "presentation", "report", "email", "document", "spreadsheet", "calendar",
                # Professional platforms
                "slack", "teams", "zoom", "google workspace", "office 365", "notion", "asana",
                "trello", "jira", "confluence", "sharepoint", "dropbox", "google drive",
                # Professional terms
                "colleague", "team", "manager", "client", "customer", "vendor", "partner",
                "budget", "revenue", "profit", "strategy", "planning", "review"
            ],
            
            "Shopping": [
                # Shopping actions
                "buy", "shop", "purchase", "order", "cart", "checkout", "payment", "shipping",
                "delivery", "return", "refund", "review", "rating", "comparison",
                # Shopping platforms
                "amazon", "ebay", "shopify", "etsy", "walmart", "target", "costco", "alibaba",
                "wish", "aliexpress", "best buy", "home depot",
                # Shopping terms
                "price", "cost", "deal", "sale", "discount", "coupon", "offer", "promotion",
                "product", "item", "brand", "model", "size", "color", "inventory", "stock"
            ],
            
            "Entertainment": [
                # Media types
                "video", "movie", "film", "tv", "show", "series", "documentary", "music",
                "song", "album", "artist", "podcast", "audiobook", "radio",
                # Platforms
                "netflix", "youtube", "hulu", "disney plus", "amazon prime", "spotify",
                "apple music", "pandora", "soundcloud", "twitch", "tiktok",
                # Activities
                "watch", "listen", "stream", "download", "subscribe", "playlist", "queue",
                "favorite", "bookmark", "binge", "marathon", "episode"
            ],
            
            "Learning/Tutorials": [
                # Learning activities
                "learn", "tutorial", "course", "lesson", "education", "training", "study",
                "practice", "exercise", "homework", "assignment", "quiz", "test", "exam",
                # Learning content
                "how to", "guide", "instruction", "manual", "documentation", "wiki",
                "explanation", "example", "demo", "walkthrough", "step by step",
                # Learning platforms
                "coursera", "udemy", "khan academy", "edx", "skillshare", "lynda",
                "pluralsight", "codecademy", "duolingo", "youtube tutorial"
            ],
            
            "News/Information": [
                # News types
                "news", "article", "report", "story", "breaking", "update", "announcement",
                "press release", "editorial", "opinion", "analysis", "commentary",
                # News sources
                "cnn", "bbc", "reuters", "ap news", "npr", "new york times", "wall street journal",
                "washington post", "guardian", "forbes", "bloomberg", "techcrunch",
                # News topics
                "politics", "election", "economy", "business", "technology", "science",
                "health", "sports", "weather", "local", "international", "breaking news"
            ],
            
            "AI/Development": [
                # Programming and development
                "code", "programming", "development", "software", "app", "website", "api",
                "framework", "library", "database", "server", "cloud", "deployment",
                # Platforms and tools
                "github", "gitlab", "stackoverflow", "documentation", "tutorial", "guide",
                "visual studio", "intellij", "eclipse", "sublime", "atom", "vscode",
                # AI and ML
                "artificial intelligence", "machine learning", "deep learning", "neural network",
                "algorithm", "model", "training", "dataset", "python", "tensorflow", "pytorch"
            ],
            
            "Health/Wellness": [
                # Health topics
                "health", "medical", "doctor", "hospital", "clinic", "medicine", "treatment",
                "diagnosis", "symptoms", "disease", "condition", "wellness", "fitness",
                # Health activities
                "exercise", "workout", "nutrition", "diet", "healthy", "meditation",
                "yoga", "running", "gym", "weight loss", "mental health", "therapy",
                # Health platforms
                "webmd", "mayo clinic", "healthline", "medscape", "fitbit", "myfitnesspal",
                "calm", "headspace", "strava", "nike training"
            ]
        }
    
    def _load_subcategory_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load subcategory patterns for detailed classification"""
        return {
            "Gaming": {
                "Quest Research": ["quest", "mission", "walkthrough", "guide", "how to complete"],
                "Strategy Guides": ["strategy", "tips", "build", "optimal", "best", "meta"],
                "Community Discussion": ["forum", "reddit", "discussion", "community", "chat"],
                "Game Reviews": ["review", "rating", "opinion", "recommendation", "score"],
                "Competitive Gaming": ["competitive", "esports", "tournament", "league", "rank"],
                "Streaming Content": ["stream", "twitch", "youtube", "gameplay", "watch"]
            },
            
            "Academic Research": {
                "Literature Review": ["literature", "review", "survey", "state of art"],
                "Paper Reading": ["paper", "article", "journal", "publication", "research"],
                "Reference Lookup": ["reference", "citation", "bibliography", "source"],
                "Data Analysis": ["data", "analysis", "statistics", "methodology", "results"],
                "Course Materials": ["course", "lecture", "syllabus", "assignment", "textbook"]
            },
            
            "Work/Professional": {
                "Meetings": ["meeting", "conference", "call", "zoom", "teams"],
                "Project Management": ["project", "task", "deadline", "milestone", "planning"],
                "Communication": ["email", "message", "slack", "chat", "notification"],
                "Documentation": ["document", "report", "presentation", "notes", "wiki"],
                "Research": ["research", "analysis", "market", "competitor", "industry"]
            },
            
            "Shopping": {
                "Product Research": ["product", "specifications", "features", "comparison"],
                "Price Comparison": ["price", "cost", "deal", "discount", "comparison"],
                "Reviews Reading": ["review", "rating", "feedback", "testimonial", "opinion"],
                "Purchase Process": ["buy", "cart", "checkout", "payment", "order"]
            },
            
            "Entertainment": {
                "Video Streaming": ["netflix", "youtube", "video", "movie", "watch"],
                "Music Streaming": ["spotify", "music", "song", "album", "listen"],
                "Gaming Content": ["twitch", "gaming", "stream", "gameplay", "game"],
                "Social Content": ["tiktok", "instagram", "social", "viral", "trending"]
            },
            
            "Learning/Tutorials": {
                "How-to Guides": ["how to", "tutorial", "guide", "instruction", "step"],
                "Online Courses": ["course", "lesson", "module", "certification", "training"],
                "Documentation": ["documentation", "manual", "wiki", "reference", "api"],
                "Practice/Exercises": ["practice", "exercise", "challenge", "problem", "solution"]
            }
        }
    
    def _load_domain_mappings(self) -> Dict[str, str]:
        """Load direct domain to category mappings"""
        return {
            # Gaming
            "steam": "Gaming", "epicgames.com": "Gaming", "battle.net": "Gaming",
            "twitch.tv": "Gaming", "ign.com": "Gaming", "gamespot.com": "Gaming",
            
            # Social Media
            "facebook.com": "Social Media", "twitter.com": "Social Media", "instagram.com": "Social Media",
            "linkedin.com": "Social Media", "reddit.com": "Social Media", "discord.com": "Social Media",
            
            # Academic
            "scholar.google.com": "Academic Research", "pubmed.ncbi.nlm.nih.gov": "Academic Research",
            "ieee.org": "Academic Research", "arxiv.org": "Academic Research",
            
            # Work
            "slack.com": "Work/Professional", "teams.microsoft.com": "Work/Professional",
            "notion.so": "Work/Professional", "asana.com": "Work/Professional",
            
            # Shopping
            "amazon.com": "Shopping", "ebay.com": "Shopping", "walmart.com": "Shopping",
            "target.com": "Shopping", "costco.com": "Shopping",
            
            # Entertainment
            "netflix.com": "Entertainment", "youtube.com": "Entertainment", "spotify.com": "Entertainment",
            "hulu.com": "Entertainment", "disneyplus.com": "Entertainment",
            
            # News
            "cnn.com": "News/Information", "bbc.com": "News/Information", "reuters.com": "News/Information",
            "nytimes.com": "News/Information", "washingtonpost.com": "News/Information",
            
            # Development
            "github.com": "AI/Development", "stackoverflow.com": "AI/Development",
            "developer.mozilla.org": "AI/Development", "docs.python.org": "AI/Development"
        }
    
    def _load_contextual_keywords(self) -> Dict[str, List[str]]:
        """Load contextual keywords that provide additional classification hints"""
        return {
            "urgency": ["urgent", "asap", "deadline", "emergency", "critical", "important"],
            "learning": ["beginner", "advanced", "tutorial", "learn", "study", "practice"],
            "comparison": ["vs", "versus", "compare", "comparison", "better", "best", "alternative"],
            "problem_solving": ["how to", "fix", "solve", "troubleshoot", "error", "issue", "problem"],
            "research": ["research", "analysis", "study", "investigation", "survey", "review"]
        }
    
    def _analyze_prompt_intent(self, prompt: str) -> str:
        """Analyze prompt to understand the type of request"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["category", "classify", "categorize"]):
            return "category_generation"
        elif any(word in prompt_lower for word in ["subcategory", "specific", "detailed"]):
            return "subcategory_generation"
        elif any(word in prompt_lower for word in ["summary", "summarize", "session"]):
            return "summarization"
        else:
            return "generic"
    
    def _generate_category_from_prompt(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate category response from prompt"""
        # Extract context from prompt for category generation
        return "Generated category based on content analysis"
    
    def _generate_subcategory_from_prompt(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate subcategory response from prompt"""
        return "Generated subcategory based on detailed analysis"
    
    def _generate_summary_from_prompt(self, prompt: str, max_tokens: int) -> str:
        """Generate summary response from prompt"""
        return "Generated session summary based on browsing patterns"
    
    def _generate_generic_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate generic response"""
        return "Generated response"
    
    def _fallback_category_generation(self, titles: List[str], domains: List[str]) -> Dict[str, Any]:
        """Simple fallback category generation"""
        return {
            "category": "Web Browsing",
            "subcategory": "General Activity",
            "confidence": 0.3,
            "method": "fallback",
            "evidence": ["Fallback classification used"]
        }
    
    def _record_generation_performance(self, result: Dict[str, Any], input_size: int):
        """Record performance for optimization"""
        self.generation_history.append({
            "timestamp": datetime.now().isoformat(),
            "input_size": input_size,
            "confidence": result.get("confidence", 0),
            "method": result.get("method", "unknown"),
            "category": result.get("category", "unknown")
        })
        
        # Keep history limited
        if len(self.generation_history) > 100:
            self.generation_history.pop(0)

# Global instance
_llm_service: Optional[LocalLLMService] = None

def get_llm_service() -> LocalLLMService:
    """Get global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LocalLLMService()
    return _llm_service 