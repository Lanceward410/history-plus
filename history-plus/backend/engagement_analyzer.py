import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

class EngagementAnalyzer:
    def __init__(self):
        # TUNABLE PARAMETERS: Page type engagement weights
        # These weights determine how different metrics contribute to engagement scores
        # Adjust based on user feedback and engagement correlation studies
        self.page_type_weights = {
            'video': {
                'time': 0.7,        # TUNE: Video engagement heavily depends on watch time
                'focus': 0.2,       # TUNE: Focus time matters but less than total time
                'interactions': 0.1  # TUNE: Minimal interactions expected for videos
            },
            'article': {
                'scroll': 0.4,      # TUNE: Reading requires scrolling through content
                'time': 0.4,        # TUNE: Time spent correlates with reading depth
                'focus': 0.2        # TUNE: Focus time filters out background tabs
            },
            'spa': {
                'interactions': 0.6, # TUNE: SPAs require user interaction to be meaningful
                'time': 0.3,        # TUNE: Time matters but interactions are key
                'focus': 0.1        # TUNE: Focus less important if actively interacting
            },
            'search': {
                'interactions': 0.8, # TUNE: Search success measured by clicks/typing
                'time': 0.2         # TUNE: Quick searches can be highly effective
            },
            'social': {
                'scroll': 0.3,      # TUNE: Social feeds involve scrolling
                'interactions': 0.4, # TUNE: Likes, comments, shares indicate engagement
                'time': 0.3         # TUNE: Social browsing can be quick but meaningful
            }
        }
        
        # TUNABLE PARAMETERS: Session classification thresholds
        # These define boundaries between different session types
        self.session_thresholds = {
            'deep_research_min_engagement': 0.7,    # TUNE: Minimum avg engagement for deep research
            'deep_research_min_pages': 3,           # TUNE: Minimum pages for research session
            'casual_browsing_max_engagement': 0.4,  # TUNE: Maximum engagement for casual browsing
            'casual_browsing_min_pages': 10,        # TUNE: Many pages = casual browsing
            'focus_session_min_time': 300,          # TUNE: 5 minutes minimum for focused work
            'meaningful_engagement_threshold': 0.6,  # TUNE: Score above which engagement is "meaningful"
            'high_confidence_threshold': 0.7        # TUNE: Confidence threshold for reliable data
        }
        
        # TUNABLE PARAMETERS: Time-based analysis windows
        # These control how we aggregate and analyze temporal patterns
        self.time_windows = {
            'session_gap_minutes': 15,              # TUNE: Gap between sessions (15 min default)
            'focus_session_min_minutes': 5,         # TUNE: Minimum duration for focus session
            'daily_analysis_hours': 24,             # TUNE: Hours to include in daily analysis
            'weekly_pattern_days': 7                # TUNE: Days for weekly pattern analysis
        }

    def analyze_session_engagement(self, session_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze engagement patterns within a browsing session
        
        Args:
            session_entries: List of history entries in chronological order
            
        Returns:
            Dictionary with session analysis including type, meaningful pages, and focus metrics
        """
        if not session_entries:
            return self._empty_session_analysis()
        
        # Calculate individual page engagement scores
        page_scores = []
        high_engagement_pages = []
        total_meaningful_time = 0
        page_types_distribution = {}
        
        for entry in session_entries:
            engagement = entry.get('engagement', {})
            score = engagement.get('contextualScore', 0)
            confidence = engagement.get('confidence', 0)
            page_type = engagement.get('pageType', 'webpage')
            
            page_scores.append(score)
            
            # Track page type distribution for session classification
            page_types_distribution[page_type] = page_types_distribution.get(page_type, 0) + 1
            
            # TUNABLE LOGIC: What constitutes "high engagement"
            # Currently: score > 0.6 AND confidence > 0.7
            # TUNE: Adjust these thresholds based on user feedback
            if (score > self.session_thresholds['meaningful_engagement_threshold'] and 
                confidence > self.session_thresholds['high_confidence_threshold']):
                high_engagement_pages.append(entry)
                total_meaningful_time += engagement.get('focusTime', 0)
        
        # Calculate session-level metrics
        avg_engagement = np.mean(page_scores) if page_scores else 0
        engagement_variance = np.var(page_scores) if len(page_scores) > 1 else 0
        
        # TUNABLE LOGIC: Session type classification
        session_type = self._classify_session_type(
            session_entries, 
            avg_engagement, 
            page_types_distribution,
            len(high_engagement_pages)
        )
        
        return {
            'type': session_type,
            'meaningful_pages': high_engagement_pages,
            'total_pages': len(session_entries),
            'focus_score': len(high_engagement_pages) / len(session_entries) if session_entries else 0,
            'avg_engagement': avg_engagement,
            'engagement_consistency': 1 - min(1, engagement_variance),  # TUNE: How we measure consistency
            'total_meaningful_time': total_meaningful_time,
            'dominant_page_type': max(page_types_distribution, key=page_types_distribution.get) if page_types_distribution else 'unknown',
            'exploration_breadth': len(set(entry.get('domain', '') for entry in session_entries))
        }

    def _classify_session_type(self, entries: List[Dict], avg_engagement: float, 
                              page_types: Dict[str, int], meaningful_count: int) -> str:
        """
        Classify the type of browsing session based on engagement patterns
        
        TUNABLE LOGIC: The entire classification logic can be tuned
        Current classification prioritizes:
        1. High engagement + multiple pages = deep research
        2. Article-heavy + many pages = research browsing  
        3. Video-heavy = entertainment
        4. Many pages + low engagement = casual browsing
        5. High engagement + few pages = task focused
        """
        total_pages = len(entries)
        thresholds = self.session_thresholds
        
        # Deep research: High engagement across multiple pages
        # TUNE: Both engagement threshold and minimum pages
        if (avg_engagement > thresholds['deep_research_min_engagement'] and 
            total_pages >= thresholds['deep_research_min_pages']):
            return 'deep_research'
        
        # Research browsing: Article-heavy with moderate engagement
        # TUNE: What percentage of articles constitutes "research"
        article_ratio = page_types.get('article', 0) / total_pages if total_pages > 0 else 0
        if article_ratio > 0.3 and total_pages > 5:  # TUNE: 30% articles threshold
            return 'research_browsing'
        
        # Entertainment: Video-heavy sessions
        # TUNE: Video ratio threshold for entertainment classification
        video_ratio = page_types.get('video', 0) / total_pages if total_pages > 0 else 0
        if video_ratio > 0.4:  # TUNE: 40% videos = entertainment
            return 'entertainment'
        
        # Casual browsing: Many pages with low engagement
        # TUNE: Page count and engagement thresholds
        if (total_pages > thresholds['casual_browsing_min_pages'] and 
            avg_engagement < thresholds['casual_browsing_max_engagement']):
            return 'casual_browsing'
        
        # Task focused: High engagement but fewer pages
        # TUNE: What constitutes "focused" vs "research"
        if meaningful_count > 0 and total_pages < 8:  # TUNE: Page count threshold
            return 'task_focused'
        
        return 'general_browsing'

    def analyze_temporal_patterns(self, history_entries: List[Dict[str, Any]], 
                                 days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze browsing patterns over time to identify habits and trends
        
        TUNABLE PARAMETERS:
        - Time windows for pattern detection
        - Minimum session lengths
        - Peak activity thresholds
        """
        if not history_entries:
            return {}
        
        # Group entries by time periods
        hourly_activity = [0] * 24
        daily_activity = {}
        weekly_patterns = {i: 0 for i in range(7)}  # 0=Monday, 6=Sunday
        
        # TUNABLE PARAMETERS: Activity scoring weights
        activity_weights = {
            'page_visit': 1.0,          # TUNE: Base weight for page visits
            'high_engagement': 3.0,     # TUNE: Weight for highly engaged pages
            'focus_time_bonus': 0.1     # TUNE: Bonus per minute of focus time
        }
        
        for entry in history_entries:
            timestamp = entry.get('timestamp', 0)
            if timestamp == 0:
                continue
                
            dt = datetime.fromtimestamp(timestamp / 1000)
            hour = dt.hour
            date_key = dt.date().isoformat()
            weekday = dt.weekday()
            
            # Calculate activity score for this entry
            engagement = entry.get('engagement', {})
            base_score = activity_weights['page_visit']
            
            # TUNABLE LOGIC: What makes an entry "high value"
            if engagement.get('contextualScore', 0) > 0.7:
                base_score *= activity_weights['high_engagement']
            
            focus_minutes = engagement.get('focusTime', 0) / (60 * 1000)  # Convert to minutes
            base_score += focus_minutes * activity_weights['focus_time_bonus']
            
            # Accumulate activity scores
            hourly_activity[hour] += base_score
            daily_activity[date_key] = daily_activity.get(date_key, 0) + base_score
            weekly_patterns[weekday] += base_score
        
        # TUNABLE ANALYSIS: Peak detection thresholds
        avg_hourly = np.mean(hourly_activity)
        peak_hours = [i for i, activity in enumerate(hourly_activity) 
                     if activity > avg_hourly * 1.5]  # TUNE: 1.5x average = peak
        
        return {
            'hourly_distribution': hourly_activity,
            'daily_totals': daily_activity,
            'weekly_patterns': weekly_patterns,
            'peak_hours': peak_hours,
            'most_active_day': max(weekly_patterns, key=weekly_patterns.get),
            'activity_variance': np.var(list(hourly_activity)),
            'total_analyzed_days': len(daily_activity)
        }

    def generate_engagement_insights(self, processed_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate high-level insights about user engagement patterns
        
        TUNABLE INSIGHTS: The thresholds and criteria for generating insights
        can be adjusted based on what proves most valuable to users
        """
        if not processed_sessions:
            return {}
        
        # Categorize sessions by type and quality
        session_types = {}
        high_quality_sessions = []
        total_meaningful_time = 0
        
        for session in processed_sessions:
            session_type = session.get('type', 'unknown')
            session_types[session_type] = session_types.get(session_type, 0) + 1
            
            # TUNABLE CRITERIA: What makes a session "high quality"
            # Currently: focus_score > 0.5 AND meaningful_time > 10 minutes
            focus_score = session.get('focus_score', 0)
            meaningful_time = session.get('total_meaningful_time', 0)
            
            if (focus_score > 0.5 and                      # TUNE: Focus threshold
                meaningful_time > 10 * 60 * 1000):        # TUNE: 10 minutes minimum
                high_quality_sessions.append(session)
            
            total_meaningful_time += meaningful_time
        
        # Calculate productivity metrics
        total_sessions = len(processed_sessions)
        productivity_score = len(high_quality_sessions) / total_sessions if total_sessions > 0 else 0
        
        # TUNABLE INSIGHTS: Trend detection sensitivity
        # Determine if user is becoming more or less focused over time
        if len(processed_sessions) >= 10:  # TUNE: Minimum sessions for trend analysis
            recent_sessions = processed_sessions[-5:]  # TUNE: Number of recent sessions
            older_sessions = processed_sessions[-10:-5]  # TUNE: Comparison window
            
            recent_avg_focus = np.mean([s.get('focus_score', 0) for s in recent_sessions])
            older_avg_focus = np.mean([s.get('focus_score', 0) for s in older_sessions])
            
            focus_trend = 'improving' if recent_avg_focus > older_avg_focus * 1.1 else \
                         'declining' if recent_avg_focus < older_avg_focus * 0.9 else 'stable'
        else:
            focus_trend = 'insufficient_data'
        
        return {
            'session_type_distribution': session_types,
            'high_quality_sessions': len(high_quality_sessions),
            'productivity_score': productivity_score,
            'total_meaningful_hours': total_meaningful_time / (60 * 60 * 1000),
            'focus_trend': focus_trend,
            'avg_session_quality': np.mean([s.get('focus_score', 0) for s in processed_sessions]),
            'most_common_session_type': max(session_types, key=session_types.get) if session_types else 'unknown'
        }

    def _empty_session_analysis(self) -> Dict[str, Any]:
        """Return empty session analysis structure"""
        return {
            'type': 'unknown',
            'meaningful_pages': [],
            'total_pages': 0,
            'focus_score': 0,
            'avg_engagement': 0,
            'engagement_consistency': 0,
            'total_meaningful_time': 0,
            'dominant_page_type': 'unknown',
            'exploration_breadth': 0
        } 