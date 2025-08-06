import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

from models.data_models import HistoryEntry, BrowsingSession, EngagementData, ProcessingResult

class DataProcessor:
    """Core data processing service for History Plus"""
    
    def __init__(self, config):
        self.config = config
        self.session_gap_minutes = config.SESSION_GAP_MINUTES
        
    def process_raw_history(self, raw_entries: List[Dict[str, Any]]) -> ProcessingResult:
        """
        Main processing pipeline for raw history data
        
        Args:
            raw_entries: List of raw history entries from Chrome extension
            
        Returns:
            ProcessingResult with processed sessions, insights, and visualization data
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse and validate raw entries
            print(f"Processing {len(raw_entries)} raw entries...")
            history_entries = self._parse_raw_entries(raw_entries)
            print(f"Successfully parsed {len(history_entries)} entries")
            
            # Step 2: Detect browsing sessions
            print("Detecting browsing sessions...")
            sessions = self._detect_sessions(history_entries)
            print(f"Detected {len(sessions)} browsing sessions")
            
            # Step 3: Analyze sessions
            print("Analyzing session characteristics...")
            analyzed_sessions = self._analyze_sessions(sessions)
            
            # Step 4: Generate visualization data
            print("Generating visualization data...")
            timeline_data = self._generate_timeline_data(analyzed_sessions)
            domain_distribution = self._generate_domain_distribution(history_entries)
            engagement_heatmap = self._generate_engagement_heatmap(history_entries)
            
            # Step 5: Generate insights
            print("Generating insights...")
            insights = self._generate_insights(analyzed_sessions, history_entries)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                total_entries=len(raw_entries),
                processed_entries=len(history_entries),
                processing_time_seconds=processing_time,
                sessions=analyzed_sessions,
                timeline_data=timeline_data,
                domain_distribution=domain_distribution,
                engagement_heatmap=engagement_heatmap,
                insights=insights
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Error during processing: {e}")
            
            return ProcessingResult(
                total_entries=len(raw_entries),
                processed_entries=0,
                processing_time_seconds=processing_time,
                errors=[str(e)]
            )
    
    def _parse_raw_entries(self, raw_entries: List[Dict[str, Any]]) -> List[HistoryEntry]:
        """Parse raw entries into structured HistoryEntry objects"""
        parsed_entries = []
        
        for raw_entry in raw_entries:
            try:
                # Extract engagement data if present
                engagement_data = None
                if 'engagement' in raw_entry:
                    eng_raw = raw_entry['engagement']
                    engagement_data = EngagementData(
                        time_on_page=eng_raw.get('timeOnPage', 0),
                        focus_time=eng_raw.get('focusTime', 0),
                        active_time=eng_raw.get('activeTime', 0),
                        idle_time=eng_raw.get('idleTime', 0),
                        background_time=eng_raw.get('backgroundTime', 0),
                        scroll_depth=eng_raw.get('scrollDepth', 0.0),
                        mouse_movements=eng_raw.get('mouseMovements', 0),
                        key_presses=eng_raw.get('keyPresses', 0),
                        page_type=eng_raw.get('pageType', 'webpage'),
                        content_length=eng_raw.get('contentLength', 0),
                        contextual_score=eng_raw.get('contextualScore', 0.0),
                        confidence=eng_raw.get('confidence', 0.0),
                        focus_ratio=eng_raw.get('focusRatio', 0.0),
                        active_ratio=eng_raw.get('activeRatio', 0.0),
                        engagement_ratio=eng_raw.get('engagementRatio', 0.0),
                        time_breakdown=eng_raw.get('timeBreakdown', {})
                    )
                
                # Create history entry
                entry = HistoryEntry(
                    url=raw_entry.get('url', ''),
                    title=raw_entry.get('title', ''),
                    domain=raw_entry.get('domain', ''),
                    timestamp=raw_entry.get('timestamp', 0),
                    visit_count=raw_entry.get('visitCount', 1),
                    session_id=raw_entry.get('sessionId'),
                    entry_type=raw_entry.get('type', 'history_entry'),
                    engagement=engagement_data
                )
                
                parsed_entries.append(entry)
                
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue
        
        return parsed_entries
    
    def _detect_sessions(self, entries: List[HistoryEntry]) -> List[BrowsingSession]:
        """
        Detect browsing sessions based on temporal gaps
        
        TUNABLE PARAMETERS:
        - session_gap_minutes: Time gap that defines session boundaries
        """
        if not entries:
            return []
        
        # Sort entries by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        
        sessions = []
        current_session_entries = [sorted_entries[0]]
        session_start = sorted_entries[0].timestamp
        
        for entry in sorted_entries[1:]:
            # Calculate time gap in minutes
            time_gap_ms = entry.timestamp - current_session_entries[-1].timestamp
            time_gap_minutes = time_gap_ms / 1000 / 60
            
            if time_gap_minutes > self.session_gap_minutes:
                # End current session and start new one
                if len(current_session_entries) >= 1:  # Minimum session size
                    session = BrowsingSession(
                        id=f"session_{session_start}",
                        start_time=current_session_entries[0].timestamp,
                        end_time=current_session_entries[-1].timestamp,
                        entries=current_session_entries.copy()
                    )
                    sessions.append(session)
                
                # Start new session
                current_session_entries = [entry]
                session_start = entry.timestamp
            else:
                current_session_entries.append(entry)
        
        # Don't forget the last session
        if len(current_session_entries) >= 1:
            session = BrowsingSession(
                id=f"session_{session_start}",
                start_time=current_session_entries[0].timestamp,
                end_time=current_session_entries[-1].timestamp,
                entries=current_session_entries.copy()
            )
            sessions.append(session)
        
        return sessions
    
    def _analyze_sessions(self, sessions: List[BrowsingSession]) -> List[BrowsingSession]:
        """Analyze and categorize browsing sessions"""
        
        for session in sessions:
            # Categorize session type
            session.session_type = self._classify_session_type(session)
            
            # Generate basic topic summary
            session.topic_summary = self._generate_basic_topic_summary(session)
            
            # Set category based on dominant domains/page types
            session.category = self._determine_session_category(session)
        
        return sessions
    
    def _classify_session_type(self, session: BrowsingSession) -> str:
        """
        Classify session type based on engagement patterns
        
        TUNABLE PARAMETERS:
        - Thresholds for different session types
        """
        page_types = [entry.engagement.page_type for entry in session.entries if entry.engagement]
        avg_engagement = session.avg_engagement
        total_pages = session.total_pages
        unique_domains = session.unique_domains
        
        # TUNABLE: Classification thresholds
        if avg_engagement > 0.7 and total_pages >= 3:
            return 'deep_research'
        elif 'video' in page_types and page_types.count('video') / len(page_types) > 0.6:
            return 'entertainment'
        elif unique_domains > total_pages * 0.8:
            return 'exploration'
        elif avg_engagement < 0.3 and total_pages > 10:
            return 'casual_browsing'
        else:
            return 'focused_work'
    
    def _generate_basic_topic_summary(self, session: BrowsingSession) -> str:
        """Generate basic topic summary without LLM"""
        if not session.entries:
            return "Empty session"
        
        # Get most common domain
        domains = [entry.domain for entry in session.entries]
        most_common_domain = Counter(domains).most_common(1)[0][0]
        
        # Get page types
        page_types = [
            entry.engagement.page_type 
            for entry in session.entries 
            if entry.engagement
        ]
        most_common_type = Counter(page_types).most_common(1)[0][0] if page_types else 'web'
        
        return f"{session.session_type.replace('_', ' ').title()} session on {most_common_domain} ({most_common_type} content)"
    
    def _determine_session_category(self, session: BrowsingSession) -> str:
        """Determine high-level category for session"""
        domains = [entry.domain for entry in session.entries]
        
        # Simple domain-based categorization
        if any(domain in ['github.com', 'stackoverflow.com', 'docs.python.org'] for domain in domains):
            return 'development'
        elif any(domain in ['youtube.com', 'netflix.com', 'spotify.com'] for domain in domains):
            return 'entertainment'
        elif any(domain in ['news.ycombinator.com', 'reddit.com', 'twitter.com'] for domain in domains):
            return 'social'
        else:
            return 'general'
    
    def _generate_timeline_data(self, sessions: List[BrowsingSession]) -> Dict[str, Any]:
        """Generate timeline visualization data"""
        daily_stats = defaultdict(lambda: {
            'visits': 0, 
            'sessions': 0, 
            'avg_engagement': 0,
            'engagement_scores': []
        })
        
        for session in sessions:
            date_key = datetime.fromtimestamp(session.start_time / 1000).strftime('%Y-%m-%d')
            daily_stats[date_key]['visits'] += session.total_pages
            daily_stats[date_key]['sessions'] += 1
            daily_stats[date_key]['engagement_scores'].append(session.avg_engagement)
        
        # Calculate average engagement per day
        for date_data in daily_stats.values():
            if date_data['engagement_scores']:
                date_data['avg_engagement'] = np.mean(date_data['engagement_scores'])
            del date_data['engagement_scores']  # Remove temporary data
        
        return {
            'daily_stats': dict(daily_stats),
            'total_days': len(daily_stats)
        }
    
    def _generate_domain_distribution(self, entries: List[HistoryEntry]) -> Dict[str, Any]:
        """Generate domain distribution for pie chart"""
        domain_counts = Counter(entry.domain for entry in entries)
        
        # Get top domains and group others
        top_domains = domain_counts.most_common(8)
        other_count = sum(count for domain, count in domain_counts.items() 
                         if domain not in [d[0] for d in top_domains])
        
        if other_count > 0:
            top_domains.append(('Other', other_count))
        
        return {
            'labels': [domain for domain, count in top_domains],
            'values': [count for domain, count in top_domains],
            'total_unique_domains': len(domain_counts)
        }
    
    def _generate_engagement_heatmap(self, entries: List[HistoryEntry]) -> Dict[str, Any]:
        """Generate engagement heatmap data (hour x day of week)"""
        # Create 7x24 matrix (days x hours)
        heatmap = np.zeros((7, 24))
        counts = np.zeros((7, 24))
        
        for entry in entries:
            if entry.engagement:
                dt = datetime.fromtimestamp(entry.timestamp / 1000)
                day_of_week = dt.weekday()  # 0=Monday
                hour = dt.hour
                
                heatmap[day_of_week][hour] += entry.engagement.contextual_score
                counts[day_of_week][hour] += 1
        
        # Calculate average engagement per time slot
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_heatmap = np.divide(heatmap, counts, 
                                   out=np.zeros_like(heatmap), 
                                   where=counts!=0)
        
        return {
            'data': avg_heatmap.tolist() if hasattr(avg_heatmap, 'tolist') else list(avg_heatmap),
            'labels': {
                'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'hours': [f'{h:02d}:00' for h in range(24)]
            },
            'max_value': float(np.max(avg_heatmap)),
            'total_data_points': int(np.sum(counts))
        }
    
    def _generate_insights(self, sessions: List[BrowsingSession], 
                          entries: List[HistoryEntry]) -> Dict[str, Any]:
        """Generate high-level insights about browsing patterns"""
        
        if not sessions:
            return {'message': 'No sessions to analyze'}
        
        # Session type distribution
        session_types = Counter(session.session_type for session in sessions)
        
        # Calculate productivity metrics
        high_engagement_sessions = [
            s for s in sessions 
            if s.avg_engagement > 0.6 and s.total_time_minutes > 5
        ]
        
        productivity_score = len(high_engagement_sessions) / len(sessions) if sessions else 0
        
        # Time analysis
        total_time_hours = sum(s.total_time_minutes for s in sessions) / 60
        avg_session_length = np.mean([s.total_time_minutes for s in sessions])
        
        # Domain analysis
        all_domains = [entry.domain for entry in entries]
        unique_domains = len(set(all_domains))
        top_domain = Counter(all_domains).most_common(1)[0] if all_domains else ('unknown', 0)
        
        return {
            'session_analysis': {
                'total_sessions': len(sessions),
                'session_types': dict(session_types),
                'avg_session_length_minutes': float(avg_session_length),
                'high_engagement_sessions': len(high_engagement_sessions),
                'productivity_score': float(productivity_score)
            },
            'time_analysis': {
                'total_browsing_hours': float(total_time_hours),
                'avg_daily_browsing': float(total_time_hours / max(1, len(set(
                    datetime.fromtimestamp(s.start_time / 1000).date() 
                    for s in sessions
                ))))
            },
            'domain_analysis': {
                'unique_domains': unique_domains,
                'top_domain': top_domain[0],
                'domain_diversity': float(unique_domains / len(all_domains)) if all_domains else 0
            }
        } 