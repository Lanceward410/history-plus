"""
Enhanced Database Service Layer for History Plus
Provides comprehensive database operations with advanced querying,
real-time updates, and semantic search capabilities.
"""

import asyncio
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import numpy as np

logger = logging.getLogger(__name__)

class QueryOperator(Enum):
    """Query operators for filtering"""
    EQUALS = "eq"
    NOT_EQUALS = "ne" 
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

@dataclass
class QueryFilter:
    """Advanced query filter specification"""
    field: str
    operator: QueryOperator
    value: Any
    case_sensitive: bool = True

@dataclass
class QueryOptions:
    """Query execution options"""
    limit: Optional[int] = None
    offset: int = 0
    order_by: Optional[str] = None
    order_desc: bool = False
    group_by: Optional[str] = None
    distinct: bool = False

@dataclass
class AggregationSpec:
    """Aggregation specification"""
    function: str  # 'count', 'sum', 'avg', 'min', 'max', 'group_concat'
    field: Optional[str] = None
    alias: Optional[str] = None

class DatabaseChangeType(Enum):
    """Types of database changes for events"""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_INSERT = "bulk_insert"

@dataclass
class DatabaseEvent:
    """Database change event"""
    change_type: DatabaseChangeType
    table: str
    record_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DatabaseService:
    """
    Advanced database service providing:
    - Advanced querying with filters, aggregations, joins
    - Real-time event notifications
    - Semantic search integration
    - Performance optimization with caching
    - Thread-safe operations
    """
    
    def __init__(self, db_path: str = "history_plus.db"):
        self.db_path = db_path
        
        # Event system
        self.event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = None
        self._event_loop = None
        self._event_thread = None
        
        # Caching
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Performance monitoring
        self.query_stats = defaultdict(list)
        
        # Threading
        self.db_lock = threading.RLock()
        
        # Initialize database
        self._initialize_database()
        
        # Start event processing
        self._start_event_processor()
    
    def _initialize_database(self):
        """Initialize the enhanced database schema"""
        try:
            with self._get_connection() as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # History entries table with full-text search
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS history_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        title TEXT,
                        domain TEXT,
                        timestamp INTEGER,
                        visit_count INTEGER DEFAULT 1,
                        session_id TEXT,
                        category TEXT,
                        subcategory TEXT,
                        confidence REAL,
                        classification_method TEXT,
                        engagement_score REAL,
                        time_on_page INTEGER,
                        scroll_depth REAL,
                        interaction_count INTEGER,
                        page_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Browsing sessions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS browsing_sessions (
                        id TEXT PRIMARY KEY,
                        start_time INTEGER,
                        end_time INTEGER,
                        session_type TEXT,
                        dominant_category TEXT,
                        total_pages INTEGER DEFAULT 0,
                        unique_domains INTEGER DEFAULT 0,
                        avg_engagement REAL DEFAULT 0.0,
                        total_time_minutes REAL DEFAULT 0.0,
                        topic_summary TEXT,
                        productivity_score REAL,
                        multitasking_intensity REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Task clusters table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_clusters (
                        id TEXT PRIMARY KEY,
                        category TEXT NOT NULL,
                        subcategory TEXT NOT NULL,
                        confidence REAL,
                        classification_method TEXT,
                        tab_count INTEGER,
                        engagement_level TEXT,
                        task_progression TEXT,
                        total_time_spent INTEGER,
                        session_id TEXT,
                        cluster_coherence REAL,
                        user_validation BOOLEAN,
                        framework_matched BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES browsing_sessions(id)
                    )
                """)
                
                # Category hierarchy table for learning
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS category_hierarchy (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        subcategory TEXT,
                        user_frequency INTEGER DEFAULT 0,
                        last_seen TIMESTAMP,
                        confidence_sum REAL DEFAULT 0.0,
                        confidence_count INTEGER DEFAULT 0,
                        is_framework_category BOOLEAN DEFAULT FALSE,
                        user_corrections INTEGER DEFAULT 0,
                        successful_matches INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(category, subcategory)
                    )
                """)
                
                # Embeddings table for semantic search
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT UNIQUE,
                        text_content TEXT,
                        embedding_vector BLOB,
                        content_type TEXT DEFAULT 'title_url',
                        source_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Search index table for fast semantic search
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_index (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_id TEXT,
                        keywords TEXT,
                        category_tags TEXT,
                        semantic_tags TEXT,
                        content_type TEXT,
                        boost_factor REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history_entries(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_domain ON history_entries(domain)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_category ON history_entries(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON history_entries(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_time ON browsing_sessions(start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_clusters_category ON task_clusters(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(content_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_content ON search_index(content_id)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager"""
        with self.db_lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                yield conn
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"Database connection error: {e}")
                raise
            finally:
                if conn:
                    conn.close()
    
    # ===== ENHANCED QUERYING METHODS =====
    
    def query_history_entries(
        self, 
        filters: List[QueryFilter] = None,
        options: QueryOptions = None,
        aggregations: List[AggregationSpec] = None
    ) -> Union[List[Dict], Dict]:
        """
        Advanced query for history entries with filtering, aggregation, and pagination
        
        Args:
            filters: List of QueryFilter objects for filtering
            options: QueryOptions for pagination, sorting, grouping
            aggregations: List of AggregationSpec for aggregation queries
            
        Returns:
            List of records or aggregation results
        """
        start_time = time.time()
        
        try:
            # Build cache key
            cache_key = self._build_cache_key("history_entries", filters, options, aggregations)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Build SQL query
            if aggregations:
                query, params = self._build_aggregation_query("history_entries", filters, options, aggregations)
            else:
                query, params = self._build_select_query("history_entries", filters, options)
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                
                if aggregations:
                    result = dict(cursor.fetchone()) if cursor.fetchone() else {}
                else:
                    result = [dict(row) for row in cursor.fetchall()]
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Record performance
            query_time = time.time() - start_time
            self.query_stats["history_entries"].append(query_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying history entries: {e}")
            raise
    
    def semantic_search(
        self, 
        query_text: str,
        content_types: List[str] = None,
        category_filter: str = None,
        limit: int = 50,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with intelligent text matching and context awareness
        
        Args:
            query_text: Natural language search query
            content_types: Filter by content types ['title_url', 'full_content']
            category_filter: Filter by category
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        start_time = time.time()
        
        try:
            # Enhanced semantic search with intelligent matching
            with self._get_connection() as conn:
                # Build intelligent search query
                query = """
                    SELECT h.*, 'enhanced' as similarity_source
                    FROM history_entries h
                    WHERE (
                        h.title LIKE ? OR 
                        h.url LIKE ? OR 
                        h.domain LIKE ? OR
                        h.category LIKE ? OR
                        h.subcategory LIKE ?
                    )
                """
                
                # Create search patterns for better matching
                search_patterns = self._generate_search_patterns(query_text)
                params = []
                
                for pattern in search_patterns:
                    params.extend([f"%{pattern}%"] * 5)  # 5 fields to match
                
                if category_filter:
                    query += " AND h.category = ?"
                    params.append(category_filter)
                
                query += " ORDER BY h.timestamp DESC LIMIT ?"
                params.append(limit * 2)  # Get more results for better ranking
                
                cursor = conn.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    # Enhanced similarity calculation
                    similarity = self._calculate_enhanced_similarity(query_text, row)
                    
                    if similarity >= min_similarity:
                        results.append({
                            'id': row['id'],
                            'title': row['title'],
                            'url': row['url'],
                            'domain': row['domain'],
                            'category': row['category'],
                            'subcategory': row.get('subcategory'),
                            'timestamp': row['timestamp'],
                            'similarity': min(similarity, 1.0),
                            'content_type': 'title_url',
                            'engagement_score': row.get('engagement_score', 0),
                            'time_on_page': row.get('time_on_page', 0)
                        })
            
            # Enhanced ranking with multiple factors
            results = self._enhanced_ranking(results, query_text)
            results = results[:limit]
            
            # Record performance
            search_time = time.time() - start_time
            self.query_stats["semantic_search"].append(search_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _generate_search_patterns(self, query_text: str) -> List[str]:
        """Generate intelligent search patterns from query text"""
        patterns = [query_text.lower()]
        
        # Split query into words for broader matching
        words = query_text.lower().split()
        if len(words) > 1:
            patterns.extend(words)
        
        # Add common synonyms and related terms
        synonyms = self._get_query_synonyms(query_text.lower())
        patterns.extend(synonyms)
        
        return list(set(patterns))  # Remove duplicates
    
    def _get_query_synonyms(self, query: str) -> List[str]:
        """Get synonyms and related terms for better search matching"""
        synonyms = []
        
        # Development-related synonyms
        if any(term in query for term in ['python', 'programming', 'code']):
            synonyms.extend(['development', 'coding', 'software', 'programming'])
        if any(term in query for term in ['javascript', 'js', 'web']):
            synonyms.extend(['javascript', 'web', 'frontend', 'browser'])
        if any(term in query for term in ['chrome', 'extension', 'browser']):
            synonyms.extend(['extension', 'browser', 'chrome', 'addon'])
        
        # Learning-related synonyms
        if any(term in query for term in ['tutorial', 'learn', 'course']):
            synonyms.extend(['learning', 'education', 'tutorial', 'course'])
        if any(term in query for term in ['machine learning', 'ai', 'ml']):
            synonyms.extend(['ai', 'machine learning', 'artificial intelligence', 'ml'])
        
        # Research-related synonyms
        if any(term in query for term in ['research', 'paper', 'study']):
            synonyms.extend(['research', 'academic', 'paper', 'study'])
        
        return synonyms
    
    def _calculate_enhanced_similarity(self, query_text: str, row: Dict[str, Any]) -> float:
        """Calculate enhanced similarity score with multiple factors"""
        query_lower = query_text.lower()
        title = (row['title'] or '').lower()
        url = (row['url'] or '').lower()
        domain = (row['domain'] or '').lower()
        category = (row['category'] or '').lower()
        subcategory = (row.get('subcategory') or '').lower()
        
        similarity = 0.0
        
        # Title matching (highest weight)
        if query_lower in title:
            similarity += 0.5
        elif any(word in title for word in query_lower.split()):
            similarity += 0.3
        
        # URL matching
        if query_lower in url:
            similarity += 0.2
        elif any(word in url for word in query_lower.split()):
            similarity += 0.1
        
        # Domain matching
        if query_lower in domain:
            similarity += 0.15
        elif any(word in domain for word in query_lower.split()):
            similarity += 0.05
        
        # Category matching
        if query_lower in category or category in query_lower:
            similarity += 0.25
        elif subcategory and (query_lower in subcategory or subcategory in query_lower):
            similarity += 0.15
        
        # Engagement boost (high engagement content gets slight boost)
        engagement_score = row.get('engagement_score', 0)
        if engagement_score > 0.7:
            similarity += 0.05
        
        # Recency boost (recent content gets slight boost)
        timestamp = row.get('timestamp', 0)
        if timestamp > 0:
            age_days = (time.time() * 1000 - timestamp) / (1000 * 60 * 60 * 24)
            if age_days < 7:  # Within last week
                similarity += 0.03
        
        return min(similarity, 1.0)
    
    def _enhanced_ranking(self, results: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """Enhanced ranking algorithm considering multiple factors"""
        
        for result in results:
            # Calculate composite score
            base_score = result['similarity']
            
            # Engagement factor (0-0.2 boost)
            engagement_boost = min(result.get('engagement_score', 0) * 0.2, 0.2)
            
            # Recency factor (0-0.1 boost)
            timestamp = result.get('timestamp', 0)
            if timestamp > 0:
                age_days = (time.time() * 1000 - timestamp) / (1000 * 60 * 60 * 24)
                recency_boost = max(0, (30 - age_days) / 30) * 0.1  # Decay over 30 days
            else:
                recency_boost = 0
            
            # Content quality factor (0-0.1 boost)
            time_on_page = result.get('time_on_page', 0)
            quality_boost = min(time_on_page / 600000, 1.0) * 0.1  # Based on time spent
            
            # Calculate final score
            result['final_score'] = base_score + engagement_boost + recency_boost + quality_boost
            result['ranking_factors'] = {
                'base_similarity': base_score,
                'engagement_boost': engagement_boost,
                'recency_boost': recency_boost,
                'quality_boost': quality_boost
            }
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def advanced_filter_search(
        self,
        text_query: str = None,
        date_range: Tuple[datetime, datetime] = None,
        categories: List[str] = None,
        domains: List[str] = None,
        engagement_range: Tuple[float, float] = None,
        session_types: List[str] = None,
        time_spent_range: Tuple[int, int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Advanced multi-criteria search combining multiple filters
        """
        try:
            filters = []
            
            # Date range filter
            if date_range:
                start_ts = int(date_range[0].timestamp() * 1000)
                end_ts = int(date_range[1].timestamp() * 1000)
                filters.append(QueryFilter("timestamp", QueryOperator.BETWEEN, (start_ts, end_ts)))
            
            # Category filter
            if categories:
                filters.append(QueryFilter("category", QueryOperator.IN, categories))
            
            # Domain filter
            if domains:
                filters.append(QueryFilter("domain", QueryOperator.IN, domains))
            
            # Engagement filter
            if engagement_range:
                filters.append(QueryFilter("engagement_score", QueryOperator.BETWEEN, engagement_range))
            
            # Time spent filter
            if time_spent_range:
                filters.append(QueryFilter("time_on_page", QueryOperator.BETWEEN, time_spent_range))
            
            options = QueryOptions(limit=limit, order_by="timestamp", order_desc=True)
            
            # Start with structured query
            results = self.query_history_entries(filters, options)
            
            # If text query provided, combine with semantic search
            if text_query:
                semantic_results = self.semantic_search(text_query, limit=limit)
                
                # Merge and rank results
                semantic_ids = {str(r['id']) for r in semantic_results}
                semantic_scores = {str(r['id']): r['similarity'] for r in semantic_results}
                
                # Add relevance scores
                for result in results:
                    result_id = str(result['id'])
                    if result_id in semantic_ids:
                        result['relevance_score'] = semantic_scores[result_id]
                        result['match_type'] = 'semantic_and_filter'
                    else:
                        result['relevance_score'] = 0.5  # Filter match only
                        result['match_type'] = 'filter_only'
                
                # Add semantic-only results that passed filters
                for sem_result in semantic_results:
                    if not any(r['id'] == sem_result['id'] for r in results):
                        sem_result['match_type'] = 'semantic_only'
                        sem_result['relevance_score'] = sem_result['similarity']
                        results.append(sem_result)
                
                # Sort by relevance score
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                results = results[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced filter search: {e}")
            return []
    
    # ===== REAL-TIME UPDATE METHODS =====
    
    def add_event_listener(self, event_type: str, callback: Callable[[DatabaseEvent], None]):
        """Add event listener for database changes"""
        self.event_listeners[event_type].append(callback)
    
    def remove_event_listener(self, event_type: str, callback: Callable[[DatabaseEvent], None]):
        """Remove event listener"""
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
    
    def _emit_event(self, event: DatabaseEvent):
        """Emit database change event to listeners (synchronous version)"""
        try:
            # Notify all listeners for this event type
            event_type = event.change_type.value
            for callback in self.event_listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
            
            # Notify global listeners
            for callback in self.event_listeners['*']:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in global event listener: {e}")
                    
        except Exception as e:
            logger.error(f"Error emitting database event: {e}")
    
    def _start_event_processor(self):
        """Start background event processing"""
        try:
            def create_event_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._event_loop = loop
                self.event_queue = asyncio.Queue()
                loop.run_until_complete(self._event_processor())
            
            self._event_thread = threading.Thread(target=create_event_loop, daemon=True)
            self._event_thread.start()
            logger.info("Event processor started")
            
        except Exception as e:
            logger.error(f"Error starting event processor: {e}")
    
    async def _event_processor(self):
        """Process database events asynchronously"""
        while True:
            try:
                if self.event_queue:
                    event = await self.event_queue.get()
                    self._emit_event(event)
                else:
                    await asyncio.sleep(0.1)  # Prevent busy waiting
                    
            except Exception as e:
                logger.error(f"Error processing database event: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    # ===== BULK OPERATIONS =====
    
    def bulk_insert_history_entries(self, entries: List[Dict[str, Any]]) -> int:
        """
        Bulk insert history entries with automatic embedding generation
        
        Args:
            entries: List of history entry dictionaries
            
        Returns:
            Number of successfully inserted entries
        """
        if not entries:
            return 0
        
        start_time = time.time()
        inserted_count = 0
        
        try:
            with self._get_connection() as conn:
                for entry in entries:
                    try:
                        # Insert history entry
                        cursor = conn.execute("""
                            INSERT INTO history_entries (
                                url, title, domain, timestamp, visit_count, session_id,
                                category, subcategory, confidence, classification_method,
                                engagement_score, time_on_page, scroll_depth, interaction_count, page_type
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            entry.get('url', ''), 
                            entry.get('title', ''), 
                            entry.get('domain', ''), 
                            entry.get('timestamp', 0),
                            entry.get('visit_count', 1), 
                            entry.get('session_id'), 
                            entry.get('category'),
                            entry.get('subcategory'), 
                            entry.get('confidence'), 
                            entry.get('classification_method'),
                            entry.get('engagement_score'), 
                            entry.get('time_on_page'), 
                            entry.get('scroll_depth'),
                            entry.get('interaction_count'), 
                            entry.get('page_type')
                        ))
                        
                        entry_id = cursor.lastrowid
                        
                        # Generate and store embedding (placeholder for now)
                        self._generate_and_store_embedding(entry_id, f"{entry.get('title', '')} {entry.get('url', '')}", conn)
                        
                        # Update search index
                        self._update_search_index(entry_id, entry, conn)
                        
                        inserted_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error inserting entry {entry.get('url', 'unknown')}: {e}")
                        continue
                
                conn.commit()
            
            # Emit bulk insert event
            self._emit_event(DatabaseEvent(
                change_type=DatabaseChangeType.BULK_INSERT,
                table="history_entries",
                data={"count": inserted_count, "processing_time": time.time() - start_time}
            ))
            
            logger.info(f"Bulk inserted {inserted_count} history entries")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            return 0
    
    # ===== ANALYTICS AND INSIGHTS =====
    
    def get_category_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive category analytics"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            start_timestamp = int(start_date.timestamp() * 1000)
            
            with self._get_connection() as conn:
                # Category distribution
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count, AVG(engagement_score) as avg_engagement
                    FROM history_entries 
                    WHERE timestamp >= ? AND category IS NOT NULL
                    GROUP BY category
                    ORDER BY count DESC
                """, (start_timestamp,))
                category_stats = [dict(row) for row in cursor.fetchall()]
                
                # Time-based category trends
                cursor = conn.execute("""
                    SELECT 
                        DATE(timestamp/1000, 'unixepoch') as date,
                        category,
                        COUNT(*) as visits,
                        AVG(engagement_score) as avg_engagement
                    FROM history_entries 
                    WHERE timestamp >= ? AND category IS NOT NULL
                    GROUP BY DATE(timestamp/1000, 'unixepoch'), category
                    ORDER BY date DESC, visits DESC
                """, (start_timestamp,))
                daily_trends = [dict(row) for row in cursor.fetchall()]
                
                # Category transitions (what follows what)
                cursor = conn.execute("""
                    WITH ordered_entries AS (
                        SELECT category, timestamp, 
                               LAG(category) OVER (ORDER BY timestamp) as prev_category
                        FROM history_entries 
                        WHERE timestamp >= ? AND category IS NOT NULL
                    )
                    SELECT prev_category, category, COUNT(*) as transition_count
                    FROM ordered_entries
                    WHERE prev_category IS NOT NULL AND prev_category != category
                    GROUP BY prev_category, category
                    ORDER BY transition_count DESC
                    LIMIT 20
                """, (start_timestamp,))
                transitions = [dict(row) for row in cursor.fetchall()]
            
            return {
                "category_distribution": category_stats,
                "daily_trends": daily_trends,
                "category_transitions": transitions,
                "analysis_period_days": days,
                "total_categories": len(category_stats)
            }
            
        except Exception as e:
            logger.error(f"Error getting category analytics: {e}")
            return {"error": str(e)}
    
    def get_basic_category_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get basic category analytics for chrome-history mode (limited insights)"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            start_timestamp = int(start_date.timestamp() * 1000)
            
            with self._get_connection() as conn:
                # Simple category distribution
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count
                    FROM history_entries 
                    WHERE timestamp >= ? AND category IS NOT NULL
                    GROUP BY category
                    ORDER BY count DESC
                """, (start_timestamp,))
                category_stats = [dict(row) for row in cursor.fetchall()]
                
                # Basic daily activity (no engagement scores)
                cursor = conn.execute("""
                    SELECT 
                        DATE(timestamp/1000, 'unixepoch') as date,
                        COUNT(*) as total_visits
                    FROM history_entries 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp/1000, 'unixepoch')
                    ORDER BY date DESC
                """, (start_timestamp,))
                daily_activity = [dict(row) for row in cursor.fetchall()]
            
            return {
                "category_distribution": category_stats,
                "daily_activity": daily_activity,
                "analysis_period_days": days,
                "total_categories": len(category_stats),
                "mode": "chrome-history",
                "note": "Basic analytics - switch to Enhanced data mode for advanced insights"
            }
            
        except Exception as e:
            logger.error(f"Error getting basic category analytics: {e}")
            return {"error": str(e)}
    
    def get_productivity_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get productivity insights based on engagement and categories"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            start_timestamp = int(start_date.timestamp() * 1000)
            
            # Define productive vs non-productive categories
            productive_categories = ['work', 'learning', 'development', 'research', 'education']
            
            with self._get_connection() as conn:
                # Overall productivity metrics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        AVG(engagement_score) as avg_engagement,
                        SUM(time_on_page) as total_time,
                        COUNT(DISTINCT session_id) as total_sessions
                    FROM history_entries 
                    WHERE timestamp >= ?
                """, (start_timestamp,))
                overall_stats = dict(cursor.fetchone())
                
                # Productive vs non-productive time
                productive_placeholders = ','.join(['?' for _ in productive_categories])
                cursor = conn.execute(f"""
                    SELECT 
                        CASE WHEN category IN ({productive_placeholders}) THEN 'productive' ELSE 'leisure' END as type,
                        COUNT(*) as count,
                        AVG(engagement_score) as avg_engagement,
                        SUM(time_on_page) as total_time
                    FROM history_entries 
                    WHERE timestamp >= ? AND category IS NOT NULL
                    GROUP BY CASE WHEN category IN ({productive_placeholders}) THEN 'productive' ELSE 'leisure' END
                """, productive_categories + [start_timestamp] + productive_categories)
                
                productivity_breakdown = {row['type']: dict(row) for row in cursor.fetchall()}
                
                # Hourly productivity patterns
                cursor = conn.execute(f"""
                    SELECT 
                        strftime('%H', datetime(timestamp/1000, 'unixepoch')) as hour,
                        AVG(engagement_score) as avg_engagement,
                        COUNT(*) as activity_count,
                        SUM(CASE WHEN category IN ({productive_placeholders}) THEN 1 ELSE 0 END) as productive_count
                    FROM history_entries 
                    WHERE timestamp >= ?
                    GROUP BY strftime('%H', datetime(timestamp/1000, 'unixepoch'))
                    ORDER BY hour
                """, productive_categories + [start_timestamp])
                hourly_patterns = [dict(row) for row in cursor.fetchall()]
            
            # Calculate productivity score
            total_productive = productivity_breakdown.get('productive', {}).get('count', 0)
            total_entries = overall_stats.get('total_entries', 0)
            productivity_score = (total_productive / total_entries * 100) if total_entries > 0 else 0
            
            return {
                "overall_stats": overall_stats,
                "productivity_breakdown": productivity_breakdown,
                "hourly_patterns": hourly_patterns,
                "productivity_score": productivity_score,
                "analysis_period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting productivity insights: {e}")
            return {"error": str(e)}
    
    # ===== HELPER METHODS =====
    
    def _build_select_query(
        self, 
        table: str, 
        filters: List[QueryFilter] = None, 
        options: QueryOptions = None
    ) -> Tuple[str, List]:
        """Build SELECT query with filters and options"""
        try:
            query_parts = [f"SELECT * FROM {table}"]
            params = []
            
            # WHERE clause
            if filters:
                where_conditions = []
                for filter_spec in filters:
                    condition, filter_params = self._build_filter_condition(filter_spec)
                    where_conditions.append(condition)
                    params.extend(filter_params)
                
                if where_conditions:
                    query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
            
            # ORDER BY clause
            if options and options.order_by:
                direction = "DESC" if options.order_desc else "ASC"
                query_parts.append(f"ORDER BY {options.order_by} {direction}")
            
            # LIMIT and OFFSET
            if options and options.limit:
                query_parts.append(f"LIMIT {options.limit}")
                if options.offset:
                    query_parts.append(f"OFFSET {options.offset}")
            
            return " ".join(query_parts), params
            
        except Exception as e:
            logger.error(f"Error building select query: {e}")
            raise
    
    def _build_aggregation_query(
        self, 
        table: str, 
        filters: List[QueryFilter] = None, 
        options: QueryOptions = None,
        aggregations: List[AggregationSpec] = None
    ) -> Tuple[str, List]:
        """Build aggregation query"""
        try:
            # Build SELECT clause with aggregations
            agg_parts = []
            for agg in aggregations or []:
                if agg.function.upper() in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']:
                    field_part = agg.field if agg.field else '*'
                    alias_part = f" AS {agg.alias}" if agg.alias else ""
                    agg_parts.append(f"{agg.function.upper()}({field_part}){alias_part}")
            
            if not agg_parts:
                agg_parts = ["COUNT(*)"]
            
            query_parts = [f"SELECT {', '.join(agg_parts)} FROM {table}"]
            params = []
            
            # WHERE clause
            if filters:
                where_conditions = []
                for filter_spec in filters:
                    condition, filter_params = self._build_filter_condition(filter_spec)
                    where_conditions.append(condition)
                    params.extend(filter_params)
                
                if where_conditions:
                    query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
            
            # GROUP BY clause
            if options and options.group_by:
                query_parts.append(f"GROUP BY {options.group_by}")
            
            return " ".join(query_parts), params
            
        except Exception as e:
            logger.error(f"Error building aggregation query: {e}")
            raise
    
    def _build_filter_condition(self, filter_spec: QueryFilter) -> Tuple[str, List]:
        """Build SQL condition for a filter"""
        try:
            field = filter_spec.field
            op = filter_spec.operator
            value = filter_spec.value
            
            if op == QueryOperator.EQUALS:
                return f"{field} = ?", [value]
            elif op == QueryOperator.NOT_EQUALS:
                return f"{field} != ?", [value]
            elif op == QueryOperator.GREATER_THAN:
                return f"{field} > ?", [value]
            elif op == QueryOperator.LESS_THAN:
                return f"{field} < ?", [value]
            elif op == QueryOperator.GREATER_EQUAL:
                return f"{field} >= ?", [value]
            elif op == QueryOperator.LESS_EQUAL:
                return f"{field} <= ?", [value]
            elif op == QueryOperator.CONTAINS:
                pattern = f"%{value}%" if filter_spec.case_sensitive else f"%{value.lower()}%"
                field_expr = field if filter_spec.case_sensitive else f"LOWER({field})"
                return f"{field_expr} LIKE ?", [pattern]
            elif op == QueryOperator.STARTS_WITH:
                pattern = f"{value}%" if filter_spec.case_sensitive else f"{value.lower()}%"
                field_expr = field if filter_spec.case_sensitive else f"LOWER({field})"
                return f"{field_expr} LIKE ?", [pattern]
            elif op == QueryOperator.ENDS_WITH:
                pattern = f"%{value}" if filter_spec.case_sensitive else f"%{value.lower()}"
                field_expr = field if filter_spec.case_sensitive else f"LOWER({field})"
                return f"{field_expr} LIKE ?", [pattern]
            elif op == QueryOperator.IN:
                if not isinstance(value, (list, tuple)):
                    raise ValueError("IN operator requires list or tuple value")
                placeholders = ','.join(['?' for _ in value])
                return f"{field} IN ({placeholders})", list(value)
            elif op == QueryOperator.NOT_IN:
                if not isinstance(value, (list, tuple)):
                    raise ValueError("NOT_IN operator requires list or tuple value")
                placeholders = ','.join(['?' for _ in value])
                return f"{field} NOT IN ({placeholders})", list(value)
            elif op == QueryOperator.BETWEEN:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError("BETWEEN operator requires tuple/list with 2 values")
                return f"{field} BETWEEN ? AND ?", list(value)
            elif op == QueryOperator.IS_NULL:
                return f"{field} IS NULL", []
            elif op == QueryOperator.IS_NOT_NULL:
                return f"{field} IS NOT NULL", []
            else:
                raise ValueError(f"Unsupported operator: {op}")
                
        except Exception as e:
            logger.error(f"Error building filter condition: {e}")
            raise
    
    def _generate_and_store_embedding(self, entry_id: int, text_content: str, conn):
        """Generate and store embedding for search (placeholder implementation)"""
        try:
            # For now, just store a placeholder
            # This will be enhanced when the embedding service is fully integrated
            content_hash = hashlib.md5(text_content.encode()).hexdigest()
            
            conn.execute("""
                INSERT OR REPLACE INTO embeddings 
                (content_hash, text_content, embedding_vector, source_id, content_type)
                VALUES (?, ?, ?, ?, ?)
            """, (content_hash, text_content, b'placeholder', str(entry_id), 'title_url'))
            
        except Exception as e:
            logger.error(f"Error generating embedding for entry {entry_id}: {e}")
    
    def _update_search_index(self, entry_id: int, entry: Dict[str, Any], conn):
        """Update search index for fast keyword search"""
        try:
            # Extract keywords from title and URL
            text_content = f"{entry.get('title', '')} {entry.get('url', '')}"
            keywords = self._extract_keywords(text_content)
            category_tags = f"{entry.get('category', '')} {entry.get('subcategory', '')}"
            
            conn.execute("""
                INSERT OR REPLACE INTO search_index 
                (content_id, keywords, category_tags, content_type)
                VALUES (?, ?, ?, ?)
            """, (str(entry_id), ' '.join(keywords), category_tags, 'history_entry'))
            
        except Exception as e:
            logger.error(f"Error updating search index for entry {entry_id}: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for indexing"""
        try:
            import re
            words = re.findall(r'\w+', text.lower())
            # Filter out common stop words and short words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'www', 'com', 'org', 'net'
            }
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            return list(set(keywords))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _build_cache_key(
        self, 
        table: str, 
        filters: List[QueryFilter] = None, 
        options: QueryOptions = None, 
        aggregations: List[AggregationSpec] = None
    ) -> str:
        """Build cache key for query results"""
        try:
            key_parts = [table]
            
            if filters:
                filter_strs = [f"{f.field}_{f.operator.value}_{f.value}" for f in filters]
                key_parts.append("filters_" + "_".join(filter_strs))
            
            if options:
                option_str = f"limit_{options.limit}_offset_{options.offset}_order_{options.order_by}_{options.order_desc}"
                key_parts.append(option_str)
            
            if aggregations:
                agg_strs = [f"{a.function}_{a.field}_{a.alias}" for a in aggregations]
                key_parts.append("agg_" + "_".join(agg_strs))
            
            return "_".join(str(part) for part in key_parts)
            
        except Exception as e:
            logger.error(f"Error building cache key: {e}")
            return f"error_{time.time()}"
    
    def _get_cached_result(self, cache_key: str):
        """Get cached query result if still valid"""
        try:
            if cache_key in self.query_cache:
                result, timestamp = self.query_cache[cache_key]
                if datetime.now() - timestamp < self.cache_ttl:
                    return result
                else:
                    del self.query_cache[cache_key]
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result):
        """Cache query result"""
        try:
            self.query_cache[cache_key] = (result, datetime.now())
            
            # Clean old cache entries if cache gets too large
            if len(self.query_cache) > 1000:
                # Remove oldest 25% of entries
                sorted_cache = sorted(self.query_cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_cache[:250]:
                    del self.query_cache[key]
                    
        except Exception as e:
            logger.error(f"Error caching result: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        try:
            stats = {}
            for query_type, times in self.query_stats.items():
                if times:
                    stats[query_type] = {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "total_time": sum(times)
                    }
            
            stats["cache_info"] = {
                "cache_size": len(self.query_cache),
                "cache_hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_attempts', 1), 1)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e)} 