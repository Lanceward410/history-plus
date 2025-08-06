"""
Enhanced API routes with real-time integration and semantic search
"""

from flask import Blueprint, request, jsonify, current_app
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from models.data_models import APIResponse
from services.database_service import DatabaseService, QueryFilter, QueryOptions, QueryOperator
from services.realtime_service import RealtimeService

logger = logging.getLogger(__name__)

# Create enhanced API blueprint
enhanced_api_bp = Blueprint('enhanced_api', __name__)

# Global service instances (initialized in app factory)
db_service: DatabaseService = None
realtime_service: RealtimeService = None

def init_services(app_db_service: DatabaseService, app_realtime_service: RealtimeService):
    """Initialize service instances"""
    global db_service, realtime_service
    db_service = app_db_service
    realtime_service = app_realtime_service

@enhanced_api_bp.route('/semantic-search', methods=['POST'])
def semantic_search():
    """
    Advanced semantic search endpoint
    
    Expected payload:
    {
        "query": "search text",
        "mode": "fused" | "chrome-history",  // Data mode
        "filters": {
            "categories": ["work", "research"],
            "date_range": ["2024-01-01", "2024-01-31"],
            "domains": ["github.com"],
            "engagement_range": [0.5, 1.0]
        },
        "options": {
            "limit": 50,
            "min_similarity": 0.3
        }
    }
    """
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify(APIResponse(
                success=False,
                message="Request must be JSON",
                errors=["Content-Type must be application/json"]
            ).to_dict()), 400
        
        data = request.get_json()
        query_text = data.get('query', '').strip()
        data_mode = data.get('mode', 'fused')  # Default to fused mode
        
        # Validate data mode
        if data_mode not in ['chrome-history', 'fused']:
            return jsonify(APIResponse(
                success=False,
                message="Invalid data mode",
                errors=["Mode must be 'chrome-history' or 'fused'"]
            ).to_dict()), 400
        
        # Check if semantic search is allowed in this mode
        if data_mode == 'chrome-history':
            return jsonify(APIResponse(
                success=False,
                message="Semantic search not available in Chrome data only mode",
                errors=["Switch to Enhanced data mode to access AI-powered semantic search"],
                data={"mode_limitation": True, "required_mode": "fused"}
            ).to_dict()), 403
        
        if not query_text:
            return jsonify(APIResponse(
                success=False,
                message="Query text is required",
                errors=["'query' field cannot be empty"]
            ).to_dict()), 400
        
        # Parse filters
        filters = data.get('filters', {})
        options = data.get('options', {})
        
        # Validate database service
        if not db_service:
            return jsonify(APIResponse(
                success=False,
                message="Database service not available",
                errors=["Database service not initialized"]
            ).to_dict()), 503
        
        # Perform semantic search
        results = db_service.semantic_search(
            query_text=query_text,
            content_types=filters.get('content_types'),
            category_filter=filters.get('category'),
            limit=options.get('limit', 50),
            min_similarity=options.get('min_similarity', 0.3)
        )
        
        # If additional filters provided, perform advanced filtering
        if any(key in filters for key in ['date_range', 'categories', 'domains', 'engagement_range']):
            # Convert to advanced filter search
            date_range = None
            if 'date_range' in filters and len(filters['date_range']) == 2:
                try:
                    start_date = datetime.fromisoformat(filters['date_range'][0])
                    end_date = datetime.fromisoformat(filters['date_range'][1])
                    date_range = (start_date, end_date)
                except ValueError as e:
                    logger.warning(f"Invalid date range format: {e}")
            
            engagement_range = None
            if 'engagement_range' in filters and len(filters['engagement_range']) == 2:
                engagement_range = tuple(filters['engagement_range'])
            
            advanced_results = db_service.advanced_filter_search(
                text_query=query_text,
                date_range=date_range,
                categories=filters.get('categories'),
                domains=filters.get('domains'),
                engagement_range=engagement_range,
                limit=options.get('limit', 50)
            )
            
            # Merge results, prioritizing advanced filter results
            result_ids = {r['id'] for r in advanced_results}
            for semantic_result in results:
                if semantic_result['id'] not in result_ids:
                    advanced_results.append(semantic_result)
            
            results = advanced_results[:options.get('limit', 50)]
        
        processing_time = time.time() - start_time
        
        # Broadcast search results to real-time subscribers
        if realtime_service:
            try:
                realtime_service.broadcast_semantic_search_result(results, query_text)
            except Exception as e:
                logger.warning(f"Failed to broadcast search results: {e}")
        
        response = APIResponse(
            success=True,
            message=f"Found {len(results)} matching results",
            data={
                'results': results,
                'query': query_text,
                'total_results': len(results),
                'processing_time_ms': processing_time * 1000,
                'search_type': 'semantic' if not any(key in filters for key in ['date_range', 'categories', 'domains', 'engagement_range']) else 'advanced_semantic'
            },
            processing_time=processing_time
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in semantic search: {e}")
        
        response = APIResponse(
            success=False,
            message="Error performing semantic search",
            processing_time=processing_time,
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/query-database', methods=['POST'])
def query_database():
    """
    Advanced database querying endpoint
    
    Expected payload:
    {
        "table": "history_entries",
        "filters": [
            {"field": "category", "operator": "in", "value": ["work", "research"]},
            {"field": "timestamp", "operator": "between", "value": [start_ts, end_ts]}
        ],
        "options": {
            "limit": 100,
            "offset": 0,
            "order_by": "timestamp",
            "order_desc": true
        }
    }
    """
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify(APIResponse(
                success=False,
                message="Request must be JSON"
            ).to_dict()), 400
        
        data = request.get_json()
        table = data.get('table', 'history_entries')
        
        # Validate table name
        allowed_tables = ['history_entries', 'browsing_sessions', 'task_clusters']
        if table not in allowed_tables:
            return jsonify(APIResponse(
                success=False,
                message=f"Table '{table}' not allowed",
                errors=[f"Allowed tables: {', '.join(allowed_tables)}"]
            ).to_dict()), 400
        
        # Validate database service
        if not db_service:
            return jsonify(APIResponse(
                success=False,
                message="Database service not available"
            ).to_dict()), 503
        
        # Parse filters
        filters = []
        for filter_data in data.get('filters', []):
            try:
                # Validate filter data
                if not all(key in filter_data for key in ['field', 'operator', 'value']):
                    logger.warning(f"Invalid filter format: {filter_data}")
                    continue
                    
                query_filter = QueryFilter(
                    field=filter_data['field'],
                    operator=QueryOperator(filter_data['operator']),
                    value=filter_data['value'],
                    case_sensitive=filter_data.get('case_sensitive', True)
                )
                filters.append(query_filter)
            except ValueError as e:
                logger.warning(f"Invalid filter operator: {e}")
                continue
        
        # Parse options
        options_data = data.get('options', {})
        options = QueryOptions(
            limit=options_data.get('limit'),
            offset=options_data.get('offset', 0),
            order_by=options_data.get('order_by'),
            order_desc=options_data.get('order_desc', False)
        )
        
        # Execute query
        if table == 'history_entries':
            results = db_service.query_history_entries(filters, options)
        else:
            # For now, only history_entries is fully implemented
            return jsonify(APIResponse(
                success=False,
                message=f"Table '{table}' querying not implemented yet"
            ).to_dict()), 501
        
        processing_time = time.time() - start_time
        
        response = APIResponse(
            success=True,
            message=f"Retrieved {len(results)} records from {table}",
            data={
                'results': results,
                'table': table,
                'total_results': len(results),
                'processing_time_ms': processing_time * 1000,
                'filters_applied': len(filters)
            },
            processing_time=processing_time
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in database query: {e}")
        
        response = APIResponse(
            success=False,
            message="Error executing database query",
            processing_time=processing_time,
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/analytics/categories', methods=['GET'])
def get_category_analytics():
    """Get comprehensive category analytics with mode awareness"""
    try:
        days = request.args.get('days', 30, type=int)
        data_mode = request.args.get('mode', 'fused')  # Default to fused mode
        
        # Validate data mode
        if data_mode not in ['chrome-history', 'fused']:
            return jsonify(APIResponse(
                success=False,
                message="Invalid data mode",
                errors=["Mode must be 'chrome-history' or 'fused'"]
            ).to_dict()), 400
        
        # Validate parameters
        if days < 1 or days > 365:
            return jsonify(APIResponse(
                success=False,
                message="Days parameter must be between 1 and 365"
            ).to_dict()), 400
        
        if not db_service:
            return jsonify(APIResponse(
                success=False,
                message="Database service not available"
            ).to_dict()), 503
        
        # Get analytics data based on mode
        if data_mode == 'chrome-history':
            # Basic analytics for chrome-history mode
            analytics_data = db_service.get_basic_category_analytics(days)
        else:
            # Full analytics for fused mode
            analytics_data = db_service.get_category_analytics(days)
        
        # Check for errors in analytics data
        if 'error' in analytics_data:
            return jsonify(APIResponse(
                success=False,
                message="Error retrieving category analytics",
                errors=[analytics_data['error']]
            ).to_dict()), 500
        
        # Broadcast analytics update
        if realtime_service:
            try:
                realtime_service.broadcast_analytics_update({
                    'type': 'category_analytics',
                    'data': analytics_data
                })
            except Exception as e:
                logger.warning(f"Failed to broadcast analytics update: {e}")
        
        response = APIResponse(
            success=True,
            message=f"Category analytics for last {days} days",
            data=analytics_data
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error getting category analytics: {e}")
        
        response = APIResponse(
            success=False,
            message="Error retrieving category analytics",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/analytics/productivity', methods=['GET'])
def get_productivity_insights():
    """Get productivity insights with mode awareness"""
    try:
        days = request.args.get('days', 7, type=int)
        data_mode = request.args.get('mode', 'fused')  # Default to fused mode
        
        # Validate data mode
        if data_mode not in ['chrome-history', 'fused']:
            return jsonify(APIResponse(
                success=False,
                message="Invalid data mode",
                errors=["Mode must be 'chrome-history' or 'fused'"]
            ).to_dict()), 400
        
        # Validate parameters
        if days < 1 or days > 90:
            return jsonify(APIResponse(
                success=False,
                message="Days parameter must be between 1 and 90"
            ).to_dict()), 400
        
        if not db_service:
            return jsonify(APIResponse(
                success=False,
                message="Database service not available"
            ).to_dict()), 503
        
        # Get productivity data based on mode
        if data_mode == 'chrome-history':
            # Limited productivity insights for chrome-history mode
            productivity_data = {
                "mode": "chrome-history",
                "note": "Basic productivity overview - switch to Enhanced data mode for detailed engagement tracking",
                "period_days": days,
                "basic_insights": {
                    "message": "Productivity tracking requires engagement data available in Enhanced data mode",
                    "suggestion": "Switch to Enhanced data mode for accurate time tracking and productivity metrics"
                }
            }
        else:
            # Full productivity insights for fused mode
            productivity_data = db_service.get_productivity_insights(days)
        
        # Check for errors in productivity data
        if 'error' in productivity_data:
            return jsonify(APIResponse(
                success=False,
                message="Error retrieving productivity insights",
                errors=[productivity_data['error']]
            ).to_dict()), 500
        
        # Broadcast analytics update
        if realtime_service:
            try:
                realtime_service.broadcast_analytics_update({
                    'type': 'productivity_insights',
                    'data': productivity_data
                })
            except Exception as e:
                logger.warning(f"Failed to broadcast analytics update: {e}")
        
        response = APIResponse(
            success=True,
            message=f"Productivity insights for last {days} days",
            data=productivity_data
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error getting productivity insights: {e}")
        
        response = APIResponse(
            success=False,
            message="Error retrieving productivity insights",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/database/performance', methods=['GET'])
def get_database_performance():
    """Get database performance statistics"""
    try:
        if not db_service:
            return jsonify(APIResponse(
                success=False,
                message="Database service not available"
            ).to_dict()), 503
        
        performance_stats = db_service.get_performance_stats()
        
        # Check for errors in performance stats
        if 'error' in performance_stats:
            return jsonify(APIResponse(
                success=False,
                message="Error retrieving database performance",
                errors=[performance_stats['error']]
            ).to_dict()), 500
        
        response = APIResponse(
            success=True,
            message="Database performance statistics",
            data=performance_stats
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error getting database performance: {e}")
        
        response = APIResponse(
            success=False,
            message="Error retrieving database performance",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/realtime/stats', methods=['GET'])
def get_realtime_stats():
    """Get real-time connection statistics"""
    try:
        if not realtime_service:
            return jsonify(APIResponse(
                success=False,
                message="Real-time service not available"
            ).to_dict()), 503
        
        stats = realtime_service.get_connection_stats()
        
        # Check for errors in connection stats
        if 'error' in stats:
            return jsonify(APIResponse(
                success=False,
                message="Error retrieving real-time statistics",
                errors=[stats['error']]
            ).to_dict()), 500
        
        response = APIResponse(
            success=True,
            message="Real-time connection statistics",
            data=stats
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error getting real-time stats: {e}")
        
        response = APIResponse(
            success=False,
            message="Error retrieving real-time statistics",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/database/bulk-insert', methods=['POST'])
def bulk_insert_entries():
    """Bulk insert history entries"""
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify(APIResponse(
                success=False,
                message="Request must be JSON"
            ).to_dict()), 400
        
        data = request.get_json()
        entries = data.get('entries', [])
        
        if not isinstance(entries, list):
            return jsonify(APIResponse(
                success=False,
                message="'entries' must be a list"
            ).to_dict()), 400
        
        if len(entries) == 0:
            return jsonify(APIResponse(
                success=False,
                message="No entries provided"
            ).to_dict()), 400
        
        # Validate entry count
        max_entries = current_app.config.get('MAX_BULK_INSERT_ENTRIES', 1000)
        if len(entries) > max_entries:
            return jsonify(APIResponse(
                success=False,
                message=f"Too many entries. Maximum allowed: {max_entries}",
                errors=[f"Received {len(entries)} entries"]
            ).to_dict()), 400
        
        if not db_service:
            return jsonify(APIResponse(
                success=False,
                message="Database service not available"
            ).to_dict()), 503
        
        # Perform bulk insert
        inserted_count = db_service.bulk_insert_history_entries(entries)
        
        processing_time = time.time() - start_time
        
        response = APIResponse(
            success=True,
            message=f"Successfully inserted {inserted_count} out of {len(entries)} entries",
            data={
                'inserted_count': inserted_count,
                'total_provided': len(entries),
                'success_rate': inserted_count / len(entries) if entries else 0,
                'processing_time_ms': processing_time * 1000
            },
            processing_time=processing_time
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in bulk insert: {e}")
        
        response = APIResponse(
            success=False,
            message="Error performing bulk insert",
            processing_time=processing_time,
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/realtime/events', methods=['GET'])
def get_event_history():
    """Get real-time event history"""
    try:
        if not realtime_service:
            return jsonify(APIResponse(
                success=False,
                message="Real-time service not available"
            ).to_dict()), 503
        
        event_type = request.args.get('type')
        limit = request.args.get('limit', 100, type=int)
        
        # Validate limit
        if limit < 1 or limit > 500:
            return jsonify(APIResponse(
                success=False,
                message="Limit must be between 1 and 500"
            ).to_dict()), 400
        
        events = realtime_service.get_event_history(event_type, limit)
        
        response = APIResponse(
            success=True,
            message=f"Retrieved {len(events)} events",
            data={
                'events': events,
                'total_events': len(events),
                'filtered_by_type': event_type,
                'limit': limit
            }
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error getting event history: {e}")
        
        response = APIResponse(
            success=False,
            message="Error retrieving event history",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/realtime/rooms/<room_name>', methods=['GET'])
def get_room_info(room_name):
    """Get information about a specific real-time room"""
    try:
        if not realtime_service:
            return jsonify(APIResponse(
                success=False,
                message="Real-time service not available"
            ).to_dict()), 503
        
        room_info = realtime_service.get_room_info(room_name)
        
        # Check for errors in room info
        if 'error' in room_info:
            return jsonify(APIResponse(
                success=False,
                message="Error retrieving room information",
                errors=[room_info['error']]
            ).to_dict()), 500
        
        response = APIResponse(
            success=True,
            message=f"Room information for '{room_name}'",
            data=room_info
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error getting room info for {room_name}: {e}")
        
        response = APIResponse(
            success=False,
            message="Error retrieving room information",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@enhanced_api_bp.route('/analyze-content', methods=['POST'])
def analyze_content():
    """
    LLM-powered content analysis for semantic understanding
    """
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify(APIResponse(
                success=False,
                message="Request must be JSON"
            ).to_dict()), 400
        
        data = request.get_json()
        
        # Extract content for analysis
        url = data.get("url", "")
        title = data.get("title", "")
        domain = data.get("domain", "")
        content_summary = data.get("content_summary", "")
        meta_description = data.get("meta_description", "")
        page_type = data.get("page_type", "webpage")
        keywords = data.get("keywords", [])
        engagement_context = data.get("engagement_context", {})
        
        # Perform content analysis
        analysis_result = perform_content_analysis({
            "url": url,
            "title": title,
            "domain": domain,
            "summary": content_summary,
            "description": meta_description,
            "keywords": keywords,
            "engagement": engagement_context
        })
        
        # Broadcast analysis completion
        if realtime_service:
            realtime_service.broadcast_content_analysis_complete(analysis_result)
        
        processing_time = time.time() - start_time
        
        response = APIResponse(
            success=True,
            message="Content analysis completed",
            data={
                "analysis": analysis_result,
                "processing_time_ms": round(processing_time * 1000, 2)
            }
        )
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        logger.error(f"Error in content analysis: {e}")
        
        response = APIResponse(
            success=False,
            message="Content analysis failed",
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500


def perform_content_analysis(content_data):
    """
    Analyze content for enhanced categorization and semantic understanding
    This is where you would integrate with your preferred LLM
    """
    try:
        title = content_data.get("title", "").lower()
        domain = content_data.get("domain", "").lower()
        summary = content_data.get("summary", "").lower()
        description = content_data.get("description", "").lower()
        keywords = [k.lower() for k in content_data.get("keywords", [])]
        engagement = content_data.get("engagement", {})
        
        # Combine all text for analysis
        all_text = f"{title} {domain} {summary} {description} {' '.join(keywords)}"
        
        # Enhanced category classification with context
        category, intent, confidence = classify_content_with_context(all_text, engagement)
        
        # Generate semantic tags
        tags = extract_semantic_tags(all_text, category)
        
        # Generate insights based on engagement patterns
        insights = generate_content_insights(content_data, engagement)
        
        return {
            "category": category,
            "intent": intent,
            "insights": insights,
            "tags": tags,
            "relevance": confidence,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Content analysis error: {e}")
        return {
            "category": "other",
            "intent": "unknown",
            "insights": "Analysis unavailable",
            "tags": [],
            "relevance": 0.5
        }


def classify_content_with_context(text, engagement):
    """Smart classification based on content and engagement patterns"""
    
    # High engagement indicators
    high_engagement = engagement.get("time_spent", 0) > 60000  # > 1 minute
    deep_scroll = engagement.get("scroll_depth", 0) > 0.7  # > 70% scroll
    active_interaction = engagement.get("interactions", 0) > 10
    
    # Development content
    dev_keywords = ["python", "javascript", "code", "programming", "api", "github", "stackoverflow", 
                   "developer", "coding", "software", "algorithm", "database", "framework"]
    if any(keyword in text for keyword in dev_keywords):
        intent = "learning_coding" if high_engagement else "quick_reference"
        return "development", intent, 0.9
    
    # Work/productivity content
    work_keywords = ["document", "spreadsheet", "meeting", "presentation", "report", "business",
                    "office", "productivity", "team", "project", "deadline", "management"]
    if any(keyword in text for keyword in work_keywords):
        intent = "focused_work" if high_engagement else "quick_task"
        return "work", intent, 0.85
    
    # Learning/education content
    learning_keywords = ["tutorial", "course", "learn", "education", "lesson", "study", "training",
                        "university", "academic", "research", "knowledge", "skill"]
    if any(keyword in text for keyword in learning_keywords):
        intent = "deep_learning" if high_engagement and deep_scroll else "browsing_content"
        return "learning", intent, 0.9
    
    # Entertainment content
    entertainment_keywords = ["video", "movie", "music", "game", "entertainment", "fun", "comedy",
                           "drama", "series", "show", "stream", "watch"]
    if any(keyword in text for keyword in entertainment_keywords):
        intent = "active_consumption" if high_engagement else "casual_browsing"
        return "entertainment", intent, 0.8
    
    # News content
    news_keywords = ["news", "breaking", "article", "journalist", "report", "current", "events",
                    "politics", "world", "update", "headline"]
    if any(keyword in text for keyword in news_keywords):
        intent = "staying_informed" if high_engagement else "headline_scanning"
        return "news", intent, 0.75
    
    # Social media
    social_keywords = ["twitter", "facebook", "linkedin", "reddit", "social", "post", "share",
                      "comment", "follow", "like", "discussion"]
    if any(keyword in text for keyword in social_keywords):
        intent = "active_social" if active_interaction else "passive_scrolling"
        return "social", intent, 0.7
    
    # Shopping
    shopping_keywords = ["buy", "purchase", "cart", "price", "order", "shop", "product", "deal",
                        "sale", "amazon", "store", "checkout"]
    if any(keyword in text for keyword in shopping_keywords):
        intent = "purchasing" if high_engagement else "browsing_products"
        return "shopping", intent, 0.8
    
    # Default classification
    if high_engagement:
        return "research", "deep_dive", 0.6
    else:
        return "other", "casual_browsing", 0.4


def extract_semantic_tags(text, category):
    """Extract relevant semantic tags for search and categorization"""
    
    tags = []
    
    # Category-specific tags
    if category == "development":
        dev_tags = ["programming", "coding", "software", "technical", "tutorial"]
        tags.extend([tag for tag in dev_tags if tag in text])
    
    elif category == "work":
        work_tags = ["professional", "business", "productivity", "documents", "collaboration"]
        tags.extend([tag for tag in work_tags if tag in text])
    
    elif category == "learning":
        learning_tags = ["education", "knowledge", "skill-building", "tutorial", "academic"]
        tags.extend([tag for tag in learning_tags if tag in text])
    
    elif category == "entertainment":
        entertainment_tags = ["media", "video", "music", "leisure", "relaxation"]
        tags.extend([tag for tag in entertainment_tags if tag in text])
    
    # Common semantic tags
    if "how to" in text or "tutorial" in text:
        tags.append("instructional")
    if "review" in text or "opinion" in text:
        tags.append("evaluative")
    if "breaking" in text or "urgent" in text:
        tags.append("urgent")
    
    return list(set(tags))  # Remove duplicates


def generate_content_insights(content_data, engagement):
    """Generate human-readable insights about the content"""
    
    title = content_data.get("title", "")
    domain = content_data.get("domain", "")
    time_spent = engagement.get("time_spent", 0)
    scroll_depth = engagement.get("scroll_depth", 0)
    interactions = engagement.get("interactions", 0)
    
    insights = []
    
    # Engagement insights
    if time_spent > 300000:  # > 5 minutes
        insights.append("Long-form content consumption")
    elif time_spent > 60000:  # > 1 minute
        insights.append("Focused reading")
    else:
        insights.append("Quick scan or reference")
    
    # Interaction insights
    if interactions > 20:
        insights.append("High user interaction")
    elif interactions > 5:
        insights.append("Moderate engagement")
    
    # Content type insights
    if "video" in title.lower() or "watch" in title.lower():
        insights.append("Video content")
    if "tutorial" in title.lower() or "how to" in title.lower():
        insights.append("Educational content")
    
    return "; ".join(insights) if insights else "Standard web content"


# Error handlers for enhanced API
@enhanced_api_bp.errorhandler(404)
def enhanced_not_found(error):
    response = APIResponse(
        success=False,
        message="Enhanced API endpoint not found",
        errors=["The requested endpoint does not exist in the enhanced API"]
    )
    return jsonify(response.to_dict()), 404

@enhanced_api_bp.errorhandler(405)
def enhanced_method_not_allowed(error):
    response = APIResponse(
        success=False,
        message="Method not allowed",
        errors=["The requested method is not allowed for this endpoint"]
    )
    return jsonify(response.to_dict()), 405

@enhanced_api_bp.errorhandler(429)
def enhanced_rate_limit(error):
    response = APIResponse(
        success=False,
        message="Rate limit exceeded",
        errors=["Too many requests. Please try again later."]
    )
    return jsonify(response.to_dict()), 429

@enhanced_api_bp.errorhandler(500)
def enhanced_internal_error(error):
    response = APIResponse(
        success=False,
        message="Internal server error",
        errors=["An unexpected error occurred in the enhanced API"]
    )
    return jsonify(response.to_dict()), 500

