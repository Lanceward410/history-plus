from flask import Blueprint, request, jsonify, current_app
import time
from typing import Dict, Any

from models.data_models import APIResponse
from services.data_processor import DataProcessor

# Create API blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the dashboard"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'ai_services': {
            'local_llm': True,
            'embedding_service': True,
            'clustering_service': True
        }
    })

@api_bp.route('/process-history', methods=['POST'])
def process_history():
    """
    Main history processing endpoint
    
    Expected payload:
    {
        "entries": [list of raw history entries from Chrome extension]
    }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            response = APIResponse(
                success=False,
                message="Request must be JSON",
                errors=["Content-Type must be application/json"]
            )
            return jsonify(response.to_dict()), 400
        
        data = request.get_json()
        
        if 'entries' not in data:
            response = APIResponse(
                success=False,
                message="Missing 'entries' field in request",
                errors=["Request must contain 'entries' array"]
            )
            return jsonify(response.to_dict()), 400
        
        entries = data['entries']
        
        if not isinstance(entries, list):
            response = APIResponse(
                success=False,
                message="'entries' must be an array",
                errors=["Invalid data type for 'entries'"]
            )
            return jsonify(response.to_dict()), 400
        
        # Check entry count limits
        max_entries = current_app.config.get('MAX_HISTORY_ENTRIES', 10000)
        if len(entries) > max_entries:
            response = APIResponse(
                success=False,
                message=f"Too many entries. Maximum allowed: {max_entries}",
                errors=[f"Received {len(entries)} entries, maximum is {max_entries}"]
            )
            return jsonify(response.to_dict()), 400
        
        print(f"Processing {len(entries)} history entries...")
        
        # Process the data
        processor = DataProcessor(current_app.config)
        result = processor.process_raw_history(entries)
        
        processing_time = time.time() - start_time
        
        # Prepare response data
        response_data = {
            'processing_summary': {
                'total_entries': result.total_entries,
                'processed_entries': result.processed_entries,
                'processing_time_seconds': result.processing_time_seconds,
                'sessions_detected': len(result.sessions)
            },
            'sessions': [
                {
                    'id': session.id,
                    'start_time': session.start_time,
                    'end_time': session.end_time,
                    'type': session.session_type,
                    'category': session.category,
                    'total_pages': session.total_pages,
                    'unique_domains': session.unique_domains,
                    'avg_engagement': session.avg_engagement,
                    'total_time_minutes': session.total_time_minutes,
                    'topic_summary': session.topic_summary
                }
                for session in result.sessions
            ],
            'visualizations': {
                'timeline': result.timeline_data,
                'domain_distribution': result.domain_distribution,
                'engagement_heatmap': result.engagement_heatmap
            },
            'insights': result.insights
        }
        
        response = APIResponse(
            success=True,
            data=response_data,
            message=f"Successfully processed {result.processed_entries} entries into {len(result.sessions)} sessions",
            processing_time=processing_time
        )
        
        if result.errors:
            response.errors = result.errors
        
        return jsonify(response.to_dict())
        
    except Exception as e:
        processing_time = time.time() - start_time
        current_app.logger.error(f"Error processing history: {e}")
        
        response = APIResponse(
            success=False,
            message="Internal server error during processing",
            processing_time=processing_time,
            errors=[str(e)]
        )
        
        return jsonify(response.to_dict()), 500

@api_bp.route('/analyze-sessions', methods=['POST'])
def analyze_sessions():
    """
    Analyze specific sessions for detailed insights
    """
    # Placeholder for future session analysis endpoint
    return jsonify({
        'success': True,
        'message': 'Session analysis endpoint - coming soon',
        'timestamp': time.time()
    })

# Semantic search endpoint removed - use enhanced API instead
# /api/enhanced/semantic-search provides full LLM-powered functionality

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    response = APIResponse(
        success=False,
        message="Endpoint not found",
        errors=["The requested endpoint does not exist"]
    )
    return jsonify(response.to_dict()), 404

@api_bp.errorhandler(500)
def internal_error(error):
    response = APIResponse(
        success=False,
        message="Internal server error",
        errors=["An unexpected error occurred"]
    )
    return jsonify(response.to_dict()), 500 