"""
Current Tasks API Routes with Enhanced AI Integration

Provides real-time task detection and analysis using the complete AI pipeline
including semantic embeddings, clustering, and intelligent classification.

Key Features:
- Real-time tab clustering and classification
- Semantic analysis of current browsing behavior
- Category validation and user feedback processing
- Performance monitoring and optimization
- Graceful fallback when AI services are unavailable

This module integrates with the full Local AI Services architecture
for maximum intelligence and reliability.
"""

from flask import Blueprint, request, jsonify
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Import enhanced AI services
from ai.local.category_classifier import get_category_classifier
from ai.local.embedding_service import get_embedding_service
from ai.local.clustering_service import get_clustering_service
from ai.local.local_llm_service import get_llm_service
from ai.interfaces.ai_processor import ProcessingConfig
from ai.category_framework import CategoryFramework
from models.data_models import (
    CurrentTasksResponse, TaskCluster, TabInfo, CategoryValidationRequest,
    ClassificationMethod, EngagementLevel, TaskProgression
)

# Create blueprint
current_tasks_bp = Blueprint('current_tasks', __name__)

# Global services (initialized on first use)
category_classifier = None
framework = CategoryFramework()

def get_classifier():
    """Get global category classifier instance with proper initialization"""
    global category_classifier
    if category_classifier is None:
        # Create optimized config for real-time processing
        config = ProcessingConfig(
            # TUNABLE: Real-time processing optimizations
            embedding_batch_size=16,           # Smaller batches for speed
            max_processing_time=3.0,          # 3 second timeout for real-time
            max_embedding_time=1.5,           # Fast embedding generation
            confidence_threshold=0.6,          # Moderate confidence threshold
            enable_performance_monitoring=True, # Track performance
            enable_debug_logging=False        # Disable debug for speed
        )
        category_classifier = get_category_classifier(config)
    return category_classifier

@current_tasks_bp.route('/api/analyze-current-tasks', methods=['POST'])
def analyze_current_tasks():
    """
    Analyze current browser tabs and return intelligent task clusters
    
    Request format:
    {
        "tabs": [
            {
                "id": "tab_id",
                "title": "Tab Title",
                "url": "https://example.com/page",
                "domain": "example.com",
                "is_active": true,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        ],
        "options": {
            "max_clusters": 10,
            "min_cluster_size": 2,
            "include_singles": true
        }
    }
    
    Returns:
    CurrentTasksResponse with intelligent task clusters
    """
    try:
        start_time = time.time()
        
        # Parse request data
        data = request.get_json()
        if not data or 'tabs' not in data:
            return jsonify({
                "success": False,
                "error": "Missing tabs data in request"
            }), 400
        
        tabs_data = data['tabs']
        options = data.get('options', {})
        
        # Validate input
        if not tabs_data or not isinstance(tabs_data, list):
            return jsonify({
                "success": False,
                "error": "Invalid tabs data format"
            }), 400
        
        # Process tabs with enhanced AI pipeline
        processing_result = _process_tabs_with_ai(tabs_data, options)
        
        # Calculate processing metrics
        total_time = (time.time() - start_time) * 1000
        
        # Create response with correct CurrentTasksResponse format
        response = CurrentTasksResponse(
            current_tasks=processing_result['clusters'],
            task_count=len(processing_result['clusters']),
            dominant_task=processing_result.get('dominant_task'),
            multitasking_score=processing_result['multitasking_score'],
            summary=processing_result['summary'],
            total_tabs_analyzed=len(tabs_data),
            categorization_confidence=processing_result['confidence'],
            new_categories_discovered=processing_result.get('new_categories', 0),
            framework_categories_used=processing_result.get('framework_categories', 0),
            analysis_timestamp=datetime.now(),
            processing_time_ms=int(total_time)
        )
        
        return jsonify(response.dict())
        
    except Exception as e:
        print(f"Error in analyze_current_tasks: {e}")
        
        # Return graceful fallback
        fallback_response = _create_fallback_response(
            data.get('tabs', []) if 'data' in locals() else []
        )
        return jsonify(fallback_response.dict())

def _process_tabs_with_ai(tabs_data: List[Dict], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process tabs using the full AI pipeline for intelligent clustering and classification
    
    Args:
        tabs_data: List of tab dictionaries
        options: Processing options
        
    Returns:
        Dictionary with clusters, scores, and metadata
    """
    try:
        classifier = get_classifier()
        
        # Convert raw tab data to TabInfo objects
        tab_objects = []
        for tab_data in tabs_data:
            tab_info = TabInfo(
                id=tab_data.get('id', ''),
                title=tab_data.get('title', ''),
                url=tab_data.get('url', ''),
                domain=tab_data.get('domain', ''),
                is_active=tab_data.get('is_active', False),
                timestamp=tab_data.get('timestamp', datetime.now().isoformat())
            )
            tab_objects.append(tab_info)
        
        # STAGE 1: Intelligent clustering
        clusters = _perform_intelligent_clustering(tab_objects, options)
        
        # STAGE 2: Classify each cluster
        classified_clusters = []
        total_confidence = 0.0
        
        for cluster_tabs in clusters:
            if len(cluster_tabs) == 0:
                continue
                
            # Convert back to dict format for classifier
            cluster_dicts = [tab.dict() for tab in cluster_tabs]
            
            # Classify the cluster
            classification_result = classifier.classify_tab_cluster(cluster_dicts)
            
            if classification_result.success:
                # Create TaskCluster with rich metadata
                task_cluster = TaskCluster(
                    category=classification_result.category,
                    subcategory=classification_result.subcategory,
                    confidence=classification_result.confidence,
                    classification_method=ClassificationMethod.AI_GENERATED,
                    tab_count=len(cluster_tabs),
                    tabs=cluster_tabs,
                    engagement_level=_calculate_engagement_level(cluster_tabs),
                    task_progression=_analyze_task_progression(cluster_tabs),
                    estimated_duration=f"Active for {_estimate_time_spent(cluster_tabs) // 60000} minutes" if _estimate_time_spent(cluster_tabs) > 0 else None,
                    total_time_spent=_estimate_time_spent(cluster_tabs),
                    cluster_coherence=classification_result.cluster_coherence if hasattr(classification_result, 'cluster_coherence') else None,
                    classification_metadata={
                        "processing_time_ms": classification_result.processing_time_ms if hasattr(classification_result, 'processing_time_ms') else 0,
                        "semantic_similarity": classification_result.semantic_similarity if hasattr(classification_result, 'semantic_similarity') else 0.0,
                        "classification_method": classification_result.method if hasattr(classification_result, 'method') else 'ai_enhanced',
                        "embeddings_shape": str(classification_result.embeddings_shape) if hasattr(classification_result, 'embeddings_shape') and classification_result.embeddings_shape else None
                    }
                )
                
                classified_clusters.append(task_cluster)
                total_confidence += classification_result.confidence
        
        # STAGE 3: Generate session summary
        session_summary = _generate_enhanced_session_summary(classified_clusters, tab_objects)
        
        # STAGE 4: Calculate multitasking score
        multitasking_score = _calculate_multitasking_score(classified_clusters, tab_objects)
        
        # Calculate overall confidence
        avg_confidence = total_confidence / len(classified_clusters) if classified_clusters else 0.0
        
        return {
            'clusters': classified_clusters,
            'multitasking_score': multitasking_score,
            'summary': session_summary,
            'confidence': avg_confidence,
            'method': 'ai_enhanced_pipeline'
        }
        
    except Exception as e:
        print(f"AI processing failed: {e}")
        # Fall back to simple domain-based clustering
        return _fallback_processing(tabs_data, options)

def _perform_intelligent_clustering(tab_objects: List[TabInfo], options: Dict[str, Any]) -> List[List[TabInfo]]:
    """
    Perform intelligent clustering using embeddings and AI services
    
    Args:
        tab_objects: List of TabInfo objects
        options: Clustering options
        
    Returns:
        List of clusters (each cluster is a list of TabInfo objects)
    """
    try:
        # Check if we have enough tabs for meaningful clustering
        if len(tab_objects) <= 1:
            return [tab_objects] if tab_objects else []
        
        # TUNABLE: Clustering parameters
        min_cluster_size = options.get('min_cluster_size', 1)
        max_clusters = options.get('max_clusters', 10)
        include_singles = options.get('include_singles', True)
        
        # Get embedding service
        embedding_service = get_embedding_service()
        
        if not embedding_service.model:
            # Fall back to domain-based clustering
            return _domain_based_clustering(tab_objects, min_cluster_size)
        
        # Prepare texts for embedding
        texts = []
        for tab in tab_objects:
            # Combine title and domain for better semantic understanding
            combined_text = f"{tab.title} {tab.domain}".strip()
            if combined_text:
                texts.append(combined_text)
            else:
                texts.append(tab.domain or tab.url or "unknown")
        
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(texts)
        
        if embeddings.size == 0:
            return _domain_based_clustering(tab_objects, min_cluster_size)
        
        # Perform clustering
        clustering_service = get_clustering_service()
        config = ProcessingConfig(
            min_cluster_size=min_cluster_size,
            clustering_algorithm="HDBSCAN" if len(tab_objects) >= 5 else "DBSCAN"
        )
        
        cluster_labels, clustering_metadata = clustering_service.cluster_embeddings(embeddings, config)
        
        # Group tabs by cluster labels
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tab_objects[i])
        
        # Convert to list of clusters
        cluster_list = list(clusters.values())
        
        # Filter small clusters if requested
        if not include_singles:
            cluster_list = [cluster for cluster in cluster_list if len(cluster) >= min_cluster_size]
        
        # Limit number of clusters
        if len(cluster_list) > max_clusters:
            # Sort by cluster size and take largest
            cluster_list.sort(key=len, reverse=True)
            cluster_list = cluster_list[:max_clusters]
        
        return cluster_list
        
    except Exception as e:
        print(f"Intelligent clustering failed: {e}")
        return _domain_based_clustering(tab_objects, min_cluster_size)

def _domain_based_clustering(tab_objects: List[TabInfo], min_cluster_size: int = 1) -> List[List[TabInfo]]:
    """Simple fallback clustering based on domain similarity"""
    
    domain_clusters = {}
    
    for tab in tab_objects:
        domain = tab.domain or "unknown"
        
        if domain not in domain_clusters:
            domain_clusters[domain] = []
        domain_clusters[domain].append(tab)
    
    # Convert to list and filter by size
    clusters = [cluster for cluster in domain_clusters.values() 
               if len(cluster) >= min_cluster_size]
    
    return clusters

def _calculate_engagement_level(tabs: List[TabInfo]) -> EngagementLevel:
    """Calculate engagement level for a cluster of tabs"""
    
    # Simple heuristic based on tab count and activity
    active_tabs = sum(1 for tab in tabs if tab.is_active)
    total_tabs = len(tabs)
    
    if total_tabs >= 5 or active_tabs >= 2:
        return EngagementLevel.HIGH
    elif total_tabs >= 2 or active_tabs >= 1:
        return EngagementLevel.MEDIUM
    else:
        return EngagementLevel.LOW

def _analyze_task_progression(tabs: List[TabInfo]) -> TaskProgression:
    """Analyze task progression based on tab patterns"""
    
    # Simple analysis based on tab titles and domains
    titles = [tab.title.lower() for tab in tabs if tab.title]
    
    # Look for research patterns
    research_keywords = ['search', 'result', 'find', 'compare', 'review', 'vs']
    if any(keyword in ' '.join(titles) for keyword in research_keywords):
        return TaskProgression.RESEARCH
    
    # Look for completion patterns
    completion_keywords = ['complete', 'finish', 'done', 'submit', 'checkout', 'order']
    if any(keyword in ' '.join(titles) for keyword in completion_keywords):
        return TaskProgression.COMPLETION
    
    # Default to in_progress
    return TaskProgression.IN_PROGRESS

def _estimate_time_spent(tabs: List[TabInfo]) -> int:
    """Estimate time spent on cluster in minutes"""
    
    # Simple estimation based on tab count and types
    # TUNABLE: Time estimation formula
    base_time = len(tabs) * 2  # 2 minutes per tab baseline
    
    # Boost for certain domains that typically require more time
    time_intensive_domains = ['youtube.com', 'netflix.com', 'coursera.org', 'udemy.com']
    for tab in tabs:
        if any(domain in tab.domain for domain in time_intensive_domains):
            base_time += 10  # Add 10 minutes for time-intensive content
    
    return min(base_time, 120)  # Cap at 2 hours

def _generate_enhanced_session_summary(clusters: List[TaskCluster], all_tabs: List[TabInfo]) -> str:
    """Generate intelligent session summary using LLM service"""
    
    try:
        llm_service = get_llm_service()
        
        # Prepare tab data for summarization
        tab_data = [tab.dict() for tab in all_tabs]
        
        # Generate summary
        summary = llm_service.summarize_session(tab_data, max_length=150)
        
        # Enhance with cluster information
        if clusters:
            cluster_info = f" Currently working on {len(clusters)} main tasks: "
            cluster_categories = [f"{cluster.category}" for cluster in clusters[:3]]
            cluster_info += ", ".join(cluster_categories)
            if len(clusters) > 3:
                cluster_info += f" and {len(clusters) - 3} others"
            
            summary += "." + cluster_info
        
        return summary
        
    except Exception as e:
        print(f"Enhanced summary generation failed: {e}")
        return _simple_session_summary(clusters, all_tabs)

def _simple_session_summary(clusters: List[TaskCluster], all_tabs: List[TabInfo]) -> str:
    """Simple fallback session summary"""
    
    if not all_tabs:
        return "No active browsing session"
    
    total_tabs = len(all_tabs)
    unique_domains = len(set(tab.domain for tab in all_tabs))
    
    if len(clusters) == 1:
        return f"Focused session with {total_tabs} tabs on {clusters[0].category}"
    elif len(clusters) <= 3:
        categories = [cluster.category for cluster in clusters]
        return f"Multi-task session: {', '.join(categories)} ({total_tabs} tabs, {unique_domains} domains)"
    else:
        return f"Complex session with {len(clusters)} different tasks ({total_tabs} tabs across {unique_domains} domains)"

def _calculate_multitasking_score(clusters: List[TaskCluster], all_tabs: List[TabInfo]) -> float:
    """Calculate multitasking score based on cluster diversity and engagement"""
    
    if not clusters or not all_tabs:
        return 0.0
    
    # TUNABLE: Multitasking score calculation
    n_clusters = len(clusters)
    n_tabs = len(all_tabs)
    unique_categories = len(set(cluster.category for cluster in clusters))
    
    # Base score from cluster count
    cluster_score = min(1.0, n_clusters / 5.0)  # TUNE: Max meaningful clusters
    
    # Diversity bonus
    diversity_score = unique_categories / max(1, n_clusters)
    
    # Engagement penalty for too many tabs
    engagement_penalty = 1.0 if n_tabs <= 10 else max(0.5, 1.0 - (n_tabs - 10) / 20.0)
    
    # Combine scores
    multitasking_score = (cluster_score * 0.5 + diversity_score * 0.3) * engagement_penalty
    
    return min(1.0, multitasking_score)

def _fallback_processing(tabs_data: List[Dict], options: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback processing when AI services fail"""
    
    # Simple domain-based grouping
    domain_groups = {}
    for tab in tabs_data:
        domain = tab.get('domain', 'unknown')
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(TabInfo(**tab))
    
    # Create simple clusters
    clusters = []
    for domain, tabs in domain_groups.items():
        if len(tabs) >= options.get('min_cluster_size', 1):
            cluster = TaskCluster(
                category="Web Browsing",
                subcategory=f"Browsing {domain}",
                confidence=0.5,
                classification_method=ClassificationMethod.FRAMEWORK_MATCH,
                tab_count=len(tabs),
                tabs=tabs,
                engagement_level=EngagementLevel.MEDIUM,
                task_progression=TaskProgression.MAINTENANCE,
                estimated_duration=f"Active for {len(tabs) * 2} minutes",
                total_time_spent=len(tabs) * 120000,  # 2 minutes per tab in milliseconds
                cluster_coherence=0.0,
                classification_metadata={"fallback": True, "method": "domain_based"}
            )
            clusters.append(cluster)
    
    return {
        'clusters': clusters,
        'multitasking_score': min(1.0, len(clusters) / 3.0),
        'summary': f"Simple domain-based analysis: {len(clusters)} task groups",
        'confidence': 0.3,
        'method': 'fallback_domain_based'
    }

def _create_fallback_response(tabs_data: List[Dict]) -> CurrentTasksResponse:
    """Create fallback response when everything fails"""
    
    return CurrentTasksResponse(
        current_tasks=[],
        task_count=0,
        dominant_task=None,
        multitasking_score=0.0,
        summary="Analysis temporarily unavailable",
        total_tabs_analyzed=len(tabs_data),
        categorization_confidence=0.0,
        new_categories_discovered=0,
        framework_categories_used=0,
        analysis_timestamp=datetime.now(),
        processing_time_ms=0
    )

@current_tasks_bp.route('/api/validate-category', methods=['POST'])
def validate_category():
    """
    Validate user feedback on category classifications
    
    Request format:
    {
        "cluster_id": "cluster_1",
        "correct_category": "Work/Professional",
        "correct_subcategory": "Meetings",
        "user_confidence": 0.9,
        "evidence": ["tab titles", "domains", "etc"],
        "feedback": "optional user comment"
    }
    """
    try:
        data = request.get_json()
        validation_request = CategoryValidationRequest(**data)
        
        # Use LLM service to validate the feedback
        llm_service = get_llm_service()
        validation_result = llm_service.validate_category(
            category=validation_request.correct_category,
            subcategory=validation_request.correct_subcategory,
            evidence=validation_request.evidence
        )
        
        # Store feedback for future model improvement
        # TODO: Implement feedback storage and learning
        
        return jsonify({
            "success": True,
            "validation_result": validation_result,
            "message": "Feedback received and validated"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Validation failed: {str(e)}"
        }), 400

@current_tasks_bp.route('/api/category-framework', methods=['GET'])
def get_category_framework():
    """Get the complete category framework for UI display"""
    
    try:
        framework_data = {
            "categories": framework.get_all_categories(),
            "framework": {
                category: {
                    "description": info.get("description", ""),
                    "keywords": info.get("keywords", [])[:10],  # Limit for UI
                    "domains": info.get("domains", [])[:5],     # Limit for UI
                    "subcategories": info.get("subcategories", [])
                }
                for category, info in framework.framework.items()
            },
            "statistics": framework.get_framework_stats()
        }
        
        return jsonify({
            "success": True,
            "data": framework_data
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get framework: {str(e)}"
        }), 500

@current_tasks_bp.route('/api/performance-stats', methods=['GET'])
def get_performance_stats():
    """Get AI services performance statistics"""
    
    try:
        classifier = get_classifier()
        embedding_service = get_embedding_service()
        clustering_service = get_clustering_service()
        
        stats = {
            "classifier_stats": classifier.get_performance_statistics(),
            "embedding_stats": embedding_service.get_cache_statistics(),
            "clustering_stats": clustering_service.get_performance_statistics(),
            "system_status": {
                "embedding_service_available": bool(embedding_service.model),
                "clustering_algorithms": clustering_service.algorithms_available,
                "framework_categories": len(framework.get_all_categories())
            }
        }
        
        return jsonify({
            "success": True,
            "data": stats
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to get performance stats: {str(e)}"
        }), 500 