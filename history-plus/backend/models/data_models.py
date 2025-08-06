"""
Enhanced data models for History Plus hierarchical category system
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ClassificationMethod(str, Enum):
    """How the category was determined"""
    FRAMEWORK_MATCH = "framework_match"    # Matched existing category framework
    AI_GENERATED = "ai_generated"          # AI created new category
    USER_VALIDATED = "user_validated"      # User confirmed/corrected
    HYBRID = "hybrid"                      # Combination of methods
    EMPTY = "empty"                        # No classification possible

class EngagementLevel(str, Enum):
    """User engagement intensity"""
    HIGH = "high"        # Active interaction, long time spent, frequent switching
    MEDIUM = "medium"    # Moderate engagement, some interaction
    LOW = "low"         # Brief visits, minimal interaction

class TaskProgression(str, Enum):
    """Type of browsing behavior detected"""
    DEEP_DIVE = "deep_dive"          # Focused, extended research on one topic
    EXPLORATION = "exploration"       # Browsing, discovering new content
    QUICK_LOOKUP = "quick_lookup"     # Fast fact-finding, brief searches
    MAINTENANCE = "maintenance"       # Routine tasks, checking updates
    MULTITASKING = "multitasking"    # Juggling multiple different activities

class TabInfo(BaseModel):
    """Information about a single browser tab"""
    id: int
    title: str
    url: str
    domain: str
    is_active: bool
    
    # Engagement metrics
    time_on_page: Optional[int] = None      # Milliseconds
    engagement_score: Optional[float] = None # 0.0-1.0
    scroll_depth: Optional[float] = None     # 0.0-1.0
    interaction_count: Optional[int] = None  # Mouse/keyboard events
    
    # Timing information
    opened_at: Optional[datetime] = None
    last_active: Optional[datetime] = None

class TaskCluster(BaseModel):
    """A group of related browser tabs representing a user task"""
    
    # Core classification - THE HEART OF THE HIERARCHICAL SYSTEM
    category: str = Field(..., description="Main category (e.g., 'Gaming', 'Social Media', or AI-discovered)")
    subcategory: str = Field(..., description="Specific activity (e.g., 'Quest Research', 'Professional Networking')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    classification_method: ClassificationMethod
    
    # Tab information
    tab_count: int = Field(..., ge=0)
    tabs: List[TabInfo]
    
    # Engagement analytics
    engagement_level: EngagementLevel
    task_progression: TaskProgression
    estimated_duration: Optional[str] = None       # "Active for 2.5 hours"
    total_time_spent: Optional[int] = None         # Total milliseconds across all tabs
    
    # Metadata for analysis and learning
    created_at: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    related_clusters: List[str] = Field(default_factory=list)  # IDs of similar past clusters
    
    # Quality and validation metrics
    cluster_coherence: Optional[float] = None      # How well tabs fit together (0.0-1.0)
    user_validation: Optional[bool] = None         # User confirmed/corrected classification
    framework_matched: bool = Field(default=False) # True if matched existing framework
    
    # Detailed classification metadata
    classification_metadata: Dict[str, Any] = Field(default_factory=dict)

class CategoryHierarchy(BaseModel):
    """Tracks discovered categories for consistency and learning"""
    category: str = Field(..., description="Category name")
    subcategories_seen: List[str] = Field(default_factory=list)  # All subcategories under this category
    example_domains: List[str] = Field(default_factory=list)     # Domains typically found here
    keywords_associated: List[str] = Field(default_factory=list) # Keywords that trigger this category
    
    # Usage statistics for personalization
    user_frequency: int = Field(default=0)         # How often user does this activity
    last_seen: datetime = Field(default_factory=datetime.now)
    confidence_scores: List[float] = Field(default_factory=list)  # Historical confidence
    
    # Learning and improvement data
    is_framework_category: bool = Field(default=False)  # True if from CATEGORY_FRAMEWORK
    user_corrections: int = Field(default=0)            # Times user corrected this classification
    successful_matches: int = Field(default=0)          # Times this category was correctly used
    
    # AI learning metadata
    embedding_keywords: List[str] = Field(default_factory=list)  # Keywords that embed well
    domain_patterns: List[str] = Field(default_factory=list)     # Common domain patterns

class CurrentTasksResponse(BaseModel):
    """API response for current task analysis"""
    current_tasks: List[TaskCluster]
    task_count: int = Field(..., ge=0)
    dominant_task: Optional[TaskCluster] = None     # Task with most tabs/engagement
    multitasking_score: float = Field(..., ge=0.0, le=1.0)  # How scattered attention is
    summary: str                                    # Natural language summary
    
    # Analytics and insights
    total_tabs_analyzed: int
    categorization_confidence: float               # Average confidence across all tasks
    new_categories_discovered: int = Field(default=0)
    framework_categories_used: int = Field(default=0)
    
    # Real-time metrics
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None

class CategoryValidationRequest(BaseModel):
    """User feedback for category validation"""
    cluster_id: str
    original_category: str
    original_subcategory: str
    
    # User corrections
    corrected_category: Optional[str] = None
    corrected_subcategory: Optional[str] = None
    user_confidence: Optional[float] = None         # User's confidence in their correction
    
    # Feedback metadata
    feedback_type: str = Field(default="correction")  # "correction", "confirmation", "suggestion"
    user_comment: Optional[str] = None

class EngagementMetrics(BaseModel):
    """Detailed engagement metrics for a tab or cluster"""
    
    # Time-based metrics
    total_time: int = Field(default=0)              # Total milliseconds
    active_time: int = Field(default=0)             # Time with focus/interaction
    idle_time: int = Field(default=0)               # Time without interaction
    
    # Interaction metrics  
    scroll_events: int = Field(default=0)
    click_events: int = Field(default=0)
    key_events: int = Field(default=0)
    tab_switches: int = Field(default=0)
    
    # Content metrics
    scroll_depth_max: float = Field(default=0.0, ge=0.0, le=1.0)
    scroll_speed: Optional[float] = None            # Average scroll speed
    page_height: Optional[int] = None               # Estimated page height in pixels
    
    # Engagement quality indicators
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    attention_quality: str = Field(default="unknown")  # "focused", "skimming", "distracted"
    task_relevance: Optional[float] = None          # How relevant to the current task

class SessionData(BaseModel):
    """Enhanced session data with hierarchical categories"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Task clusters in this session
    task_clusters: List[TaskCluster] = Field(default_factory=list)
    dominant_category: Optional[str] = None
    category_distribution: Dict[str, int] = Field(default_factory=dict)  # Category -> tab count
    
    # Session-level metrics
    total_tabs: int = Field(default=0)
    unique_domains: int = Field(default=0)
    multitasking_intensity: float = Field(default=0.0)
    
    # Productivity insights
    productive_time: Optional[int] = None           # Time on work/learning categories
    entertainment_time: Optional[int] = None       # Time on entertainment categories
    productivity_score: Optional[float] = None     # Overall productivity rating

class ProcessingResult(BaseModel):
    """Result of AI processing on browsing data"""
    success: bool
    processing_time_ms: int
    
    # Core results
    task_clusters: List[TaskCluster] = Field(default_factory=list)
    category_insights: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality metrics
    clustering_quality: Optional[float] = None     # How well tabs were clustered
    classification_accuracy: Optional[float] = None # Estimated accuracy of classifications
    
    # AI model information
    models_used: List[str] = Field(default_factory=list)  # Which AI models were used
    processing_mode: str = Field(default="local")  # "local" or "cloud"
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# Configuration and settings models
class AIConfiguration(BaseModel):
    """AI processing configuration"""
    mode: str = Field(default="local")              # "local" or "cloud"
    
    # Model settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    clustering_algorithm: str = Field(default="HDBSCAN")
    
    # Tunable parameters
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    domain_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Cloud settings (if applicable)
    openai_api_key: Optional[str] = None
    max_tokens: int = Field(default=500)
    
    # Privacy settings
    send_full_content: bool = Field(default=False)  # Whether to send page content to cloud
    anonymize_domains: bool = Field(default=False)

class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'success': self.success,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.data is not None:
            result['data'] = self.data
            
        if self.errors:
            result['errors'] = self.errors
            
        if self.processing_time is not None:
            result['processing_time'] = self.processing_time
            
        return result

class HistoryEntry(BaseModel):
    """A single history entry with enhanced metadata"""
    url: str
    title: str
    domain: str
    timestamp: int  # Unix timestamp in milliseconds
    visit_count: int = 1
    last_visit_time: Optional[int] = None
    
    # Categorization
    category: Optional[str] = None
    subcategory: Optional[str] = None
    confidence: Optional[float] = None
    classification_method: Optional[ClassificationMethod] = None
    
    # Session context
    session_id: Optional[str] = None
    tab_id: Optional[int] = None
    
    # Engagement data
    time_on_page: Optional[int] = None  # Milliseconds
    engagement_score: Optional[float] = None
    
class BrowsingSession(BaseModel):
    """A browsing session with multiple history entries"""
    id: str
    start_time: int  # Unix timestamp
    end_time: Optional[int] = None
    
    # Entries in this session
    entries: List[HistoryEntry] = Field(default_factory=list)
    
    # Session metrics
    total_time: int = 0  # Total session duration in milliseconds
    unique_domains: int = 0
    total_pages: int = 0
    
    # Categorization
    dominant_category: Optional[str] = None
    categories: Dict[str, int] = Field(default_factory=dict)  # Category -> count
    
    # Engagement
    avg_engagement: float = 0.0
    engagement_distribution: Dict[str, float] = Field(default_factory=dict)
    
    # Analysis results
    topic_summary: Optional[str] = None
    session_type: str = "browsing"  # "work", "entertainment", "research", etc.

class EngagementData(BaseModel):
    """Engagement metrics for a specific page/session"""
    url: str
    domain: str
    timestamp: int
    
    # Time metrics
    time_on_page: int = 0  # Milliseconds
    active_time: int = 0   # Time with focus
    idle_time: int = 0     # Time without interaction
    background_time: int = 0  # Time in background tab
    
    # Interaction metrics
    scroll_depth: float = 0.0  # 0.0 to 1.0
    mouse_movements: int = 0
    key_presses: int = 0
    clicks: int = 0
    
    # Context
    page_type: str = "webpage"  # "search", "article", "social", etc.
    referrer: Optional[str] = None
    
    # Calculated scores
    engagement_score: float = 0.0  # 0.0 to 1.0
    attention_quality: str = "unknown"  # "focused", "skimming", "distracted"
    contextual_score: float = 0.0
    confidence: float = 0.0 