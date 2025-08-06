"""
Abstract base classes for AI processing services

This module defines the interfaces that all AI services (local and cloud) must implement.
This ensures seamless swapping between different AI backends while maintaining
consistent functionality and performance characteristics.

Key Design Principles:
- Interface compatibility between local and cloud services
- Comprehensive configuration system with tunable parameters
- Performance monitoring and quality metrics
- Graceful degradation when services fail
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ProcessingConfig:
    """
    Comprehensive configuration for AI processing parameters
    
    This class contains all tunable parameters for the AI pipeline.
    Adjust these values to optimize for your specific use case:
    - Performance vs Quality tradeoffs
    - Memory usage optimization
    - Processing time constraints
    """
    
    # =============================================================================
    # EMBEDDING CONFIGURATION - Controls text-to-vector conversion
    # =============================================================================
    
    # TUNABLE: Model selection affects quality vs speed tradeoff
    embedding_model: str = "all-MiniLM-L6-v2"  # Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (quality)
    
    # TUNABLE: Cache size affects memory usage vs speed
    embedding_cache_size: int = 1000  # Number of embeddings to cache (higher = more memory, faster repeats)
    
    # TUNABLE: Sequence length affects context vs speed
    max_sequence_length: int = 256  # Max tokens processed (higher = more context, slower processing)
    
    # TUNABLE: Batch processing affects memory vs speed
    embedding_batch_size: int = 32  # Process embeddings in batches (higher = faster, more memory)
    
    # =============================================================================
    # CLUSTERING CONFIGURATION - Controls how tabs are grouped
    # =============================================================================
    
    # TUNABLE: Algorithm selection affects cluster quality and speed
    clustering_algorithm: str = "HDBSCAN"  # Options: HDBSCAN (density-based), DBSCAN (noise-tolerant)
    
    # TUNABLE: Cluster size affects granularity of grouping
    min_cluster_size: int = 2  # Minimum tabs per cluster (lower = more clusters, higher = fewer/larger clusters)
    
    # TUNABLE: Sampling affects cluster stability
    min_samples: int = 1  # Minimum samples for core points (higher = more conservative clustering)
    
    # TUNABLE: Distance threshold for clustering algorithms
    clustering_eps: float = 0.3  # Distance threshold for DBSCAN (lower = more clusters)
    
    # TUNABLE: Clustering metric affects similarity calculation
    clustering_metric: str = "euclidean"  # Options: euclidean (geometric distance), cosine (requires precomputed)
    
    # =============================================================================
    # CLASSIFICATION CONFIGURATION - Controls category assignment
    # =============================================================================
    
    # TUNABLE: Confidence threshold for using framework vs AI generation
    confidence_threshold: float = 0.7  # Threshold for framework match (lower = more framework, higher = more AI)
    
    # TUNABLE: Feature weights for classification scoring
    domain_weight: float = 0.6  # Weight of domain matching (0.0-1.0, higher = more domain influence)
    keyword_weight: float = 0.2  # Weight of keyword matching (0.0-1.0)
    semantic_weight: float = 0.2  # Weight of semantic similarity (0.0-1.0)
    
    # Note: domain_weight + keyword_weight + semantic_weight should equal 1.0
    
    # TUNABLE: Semantic similarity thresholds
    semantic_similarity_threshold: float = 0.3  # Minimum semantic similarity for category match
    
    # =============================================================================
    # LLM CONFIGURATION - Controls language model behavior
    # =============================================================================
    
    # TUNABLE: Text generation parameters
    llm_max_tokens: int = 50  # Maximum tokens for LLM generation (higher = more detailed, slower)
    llm_temperature: float = 0.3  # Creativity vs consistency (0.0 = deterministic, 1.0 = creative)
    
    # TUNABLE: Rule-based LLM fallback parameters
    pattern_match_threshold: float = 0.5  # Minimum pattern matches for category (0.0-1.0)
    subcategory_confidence_boost: float = 0.1  # Confidence boost for good subcategory matches
    
    # =============================================================================
    # PERFORMANCE CONFIGURATION - Controls resource usage and timing
    # =============================================================================
    
    # TUNABLE: Processing time limits
    max_processing_time: float = 5.0  # Maximum seconds for classification (higher = more thorough)
    max_embedding_time: float = 2.0  # Maximum seconds for embedding generation
    max_clustering_time: float = 1.0  # Maximum seconds for clustering
    
    # TUNABLE: Batch processing limits
    max_batch_size: int = 50  # Maximum items to process at once (higher = faster, more memory)
    
    # TUNABLE: Memory management
    enable_garbage_collection: bool = True  # Whether to force garbage collection after heavy operations
    max_memory_usage_mb: float = 512.0  # Maximum memory usage in MB (approximate)
    
    # =============================================================================
    # QUALITY CONFIGURATION - Controls output quality and validation
    # =============================================================================
    
    # TUNABLE: Quality thresholds
    min_cluster_coherence: float = 0.4  # Minimum coherence score for good clusters (0.0-1.0)
    min_classification_confidence: float = 0.3  # Minimum confidence to return result
    
    # TUNABLE: Fallback behavior
    enable_fallback_classification: bool = True  # Use rule-based fallback if AI fails
    enable_quality_validation: bool = True  # Validate results before returning
    
    # =============================================================================
    # DEBUG AND MONITORING CONFIGURATION
    # =============================================================================
    
    # TUNABLE: Logging and debugging
    enable_performance_monitoring: bool = True  # Track timing and resource usage
    enable_debug_logging: bool = False  # Detailed logging for debugging
    log_cache_statistics: bool = False  # Log cache hit/miss rates
    
    # Validation method to ensure configuration is valid
    def __post_init__(self):
        """Validate configuration parameters"""
        
        # Validate weight sum
        total_weight = self.domain_weight + self.keyword_weight + self.semantic_weight
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Classification weights sum to {total_weight:.3f}, should be 1.0")
            
        # Validate ranges
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
            
        if not 0 <= self.llm_temperature <= 2:
            print(f"Warning: llm_temperature {self.llm_temperature} is outside typical range [0, 2]")
            
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")
            
        # Validate time limits
        if self.max_processing_time <= 0:
            raise ValueError("max_processing_time must be positive")

class EmbeddingService(ABC):
    """
    Abstract interface for text embedding generation services
    
    This interface ensures that both local and cloud embedding services
    provide identical functionality for seamless switching.
    """
    
    @abstractmethod
    def initialize(self, config: ProcessingConfig) -> bool:
        """
        Initialize the embedding service with given configuration
        
        Args:
            config: ProcessingConfig with embedding parameters
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded embedding model
        
        Returns:
            Dict containing model name, dimensions, device, etc.
        """
        pass
    
    @abstractmethod
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get embedding cache performance statistics
        
        Returns:
            Dict with cache hits, misses, size, etc.
        """
        pass

class ClusteringService(ABC):
    """
    Abstract interface for clustering operations
    
    Provides consistent clustering functionality across different implementations.
    """
    
    @abstractmethod
    def cluster_embeddings(self, embeddings: np.ndarray, config: ProcessingConfig) -> Tuple[List[int], Dict[str, Any]]:
        """
        Cluster embeddings into related groups
        
        Args:
            embeddings: Array of embeddings to cluster
            config: ProcessingConfig with clustering parameters
            
        Returns:
            Tuple of (cluster_labels, clustering_metadata)
        """
        pass
    
    @abstractmethod
    def get_cluster_quality_metrics(self, embeddings: np.ndarray, labels: List[int]) -> Dict[str, float]:
        """
        Calculate clustering quality metrics
        
        Args:
            embeddings: Original embeddings
            labels: Cluster labels
            
        Returns:
            Dict with quality metrics (silhouette score, etc.)
        """
        pass
    
    @abstractmethod
    def optimize_clustering_parameters(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Suggest optimal clustering parameters for given data
        
        Args:
            embeddings: Sample embeddings for parameter optimization
            
        Returns:
            Dict with suggested parameter values
        """
        pass

class LLMService(ABC):
    """
    Abstract interface for language model operations
    
    Supports both local rule-based and cloud LLM implementations.
    """
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 50, temperature: float = 0.3) -> str:
        """
        Generate text completion from prompt
        
        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate
            temperature: Generation creativity (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            str: Generated text
        """
        pass
    
    @abstractmethod
    def generate_category_name(self, titles: List[str], domains: List[str], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate category and subcategory for tab cluster
        
        Args:
            titles: List of tab titles
            domains: List of tab domains
            context: Optional context information
            
        Returns:
            Dict with category, subcategory, confidence, and method
        """
        pass
    
    @abstractmethod
    def summarize_session(self, tab_data: List[Dict], max_length: int = 100) -> str:
        """
        Generate natural language summary of browsing session
        
        Args:
            tab_data: List of tab information dictionaries
            max_length: Maximum summary length in characters
            
        Returns:
            str: Session summary
        """
        pass
    
    @abstractmethod
    def validate_category(self, category: str, subcategory: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Validate whether a category assignment makes sense
        
        Args:
            category: Proposed category
            subcategory: Proposed subcategory
            evidence: Evidence text (titles, domains, etc.)
            
        Returns:
            Dict with validation result and confidence
        """
        pass

@dataclass
class ProcessingResult:
    """
    Standardized result format for AI processing operations
    
    Contains both the results and metadata for performance monitoring
    and quality assessment.
    """
    
    # Core results
    success: bool
    category: Optional[str] = None
    subcategory: Optional[str] = None
    confidence: float = 0.0
    
    # Processing metadata
    method: str = "unknown"  # How the result was generated
    processing_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    
    # Quality metrics
    cluster_coherence: float = 0.0
    semantic_similarity: float = 0.0
    
    # Detailed information
    cluster_labels: List[int] = field(default_factory=list)
    embeddings_shape: Optional[Tuple[int, int]] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance breakdown
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def add_timing(self, operation: str, time_ms: float):
        """Add timing information for an operation"""
        self.timing_breakdown[operation] = time_ms
        
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.success = False
        
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)

class PerformanceMonitor:
    """
    Utility class for monitoring AI service performance
    
    Tracks timing, memory usage, and quality metrics across
    AI operations for optimization and debugging.
    """
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_usage: List[float] = []
        self.quality_scores: List[float] = []
        
    def start_operation(self, operation_name: str) -> datetime:
        """Start timing an operation"""
        return datetime.now()
        
    def end_operation(self, operation_name: str, start_time: datetime) -> float:
        """End timing an operation and record the duration"""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(duration_ms)
        
        return duration_ms
        
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        self.memory_usage.append(memory_mb)
        
    def record_quality_score(self, score: float):
        """Record a quality metric"""
        self.quality_scores.append(score)
        
    def add_timing(self, operation: str, time_ms: float):
        """Add timing information for an operation"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(time_ms)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "operation_stats": {},
            "memory_stats": {},
            "quality_stats": {}
        }
        
        # Operation timing statistics
        for operation, times in self.operation_times.items():
            if times:
                stats["operation_stats"][operation] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
        
        # Memory usage statistics
        if self.memory_usage:
            stats["memory_stats"] = {
                "avg_mb": sum(self.memory_usage) / len(self.memory_usage),
                "max_mb": max(self.memory_usage),
                "min_mb": min(self.memory_usage)
            }
        
        # Quality statistics
        if self.quality_scores:
            stats["quality_stats"] = {
                "avg_score": sum(self.quality_scores) / len(self.quality_scores),
                "max_score": max(self.quality_scores),
                "min_score": min(self.quality_scores)
            }
        
        return stats 