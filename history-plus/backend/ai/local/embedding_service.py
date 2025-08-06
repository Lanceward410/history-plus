"""
Local Embedding Service using Sentence Transformers

High-performance local text embedding generation optimized for browser history analysis.
Features comprehensive caching, batch processing, and performance monitoring.

Key Features:
- Sentence Transformers model integration
- Intelligent caching with LRU eviction
- Batch processing for efficiency
- Memory monitoring and management
- Text preprocessing integration
- Performance metrics and optimization

The service is designed to be fast enough for real-time classification
while maintaining high-quality semantic understanding.
"""

import numpy as np
import time
import gc
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import threading

# Handle optional imports gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

from ..interfaces.ai_processor import EmbeddingService, ProcessingConfig, PerformanceMonitor
from ..utils.cache_manager import get_embedding_cache, get_memory_monitor
from ..utils.text_preprocessing import get_text_preprocessor

class LocalEmbeddingService(EmbeddingService):
    """
    High-performance local embedding service using Sentence Transformers
    
    Optimized for browser history analysis with intelligent caching,
    batch processing, and memory management.
    """
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.config: Optional[ProcessingConfig] = None
        self.cache = None
        self.memory_monitor = get_memory_monitor()
        self.performance_monitor = PerformanceMonitor()
        self.preprocessor = get_text_preprocessor(aggressive=False)
        
        # Thread safety
        self.model_lock = threading.RLock()
        
        # Model information
        self.model_info = {}
        self.initialization_time = None
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.total_cache_hits = 0
        self.total_processing_time_ms = 0.0
        
    def initialize(self, config: ProcessingConfig) -> bool:
        """
        Initialize the embedding service with configuration
        
        Args:
            config: ProcessingConfig with embedding parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Error: sentence-transformers not available. Cannot initialize embedding service.")
            return False
        
        start_time = datetime.now()
        
        try:
            with self.model_lock:
                print(f"Initializing embedding service with model: {config.embedding_model}")
                
                # Store configuration
                self.config = config
                
                # Initialize cache
                self.cache = get_embedding_cache(
                    max_size=config.embedding_cache_size,
                    max_memory_mb=config.max_memory_usage_mb * 0.3  # Use 30% of total for embeddings
                )
                
                # Check memory before loading model
                proceed, reason = self.memory_monitor.should_proceed_with_operation(
                    estimated_memory_mb=self._estimate_model_memory_usage(config.embedding_model)
                )
                
                if not proceed:
                    print(f"Cannot initialize embedding service: {reason}")
                    return False
                
                if reason != "Memory status OK":
                    print(f"Memory warning during initialization: {reason}")
                
                # Load the sentence transformer model
                self.model = SentenceTransformer(config.embedding_model)
                
                # Configure model for optimal performance
                self._configure_model_performance(config)
                
                # Get model information
                self.model_info = self._extract_model_info(config)
                
                # Record initialization time
                self.initialization_time = (datetime.now() - start_time).total_seconds()
                
                print(f"Embedding service initialized successfully in {self.initialization_time:.2f}s")
                print(f"Model info: {self.model_info}")
                
                # Optional: Warm up the model with a test embedding
                if config.enable_performance_monitoring:
                    self._warmup_model()
                
                return True
                
        except Exception as e:
            print(f"Failed to initialize embedding service: {e}")
            return False
    
    def _estimate_model_memory_usage(self, model_name: str) -> float:
        """
        Estimate memory usage for different models
        
        Args:
            model_name: Name of the sentence transformer model
            
        Returns:
            Estimated memory usage in MB
        """
        # TUNABLE: Model memory estimates - adjust based on actual usage
        model_memory_estimates = {
            "all-MiniLM-L6-v2": 80,      # Lightweight model
            "all-mpnet-base-v2": 420,    # Higher quality model
            "all-MiniLM-L12-v2": 120,    # Medium model
            "paraphrase-MiniLM-L6-v2": 80,
            "multi-qa-MiniLM-L6-cos-v1": 80,
        }
        
        # Default estimate for unknown models
        return model_memory_estimates.get(model_name, 200)
    
    def _configure_model_performance(self, config: ProcessingConfig):
        """Configure model for optimal performance"""
        
        # Set maximum sequence length for speed
        # TUNABLE: Balance between context and speed
        self.model.max_seq_length = config.max_sequence_length
        
        # Configure device (CPU optimization)
        # Note: GPU support would require additional configuration
        device_info = str(self.model.device)
        if 'cuda' not in device_info.lower():
            # CPU optimizations
            if hasattr(self.model, '_modules'):
                # Set to evaluation mode for inference
                self.model.eval()
        
        if config.enable_debug_logging:
            print(f"Model configured for device: {self.model.device}")
            print(f"Max sequence length: {self.model.max_seq_length}")
    
    def _extract_model_info(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Extract information about the loaded model"""
        return {
            "model_name": config.embedding_model,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device),
            "model_type": type(self.model).__name__,
            "initialization_time_s": self.initialization_time
        }
    
    def _warmup_model(self):
        """Warm up the model with a test embedding to initialize CUDA/optimizations"""
        try:
            warmup_start = time.time()
            test_text = "This is a warmup embedding to initialize the model."
            _ = self.model.encode([test_text], convert_to_numpy=True, show_progress_bar=False)
            warmup_time = (time.time() - warmup_start) * 1000
            
            if self.config.enable_debug_logging:
                print(f"Model warmup completed in {warmup_time:.1f}ms")
                
        except Exception as e:
            print(f"Model warmup failed (non-critical): {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching and optimization
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model or not self.config:
            raise RuntimeError("Embedding service not initialized")
        
        if not texts:
            return np.array([])
        
        operation_start = self.performance_monitor.start_operation("generate_embeddings")
        
        try:
            # Preprocess texts
            preprocessing_start = time.time()
            processed_texts = self._preprocess_texts(texts)
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            # Handle cache lookups
            cache_start = time.time()
            embeddings_result, cache_misses = self._handle_cache_lookups(processed_texts)
            cache_time = (time.time() - cache_start) * 1000
            
            # Generate embeddings for cache misses
            if cache_misses:
                generation_start = time.time()
                new_embeddings = self._generate_new_embeddings(
                    [processed_texts[i] for i in cache_misses]
                )
                generation_time = (time.time() - generation_start) * 1000
                
                # Store new embeddings in cache and result array
                self._store_new_embeddings(cache_misses, processed_texts, new_embeddings, embeddings_result)
            else:
                generation_time = 0.0
            
            # Update statistics
            self._update_statistics(len(texts), len(cache_misses))
            
            # Performance monitoring
            total_time = self.performance_monitor.end_operation("generate_embeddings", operation_start)
            
            if self.config.enable_performance_monitoring:
                self.performance_monitor.add_timing("preprocessing", preprocessing_time)
                self.performance_monitor.add_timing("cache_lookup", cache_time)
                self.performance_monitor.add_timing("embedding_generation", generation_time)
            
            if self.config.enable_debug_logging:
                cache_hit_rate = (len(texts) - len(cache_misses)) / len(texts) * 100
                print(f"Generated embeddings for {len(texts)} texts in {total_time:.1f}ms "
                      f"(cache hit rate: {cache_hit_rate:.1f}%)")
            
            return embeddings_result
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            embedding_dim = self.model_info.get("embedding_dimension", 384)
            return np.zeros((len(texts), embedding_dim), dtype=np.float32)
        
        finally:
            # Optional garbage collection for memory management
            if self.config.enable_garbage_collection:
                gc.collect()
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for better embedding quality"""
        
        # Use appropriate preprocessing aggressiveness
        # TUNABLE: Adjust preprocessing based on your needs
        aggressive = len(texts) > 20  # Use aggressive for large batches
        
        preprocessor = get_text_preprocessor(aggressive=aggressive)
        processed = []
        
        for text in texts:
            cleaned = preprocessor.clean_text(text, context='title')
            # Fallback to original if cleaning results in empty string
            processed.append(cleaned if cleaned else text[:self.config.max_sequence_length])
        
        return processed
    
    def _handle_cache_lookups(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Handle cache lookups and return embeddings array with missing indices
        
        Returns:
            Tuple of (embeddings_array, indices_of_cache_misses)
        """
        embedding_dim = self.model_info["embedding_dimension"]
        embeddings_result = np.zeros((len(texts), embedding_dim), dtype=np.float32)
        cache_misses = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                embeddings_result[i] = cached_embedding
                self.total_cache_hits += 1
            else:
                cache_misses.append(i)
        
        return embeddings_result, cache_misses
    
    def _generate_new_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts not in cache"""
        
        if not texts:
            return np.array([])
        
        # Check memory before processing
        estimated_memory = len(texts) * self.model_info["embedding_dimension"] * 4 / (1024 * 1024)  # 4 bytes per float32
        proceed, reason = self.memory_monitor.should_proceed_with_operation(estimated_memory)
        
        if not proceed:
            print(f"Skipping embedding generation due to memory constraints: {reason}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.model_info["embedding_dimension"]), dtype=np.float32)
        
        # Process in batches for memory efficiency
        batch_size = min(self.config.embedding_batch_size, len(texts))
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check processing time limit
            batch_start = time.time()
            
            try:
                with self.model_lock:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(batch_texts)
                    )
                
                batch_time = (time.time() - batch_start) * 1000
                
                # Check if we're exceeding time limits
                if batch_time > self.config.max_embedding_time * 1000:
                    print(f"Warning: Embedding batch took {batch_time:.1f}ms, "
                          f"exceeding limit of {self.config.max_embedding_time * 1000:.1f}ms")
                
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                print(f"Error in embedding batch {i//batch_size + 1}: {e}")
                # Create zero embeddings for failed batch
                fallback_embeddings = np.zeros((len(batch_texts), self.model_info["embedding_dimension"]), dtype=np.float32)
                all_embeddings.append(fallback_embeddings)
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
    
    def _store_new_embeddings(self, cache_miss_indices: List[int], 
                            texts: List[str], new_embeddings: np.ndarray,
                            result_array: np.ndarray):
        """Store newly generated embeddings in cache and result array"""
        
        for i, embedding_idx in enumerate(cache_miss_indices):
            if i < len(new_embeddings):
                embedding = new_embeddings[i]
                text = texts[embedding_idx]
                
                # Store in result array
                result_array[embedding_idx] = embedding
                
                # Store in cache (non-blocking)
                try:
                    self.cache.set(text, embedding)
                except Exception as e:
                    if self.config.enable_debug_logging:
                        print(f"Failed to cache embedding: {e}")
    
    def _update_statistics(self, total_texts: int, cache_misses: int):
        """Update service statistics"""
        self.total_embeddings_generated += cache_misses
        cache_hits = total_texts - cache_misses
        self.total_cache_hits += cache_hits
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        operation_start = self.performance_monitor.start_operation("calculate_similarity")
        
        try:
            # Generate embeddings for both texts
            embeddings = self.generate_embeddings([text1, text2])
            
            if len(embeddings) != 2:
                return 0.0
            
            # Calculate cosine similarity
            embedding1, embedding2 = embeddings[0], embeddings[1]
            
            # Compute dot product and norms
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # Handle zero vectors
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity formula
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            # Cosine similarity is in [-1, 1], so we normalize to [0, 1]
            normalized_similarity = (similarity + 1) / 2
            
            return float(np.clip(normalized_similarity, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
        
        finally:
            self.performance_monitor.end_operation("calculate_similarity", operation_start)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        base_info = self.model_info.copy()
        
        # Add runtime statistics
        base_info.update({
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_cache_hits": self.total_cache_hits,
            "cache_hit_rate": self.total_cache_hits / max(1, self.total_embeddings_generated + self.total_cache_hits),
            "average_processing_time_ms": self.total_processing_time_ms / max(1, self.total_embeddings_generated),
            "service_uptime_s": (datetime.now() - datetime.fromtimestamp(time.time() - (self.initialization_time or 0))).total_seconds()
        })
        
        return base_info
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get embedding cache performance statistics"""
        if self.cache:
            cache_stats = self.cache.get_statistics()
            
            # Add memory information
            memory_info = self.memory_monitor.get_system_memory_info()
            cache_stats.update({
                "system_memory_info": memory_info,
                "service_memory_efficient": memory_info["process_rss_mb"] < 500  # TUNABLE threshold
            })
            
            return cache_stats
        else:
            return {"cache_initialized": False}
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze performance and suggest optimizations
        
        Returns:
            Dictionary with optimization suggestions
        """
        stats = self.performance_monitor.get_statistics()
        cache_stats = self.get_cache_statistics()
        memory_info = self.memory_monitor.get_system_memory_info()
        
        suggestions = []
        
        # Cache optimization suggestions
        if cache_stats.get("hit_rate", 0) < 0.5:  # TUNABLE: Cache hit rate threshold
            suggestions.append("Consider increasing cache size for better performance")
        
        # Memory optimization suggestions
        if memory_info["system_used_percent"] > 85:  # TUNABLE: Memory usage threshold
            suggestions.append("System memory usage high - consider reducing batch size")
        
        # Timing optimization suggestions
        avg_embedding_time = stats.get("operation_stats", {}).get("generate_embeddings", {}).get("avg_ms", 0)
        if avg_embedding_time > 100:  # TUNABLE: Acceptable embedding time
            suggestions.append("Embedding generation slow - consider smaller model or batch optimization")
        
        return {
            "performance_stats": stats,
            "cache_stats": cache_stats,
            "memory_info": memory_info,
            "optimization_suggestions": suggestions,
            "overall_health": "good" if len(suggestions) == 0 else "needs_attention"
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear()
            print("Embedding cache cleared")
    
    def shutdown(self):
        """Shutdown the embedding service and clean up resources"""
        if self.cache:
            cache_stats = self.cache.get_statistics()
            print(f"Embedding service shutdown. Final cache stats: {cache_stats}")
        
        # Clear model from memory
        with self.model_lock:
            self.model = None
        
        # Force garbage collection
        gc.collect()
        
        print("Embedding service shutdown complete")

# Global instance for easy access
_embedding_service: Optional[LocalEmbeddingService] = None

def get_embedding_service() -> LocalEmbeddingService:
    """Get global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = LocalEmbeddingService()
    return _embedding_service

def initialize_embedding_service(config: ProcessingConfig) -> bool:
    """Initialize the global embedding service"""
    service = get_embedding_service()
    return service.initialize(config) 