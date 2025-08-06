"""
Cache Manager for AI Services

Provides efficient caching for embeddings and other AI computations with:
- LRU (Least Recently Used) eviction policy
- Memory usage monitoring
- Cache statistics and performance metrics
- Configurable size limits and TTL (Time To Live)

Key Features:
- Thread-safe operations
- Memory-efficient storage
- Hit/miss ratio tracking
- Automatic cleanup of expired entries
"""

import hashlib
import time
import threading
from typing import Any, Dict, Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import pickle
import psutil
import os

class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation
    
    Efficiently stores and retrieves cached items with automatic
    eviction of least recently used items when capacity is reached.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time to live for cache entries (None = no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        
        # Statistics tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL expiration
            if self.ttl_seconds is not None:
                access_time = self.access_times.get(key, 0)
                if time.time() - access_time > self.ttl_seconds:
                    # Remove expired entry
                    del self.cache[key]
                    del self.access_times[key]
                    self.misses += 1
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            self.hits += 1
            return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Store item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            current_time = time.time()
            
            # If key already exists, update it
            if key in self.cache:
                del self.cache[key]
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            
            # Evict least recently used items if over capacity
            while len(self.cache) > self.max_size:
                # Remove oldest item (least recently used)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            # Don't reset statistics - they're cumulative
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }

class EmbeddingCache:
    """
    Specialized cache for text embeddings with memory monitoring
    
    Optimized for storing numpy arrays representing text embeddings
    with intelligent memory management and compression.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        """
        Initialize embedding cache
        
        Args:
            max_size: Maximum number of embeddings to cache
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = LRUCache(max_size=max_size)
        
        # Memory tracking
        self.current_memory_bytes = 0
        self.memory_lock = threading.Lock()
        
        # TUNABLE: Compression settings
        self.enable_compression = True  # TUNE: False for speed, True for memory
        self.compression_threshold = 1000  # TUNE: Compress embeddings larger than this
        
    def _generate_key(self, text: str) -> str:
        """Generate consistent hash key for text"""
        # TUNABLE: Hash algorithm affects collision rate vs speed
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]  # TUNE: Longer = fewer collisions
    
    def _estimate_memory_usage(self, embedding: np.ndarray) -> int:
        """Estimate memory usage of numpy array"""
        return embedding.nbytes + 200  # Add overhead estimate
    
    def _compress_embedding(self, embedding: np.ndarray) -> bytes:
        """Compress embedding for storage"""
        if self.enable_compression and embedding.size > self.compression_threshold:
            # Use pickle with compression
            return pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Store raw bytes
            return embedding.tobytes()
    
    def _decompress_embedding(self, data: bytes, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Decompress embedding from storage"""
        try:
            # Try pickle first (compressed)
            return pickle.loads(data)
        except:
            # Fall back to raw bytes
            return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text from cache
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._generate_key(text)
        cached_data = self.cache.get(key)
        
        if cached_data is None:
            return None
        
        try:
            # Unpack stored data
            compressed_data, shape, dtype_str = cached_data
            dtype = np.dtype(dtype_str)
            
            # Decompress and return embedding
            return self._decompress_embedding(compressed_data, shape, dtype)
            
        except Exception as e:
            # Remove corrupted cache entry
            print(f"Cache corruption detected for key {key}: {e}")
            return None
    
    def set(self, text: str, embedding: np.ndarray) -> bool:
        """
        Store embedding in cache
        
        Args:
            text: Input text
            embedding: Embedding vector
            
        Returns:
            True if successfully cached, False if memory limit exceeded
        """
        key = self._generate_key(text)
        
        # Estimate memory usage
        memory_needed = self._estimate_memory_usage(embedding)
        
        with self.memory_lock:
            # Check if we need to free memory
            while (self.current_memory_bytes + memory_needed > self.max_memory_bytes and 
                   self.cache.size() > 0):
                # Force eviction of oldest item
                # Note: This is approximate since we don't track individual item memory
                self.cache.cache.popitem(last=False)  # Remove oldest
                self.current_memory_bytes *= 0.9  # Rough memory reduction estimate
            
            # If still over limit, reject the cache
            if self.current_memory_bytes + memory_needed > self.max_memory_bytes:
                return False
        
        # Compress and store embedding
        try:
            compressed_data = self._compress_embedding(embedding)
            cache_entry = (compressed_data, embedding.shape, str(embedding.dtype))
            
            self.cache.set(key, cache_entry)
            
            with self.memory_lock:
                self.current_memory_bytes += memory_needed
            
            return True
            
        except Exception as e:
            print(f"Failed to cache embedding: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        cache_stats = self.cache.get_statistics()
        
        with self.memory_lock:
            memory_usage_mb = self.current_memory_bytes / (1024 * 1024)
            memory_usage_percent = (self.current_memory_bytes / self.max_memory_bytes) * 100
        
        return {
            **cache_stats,
            "memory_usage_mb": memory_usage_mb,
            "memory_limit_mb": self.max_memory_bytes / (1024 * 1024),
            "memory_usage_percent": memory_usage_percent,
            "compression_enabled": self.enable_compression
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
        with self.memory_lock:
            self.current_memory_bytes = 0

class SystemMemoryMonitor:
    """
    Monitor system memory usage for AI operations
    
    Provides real-time memory monitoring to prevent system overload
    during intensive AI processing.
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
        # TUNABLE: Memory monitoring thresholds
        self.warning_threshold_percent = 80.0  # TUNE: Warn when system memory > 80%
        self.critical_threshold_percent = 90.0  # TUNE: Stop processing when > 90%
        
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information"""
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return {
            "system_total_gb": memory.total / (1024**3),
            "system_available_gb": memory.available / (1024**3),
            "system_used_percent": memory.percent,
            "process_rss_mb": process_memory.rss / (1024**2),
            "process_vms_mb": process_memory.vms / (1024**2)
        }
    
    def check_memory_status(self) -> Tuple[str, Dict[str, float]]:
        """
        Check current memory status
        
        Returns:
            Tuple of (status, memory_info) where status is:
            - "ok": Memory usage is normal
            - "warning": Memory usage is high but manageable
            - "critical": Memory usage is critically high
        """
        memory_info = self.get_system_memory_info()
        used_percent = memory_info["system_used_percent"]
        
        if used_percent >= self.critical_threshold_percent:
            return "critical", memory_info
        elif used_percent >= self.warning_threshold_percent:
            return "warning", memory_info
        else:
            return "ok", memory_info
    
    def should_proceed_with_operation(self, estimated_memory_mb: float = 0) -> Tuple[bool, str]:
        """
        Determine if it's safe to proceed with a memory-intensive operation
        
        Args:
            estimated_memory_mb: Estimated additional memory needed
            
        Returns:
            Tuple of (should_proceed, reason)
        """
        status, memory_info = self.check_memory_status()
        
        if status == "critical":
            return False, f"System memory critically high: {memory_info['system_used_percent']:.1f}%"
        
        # Check if estimated operation would push us over critical threshold
        if estimated_memory_mb > 0:
            available_gb = memory_info["system_available_gb"]
            estimated_gb = estimated_memory_mb / 1024
            
            if estimated_gb > available_gb * 0.8:  # TUNABLE: Safety margin
                return False, f"Operation would require {estimated_gb:.1f}GB but only {available_gb:.1f}GB available"
        
        if status == "warning":
            return True, f"Proceeding with caution: system memory at {memory_info['system_used_percent']:.1f}%"
        
        return True, "Memory status OK"

# Global instances for easy access
_embedding_cache: Optional[EmbeddingCache] = None
_memory_monitor: Optional[SystemMemoryMonitor] = None

def get_embedding_cache(max_size: int = 1000, max_memory_mb: float = 100.0) -> EmbeddingCache:
    """Get global embedding cache instance"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=max_size, max_memory_mb=max_memory_mb)
    return _embedding_cache

def get_memory_monitor() -> SystemMemoryMonitor:
    """Get global memory monitor instance"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = SystemMemoryMonitor()
    return _memory_monitor 