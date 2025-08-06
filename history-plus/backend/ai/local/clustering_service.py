"""
Local Clustering Service for Tab Grouping

Advanced clustering algorithms optimized for browser tab semantic grouping.
Supports multiple algorithms with automatic parameter optimization and quality metrics.

Key Features:
- HDBSCAN for density-based clustering with varying densities
- DBSCAN for traditional density-based clustering
- Automatic parameter optimization based on data characteristics
- Comprehensive quality metrics and validation
- Memory-efficient processing for large tab sets
- Noise detection and handling

The service is designed to group semantically similar tabs while handling
noise and outliers effectively.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import warnings

# Handle optional imports gracefully
try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.metrics.pairwise import cosine_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")

from ..interfaces.ai_processor import ClusteringService, ProcessingConfig, PerformanceMonitor

class LocalClusteringService(ClusteringService):
    """
    Local clustering service with multiple algorithm support and optimization
    
    Provides intelligent clustering of browser tabs based on semantic embeddings
    with automatic parameter tuning and quality assessment.
    """
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        
        # Algorithm availability
        self.algorithms_available = {
            "HDBSCAN": HDBSCAN_AVAILABLE,
            "DBSCAN": SKLEARN_AVAILABLE
        }
        
        # Parameter optimization cache
        self.optimization_cache = {}
        
        # Performance tracking
        self.clustering_history = []
        
        # TUNABLE: Default algorithm preferences
        self.algorithm_preferences = ["HDBSCAN", "DBSCAN"]  # Order of preference
        
    def cluster_embeddings(self, embeddings: np.ndarray, config: ProcessingConfig) -> Tuple[List[int], Dict[str, Any]]:
        """
        Cluster embeddings into related groups using the best available algorithm
        
        Args:
            embeddings: Array of embeddings to cluster
            config: ProcessingConfig with clustering parameters
            
        Returns:
            Tuple of (cluster_labels, clustering_metadata)
        """
        if embeddings.size == 0:
            return [], {"algorithm": "empty", "n_clusters": 0}
        
        if len(embeddings) < 2:
            return [0] * len(embeddings), {"algorithm": "single_item", "n_clusters": 1}
        
        operation_start = self.performance_monitor.start_operation("cluster_embeddings")
        
        try:
            # Choose best available algorithm
            algorithm = self._select_algorithm(config, len(embeddings))
            
            if algorithm == "HDBSCAN":
                labels, metadata = self._cluster_hdbscan(embeddings, config)
            elif algorithm == "DBSCAN":
                labels, metadata = self._cluster_dbscan(embeddings, config)
            else:
                # Fallback to simple clustering
                labels, metadata = self._fallback_clustering(embeddings)
            
            # Post-process clustering results
            labels, metadata = self._post_process_clustering(labels, metadata, embeddings)
            
            # Record clustering performance
            processing_time = self.performance_monitor.end_operation("cluster_embeddings", operation_start)
            metadata["processing_time_ms"] = processing_time
            
            # Store in history for optimization
            self._record_clustering_performance(embeddings, labels, metadata, config)
            
            return labels, metadata
            
        except Exception as e:
            print(f"Clustering failed: {e}, using fallback")
            return self._fallback_clustering(embeddings)
    
    def _select_algorithm(self, config: ProcessingConfig, n_items: int) -> str:
        """
        Select the best clustering algorithm based on configuration and data size
        
        Args:
            config: Processing configuration
            n_items: Number of items to cluster
            
        Returns:
            Algorithm name to use
        """
        # Check user preference first
        preferred_algorithm = config.clustering_algorithm.upper()
        
        if preferred_algorithm in self.algorithms_available and self.algorithms_available[preferred_algorithm]:
            return preferred_algorithm
        
        # Use algorithm preferences based on availability and data size
        for algorithm in self.algorithm_preferences:
            if self.algorithms_available.get(algorithm, False):
                # TUNABLE: Algorithm selection based on data characteristics
                if algorithm == "HDBSCAN" and n_items >= 10:  # TUNE: Minimum items for HDBSCAN
                    return algorithm
                elif algorithm == "DBSCAN" and n_items >= 5:  # TUNE: Minimum items for DBSCAN
                    return algorithm
        
        # Fallback
        return "fallback"
    
    def _cluster_hdbscan(self, embeddings: np.ndarray, config: ProcessingConfig) -> Tuple[List[int], Dict[str, Any]]:
        """
        HDBSCAN clustering - excellent for varying density clusters
        
        HDBSCAN is particularly good for:
        - Clusters of varying densities
        - Noise detection
        - Hierarchical cluster structure
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available")
        
        # Optimize parameters if not cached
        cache_key = f"hdbscan_{len(embeddings)}_{embeddings.shape[1]}"
        if cache_key in self.optimization_cache:
            optimal_params = self.optimization_cache[cache_key]
        else:
            optimal_params = self._optimize_hdbscan_parameters(embeddings, config)
            self.optimization_cache[cache_key] = optimal_params
        
        # TUNABLE PARAMETERS: These significantly affect clustering quality
        min_cluster_size = max(2, optimal_params.get("min_cluster_size", config.min_cluster_size))
        min_samples = optimal_params.get("min_samples", config.min_samples)
        cluster_selection_epsilon = optimal_params.get("cluster_selection_epsilon", 0.0)
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=config.clustering_metric,
                cluster_selection_method='eom',  # TUNABLE: 'eom' vs 'leaf'
                alpha=1.0,  # TUNABLE: Higher = more conservative, lower = more clusters
                cluster_selection_epsilon=cluster_selection_epsilon  # TUNABLE: Can help merge clusters
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Handle noise points (label = -1) by creating singleton clusters
            processed_labels = self._handle_noise_points(cluster_labels)
            
            # Extract additional metadata
            metadata = {
                "algorithm": "HDBSCAN",
                "n_clusters": len(set(processed_labels)),
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "n_noise_points": sum(1 for label in cluster_labels if label == -1),
                "cluster_persistence": self._safe_to_list(clusterer.cluster_persistence_ if hasattr(clusterer, 'cluster_persistence_') else None),
                "exemplars": self._safe_to_list(clusterer.exemplars_ if hasattr(clusterer, 'exemplars_') else None),
                "optimization_used": True
            }
            
            return processed_labels, metadata
            
        except Exception as e:
            print(f"HDBSCAN clustering failed: {e}")
            # Fallback to simpler parameters
            return self._simple_hdbscan(embeddings, config)
    
    def _optimize_hdbscan_parameters(self, embeddings: np.ndarray, config: ProcessingConfig) -> Dict[str, Any]:
        """
        Optimize HDBSCAN parameters for the given data
        
        Uses heuristics and validation metrics to find good parameters
        """
        n_items = len(embeddings)
        
        # TUNABLE: Parameter search ranges
        min_cluster_sizes = [
            max(2, int(n_items * 0.05)),  # 5% of items
            max(2, int(n_items * 0.1)),   # 10% of items
            max(2, config.min_cluster_size),  # User preference
            max(2, min(5, n_items // 4))  # Adaptive based on size
        ]
        
        min_samples_options = [1, max(1, config.min_samples), max(1, min(3, n_items // 10))]
        
        best_params = {"min_cluster_size": config.min_cluster_size, "min_samples": config.min_samples}
        best_score = -1
        
        # Quick parameter search
        for min_cluster_size in min_cluster_sizes[:2]:  # Limit search for performance
            for min_samples in min_samples_options[:2]:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric=config.clustering_metric
                    )
                    
                    labels = clusterer.fit_predict(embeddings)
                    
                    # Calculate simple quality score
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = sum(1 for label in labels if label == -1)
                    
                    # Simple scoring: prefer moderate number of clusters with less noise
                    if n_clusters > 0:
                        noise_ratio = n_noise / len(labels)
                        cluster_ratio = n_clusters / len(labels)
                        
                        # TUNABLE: Scoring function for parameter optimization
                        score = (1 - noise_ratio) * 0.7 + (1 - abs(cluster_ratio - 0.3)) * 0.3
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "min_cluster_size": min_cluster_size,
                                "min_samples": min_samples,
                                "cluster_selection_epsilon": 0.0
                            }
                
                except Exception:
                    continue
        
        return best_params
    
    def _simple_hdbscan(self, embeddings: np.ndarray, config: ProcessingConfig) -> Tuple[List[int], Dict[str, Any]]:
        """Simple HDBSCAN with basic parameters as fallback"""
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, config.min_cluster_size),
                min_samples=config.min_samples,
                metric='euclidean'  # Simpler metric
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            processed_labels = self._handle_noise_points(cluster_labels)
            
            metadata = {
                "algorithm": "HDBSCAN_simple",
                "n_clusters": len(set(processed_labels)),
                "n_noise_points": sum(1 for label in cluster_labels if label == -1),
                "optimization_used": False
            }
            
            return processed_labels, metadata
            
        except Exception as e:
            print(f"Simple HDBSCAN also failed: {e}")
            return self._fallback_clustering(embeddings)
    
    def _cluster_dbscan(self, embeddings: np.ndarray, config: ProcessingConfig) -> Tuple[List[int], Dict[str, Any]]:
        """
        DBSCAN clustering - good for noise detection and uniform density
        
        DBSCAN is good for:
        - Noise detection
        - Clusters of similar density
        - When cluster count is unknown
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        # Optimize eps parameter
        optimal_eps = self._optimize_dbscan_eps(embeddings, config)
        
        # TUNABLE PARAMETERS: Critical for DBSCAN performance
        eps = optimal_eps
        min_samples = max(1, config.min_samples)
        
        try:
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=config.clustering_metric
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Handle noise points
            processed_labels = self._handle_noise_points(cluster_labels)
            
            metadata = {
                "algorithm": "DBSCAN",
                "n_clusters": len(set(processed_labels)),
                "eps": eps,
                "min_samples": min_samples,
                "n_noise_points": sum(1 for label in cluster_labels if label == -1),
                "optimization_used": True
            }
            
            return processed_labels, metadata
            
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
            return self._fallback_clustering(embeddings)
    
    def _optimize_dbscan_eps(self, embeddings: np.ndarray, config: ProcessingConfig) -> float:
        """
        Optimize eps parameter for DBSCAN using k-distance analysis
        
        Args:
            embeddings: Input embeddings
            config: Processing configuration
            
        Returns:
            Optimal eps value
        """
        try:
            # Calculate distance matrix
            if config.clustering_metric == 'cosine':
                distances = cosine_distances(embeddings)
            else:
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(embeddings)
            
            # For each point, find distance to k-th nearest neighbor
            k = max(1, config.min_samples)
            k_distances = []
            
            for i in range(len(distances)):
                # Sort distances for this point (excluding self)
                point_distances = np.sort(distances[i])
                if len(point_distances) > k:
                    k_distances.append(point_distances[k])  # k-th nearest (0-indexed, so k gives k+1-th)
                else:
                    k_distances.append(point_distances[-1])  # Furthest point if not enough neighbors
            
            k_distances.sort()
            
            # TUNABLE: Method for selecting eps from k-distance plot
            # Common heuristics:
            # 1. Knee/elbow of the curve
            # 2. Percentile of distances
            # 3. Mean + std of distances
            
            # Method 1: Use percentile (simple and robust)
            eps_percentile = 75  # TUNABLE: Percentile for eps selection
            optimal_eps = np.percentile(k_distances, eps_percentile)
            
            # Method 2: Look for "knee" in the curve (more sophisticated)
            if len(k_distances) > 10:
                # Simple knee detection: largest second derivative
                second_derivatives = []
                for i in range(1, len(k_distances) - 1):
                    second_deriv = k_distances[i+1] - 2*k_distances[i] + k_distances[i-1]
                    second_derivatives.append(abs(second_deriv))
                
                if second_derivatives:
                    knee_idx = np.argmax(second_derivatives) + 1
                    knee_eps = k_distances[knee_idx]
                    
                    # Use knee if it's reasonable, otherwise use percentile
                    if 0.1 <= knee_eps <= 2.0:  # TUNABLE: Reasonable eps range
                        optimal_eps = knee_eps
            
            # Ensure eps is within reasonable bounds
            # TUNABLE: Eps bounds based on your data characteristics
            min_eps = 0.05  # Minimum meaningful distance
            max_eps = 1.0   # Maximum reasonable distance for cosine metric
            
            optimal_eps = np.clip(optimal_eps, min_eps, max_eps)
            
            return float(optimal_eps)
            
        except Exception as e:
            print(f"Eps optimization failed: {e}")
            # Return default eps
            return config.clustering_eps if hasattr(config, 'clustering_eps') else 0.3
    
    def _handle_noise_points(self, labels: np.ndarray) -> List[int]:
        """
        Handle noise points by assigning them to singleton clusters
        
        Args:
            labels: Cluster labels from clustering algorithm (may contain -1 for noise)
            
        Returns:
            Processed labels with no noise points
        """
        processed_labels = labels.copy()
        
        # Find noise points (labeled as -1)
        noise_mask = processed_labels == -1
        
        if np.any(noise_mask):
            # Assign each noise point to its own cluster
            max_label = np.max(processed_labels[~noise_mask]) if np.any(~noise_mask) else -1
            next_label = max_label + 1
            
            for i, is_noise in enumerate(noise_mask):
                if is_noise:
                    processed_labels[i] = next_label
                    next_label += 1
        
        # Safe conversion to list
        return self._safe_to_list(processed_labels)
    
    def _post_process_clustering(self, labels: List[int], metadata: Dict[str, Any], 
                               embeddings: np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        """
        Post-process clustering results for quality and consistency
        
        Args:
            labels: Cluster labels
            metadata: Clustering metadata
            embeddings: Original embeddings
            
        Returns:
            Tuple of (processed_labels, updated_metadata)
        """
        # Calculate cluster quality metrics
        quality_metrics = self.get_cluster_quality_metrics(embeddings, labels)
        metadata.update(quality_metrics)
        
        # Check for and handle degenerate clusterings
        n_clusters = len(set(labels))
        n_items = len(labels)
        
        # TUNABLE: Quality thresholds for post-processing
        min_cluster_quality = 0.2  # Minimum silhouette score
        max_cluster_ratio = 0.8    # Maximum fraction of items in one cluster
        
        # Check if clustering is too degenerate
        cluster_counts = {}
        for label in labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        max_cluster_size = max(cluster_counts.values())
        if max_cluster_size / n_items > max_cluster_ratio:
            metadata["warning"] = f"Clustering is degenerate: {max_cluster_size}/{n_items} items in largest cluster"
        
        # Check silhouette score quality
        silhouette = quality_metrics.get("silhouette_score", 0)
        if silhouette < min_cluster_quality and n_clusters > 1:
            metadata["warning"] = f"Low clustering quality: silhouette score {silhouette:.3f}"
        
        return labels, metadata
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> Tuple[List[int], Dict[str, Any]]:
        """
        Simple fallback clustering when advanced algorithms fail
        
        Uses basic similarity thresholding to create clusters
        """
        try:
            n_items = len(embeddings)
            
            if n_items <= 1:
                return [0] * n_items, {"algorithm": "fallback_single", "n_clusters": min(1, n_items)}
            
            # Simple agglomerative clustering using distance threshold
            # TUNABLE: Similarity threshold for fallback clustering
            similarity_threshold = 0.7  # Cosine similarity threshold
            
            labels = [-1] * n_items
            current_label = 0
            
            for i in range(n_items):
                if labels[i] == -1:  # Unassigned
                    # Start new cluster
                    labels[i] = current_label
                    
                    # Find similar items
                    for j in range(i + 1, n_items):
                        if labels[j] == -1:  # Unassigned
                            # Calculate cosine similarity
                            similarity = np.dot(embeddings[i], embeddings[j]) / (
                                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                            )
                            
                            if similarity >= similarity_threshold:
                                labels[j] = current_label
                    
                    current_label += 1
            
            # Handle any remaining unassigned items
            for i in range(n_items):
                if labels[i] == -1:
                    labels[i] = current_label
                    current_label += 1
            
            metadata = {
                "algorithm": "fallback_similarity",
                "n_clusters": len(set(labels)),
                "similarity_threshold": similarity_threshold,
                "warning": "Advanced clustering algorithms not available"
            }
            
            return labels, metadata
            
        except Exception as e:
            print(f"Even fallback clustering failed: {e}")
            # Ultimate fallback: everything in one cluster
            return [0] * len(embeddings), {
                "algorithm": "fallback_single_cluster",
                "n_clusters": 1,
                "error": str(e)
            }
    
    def get_cluster_quality_metrics(self, embeddings: np.ndarray, labels: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive clustering quality metrics
        
        Args:
            embeddings: Original embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary with quality metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"quality_metrics_available": False}
        
        metrics = {}
        
        try:
            unique_labels = set(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return {
                    "silhouette_score": 0.0,
                    "calinski_harabasz_score": 0.0,
                    "davies_bouldin_score": float('inf'),
                    "n_clusters": n_clusters,
                    "n_items": len(labels)
                }
            
            # Suppress sklearn warnings for small clusters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Silhouette score: [-1, 1], higher is better
                try:
                    silhouette = silhouette_score(embeddings, labels, metric='cosine')
                    metrics["silhouette_score"] = float(silhouette)
                except Exception:
                    # Fallback to euclidean if cosine fails
                    try:
                        silhouette = silhouette_score(embeddings, labels, metric='euclidean')
                    metrics["silhouette_score"] = float(silhouette)
                except Exception:
                    metrics["silhouette_score"] = 0.0
                
                # Calinski-Harabasz score: higher is better
                try:
                    calinski = calinski_harabasz_score(embeddings, labels)
                    metrics["calinski_harabasz_score"] = float(calinski)
                except Exception:
                    metrics["calinski_harabasz_score"] = 0.0
                
                # Davies-Bouldin score: lower is better
                try:
                    davies_bouldin = davies_bouldin_score(embeddings, labels)
                    metrics["davies_bouldin_score"] = float(davies_bouldin)
                except Exception:
                    metrics["davies_bouldin_score"] = float('inf')
            
            # Additional custom metrics
            metrics.update(self._calculate_custom_metrics(embeddings, labels))
            
            # Cluster distribution metrics
            cluster_counts = {}
            for label in labels:
                cluster_counts[label] = cluster_counts.get(label, 0) + 1
            
            cluster_sizes = list(cluster_counts.values())
            metrics.update({
                "n_clusters": n_clusters,
                "n_items": len(labels),
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
                "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes),
                "cluster_size_std": np.std(cluster_sizes)
            })
            
        except Exception as e:
            print(f"Error calculating cluster metrics: {e}")
            metrics = {
                "error": str(e),
                "n_clusters": len(set(labels)),
                "n_items": len(labels)
            }
        
        return metrics
    
    def _calculate_custom_metrics(self, embeddings: np.ndarray, labels: List[int]) -> Dict[str, float]:
        """Calculate custom clustering quality metrics"""
        
        metrics = {}
        
        try:
            # Intra-cluster cohesion and inter-cluster separation
            unique_labels = list(set(labels))
            
            intra_cluster_distances = []
            inter_cluster_distances = []
            
            for label in unique_labels:
                cluster_mask = np.array(labels) == label
                cluster_embeddings = embeddings[cluster_mask]
                
                if len(cluster_embeddings) > 1:
                    # Calculate intra-cluster distances (within cluster)
                    for i in range(len(cluster_embeddings)):
                        for j in range(i + 1, len(cluster_embeddings)):
                            distance = 1 - np.dot(cluster_embeddings[i], cluster_embeddings[j]) / (
                                np.linalg.norm(cluster_embeddings[i]) * np.linalg.norm(cluster_embeddings[j])
                            )
                            intra_cluster_distances.append(distance)
            
            # Calculate inter-cluster distances (between clusters)
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels[i+1:], i+1):
                    mask1 = np.array(labels) == label1
                    mask2 = np.array(labels) == label2
                    
                    embeddings1 = embeddings[mask1]
                    embeddings2 = embeddings[mask2]
                    
                    # Calculate centroid distance with safety checks
                    centroid1 = np.mean(embeddings1, axis=0)
                    centroid2 = np.mean(embeddings2, axis=0)
                    
                    norm1 = np.linalg.norm(centroid1)
                    norm2 = np.linalg.norm(centroid2)
                    
                    if norm1 == 0 or norm2 == 0:
                        distance = 1.0  # Maximum distance for zero vectors
                    else:
                        distance = 1 - np.dot(centroid1, centroid2) / (norm1 * norm2)
                        # Clamp to valid range
                        distance = max(0.0, min(2.0, distance))
                    
                    inter_cluster_distances.append(distance)
            
            # Calculate metrics
            if intra_cluster_distances:
                metrics["avg_intra_cluster_distance"] = float(np.mean(intra_cluster_distances))
                metrics["std_intra_cluster_distance"] = float(np.std(intra_cluster_distances))
            
            if inter_cluster_distances:
                metrics["avg_inter_cluster_distance"] = float(np.mean(inter_cluster_distances))
                metrics["std_inter_cluster_distance"] = float(np.std(inter_cluster_distances))
            
            # Cohesion-separation ratio (lower is better)
            if intra_cluster_distances and inter_cluster_distances:
                metrics["cohesion_separation_ratio"] = metrics["avg_intra_cluster_distance"] / metrics["avg_inter_cluster_distance"]
            
        except Exception as e:
            print(f"Error calculating custom metrics: {e}")
            metrics["custom_metrics_error"] = str(e)
        
        return metrics
    
    def optimize_clustering_parameters(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Suggest optimal clustering parameters for given data
        
        Args:
            embeddings: Sample embeddings for parameter optimization
            
        Returns:
            Dictionary with suggested parameter values
        """
        n_items = len(embeddings)
        
        suggestions = {
            "data_characteristics": {
                "n_items": n_items,
                "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "algorithms_available": self.algorithms_available
            }
        }
        
        if n_items < 5:
            suggestions["recommendation"] = "Too few items for meaningful clustering"
            return suggestions
        
        # Analyze data characteristics
        if len(embeddings) > 1:
            # Calculate pairwise similarities
            similarities = []
            for i in range(min(50, len(embeddings))):  # Sample for performance
                for j in range(i + 1, min(50, len(embeddings))):
                    # Cosine similarity with safety checks
                    norm_i = np.linalg.norm(embeddings[i])
                    norm_j = np.linalg.norm(embeddings[j])
                    
                    if norm_i == 0 or norm_j == 0:
                        sim = 0.0
                    else:
                        sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
                        # Clamp to valid range
                        sim = max(-1.0, min(1.0, sim))
                    
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                
                suggestions["data_characteristics"].update({
                    "avg_similarity": float(avg_similarity),
                    "std_similarity": float(std_similarity),
                    "similarity_distribution": "tight" if std_similarity < 0.1 else "spread"
                })
        
        # Parameter suggestions based on data size and characteristics
        if n_items >= 10 and HDBSCAN_AVAILABLE:
            suggestions["algorithm"] = "HDBSCAN"
            suggestions["parameters"] = {
                "min_cluster_size": max(2, min(5, n_items // 5)),
                "min_samples": max(1, n_items // 20),
                "metric": "euclidean"
            }
        elif n_items >= 5 and SKLEARN_AVAILABLE:
            suggestions["algorithm"] = "DBSCAN"
            suggestions["parameters"] = {
                "eps": 0.3,  # Will be optimized dynamically
                "min_samples": max(1, n_items // 10),
                "metric": "euclidean"
            }
        else:
            suggestions["algorithm"] = "fallback"
            suggestions["parameters"] = {
                "similarity_threshold": 0.7
            }
        
        return suggestions
    
    def _record_clustering_performance(self, embeddings: np.ndarray, labels: List[int], 
                                     metadata: Dict[str, Any], config: ProcessingConfig):
        """Record clustering performance for future optimization"""
        
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "n_items": len(embeddings),
            "n_clusters": len(set(labels)),
            "algorithm": metadata.get("algorithm"),
            "processing_time_ms": metadata.get("processing_time_ms", 0),
            "silhouette_score": metadata.get("silhouette_score", 0),
            "config_used": {
                "min_cluster_size": config.min_cluster_size,
                "min_samples": config.min_samples,
                "clustering_algorithm": config.clustering_algorithm
            }
        }
        
        # Keep history limited for memory
        self.clustering_history.append(performance_record)
        if len(self.clustering_history) > 100:  # TUNABLE: History size limit
            self.clustering_history.pop(0)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get clustering performance statistics"""
        
        if not self.clustering_history:
            return {"no_history": True}
        
        # Calculate statistics from history
        processing_times = [record["processing_time_ms"] for record in self.clustering_history]
        silhouette_scores = [record["silhouette_score"] for record in self.clustering_history if record["silhouette_score"] > 0]
        
        stats = {
            "total_clusterings": len(self.clustering_history),
            "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
            "avg_silhouette_score": np.mean(silhouette_scores) if silhouette_scores else 0,
            "algorithms_used": list(set(record["algorithm"] for record in self.clustering_history)),
            "performance_trend": "improving" if len(silhouette_scores) > 5 and 
                                np.mean(silhouette_scores[-3:]) > np.mean(silhouette_scores[:3]) else "stable"
        }
        
        return stats

    def _safe_to_list(self, obj):
        """Safely convert an object to a list, handling numpy arrays and existing lists"""
        if obj is None:
            return []
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return list(obj)
        else:
            return [obj]  # Single item

# Global instance
_clustering_service: Optional[LocalClusteringService] = None

def get_clustering_service() -> LocalClusteringService:
    """Get global clustering service instance"""
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = LocalClusteringService()
    return _clustering_service 