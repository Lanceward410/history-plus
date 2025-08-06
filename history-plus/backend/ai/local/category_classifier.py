"""
Enhanced Smart Category Classifier with Full AI Pipeline Integration

The category classifier now serves as the main orchestrator for the AI pipeline,
integrating embedding generation, clustering, and LLM services for comprehensive
tab classification with semantic understanding.

Key Features:
- Full AI pipeline integration (embeddings → clustering → classification)
- Hybrid framework/AI approach for consistency and flexibility  
- Comprehensive semantic analysis and confidence scoring
- Performance monitoring and optimization
- Graceful degradation when AI services fail
- Rich metadata and evidence tracking

This classifier provides the intelligence layer for real-time browser tab
categorization with high accuracy and interpretability.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import new AI services
from .embedding_service import get_embedding_service, initialize_embedding_service
from .clustering_service import get_clustering_service
from .local_llm_service import get_llm_service
from ..interfaces.ai_processor import ProcessingConfig, PerformanceMonitor, ProcessingResult
from ..category_framework import CategoryFramework
from ..utils.text_preprocessing import get_text_preprocessor
from ..utils.cache_manager import get_memory_monitor

class SmartCategoryClassifier:
    """
    Enhanced category classifier with full AI pipeline integration
    
    Orchestrates the complete AI workflow from raw tab data to final
    category assignments with confidence scores and rich metadata.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the smart category classifier
        
        Args:
            config: Optional processing configuration. Creates default if not provided.
        """
        # Use provided config or create default
        self.config = config or ProcessingConfig()
        
        # Initialize AI services
        self.embedding_service = get_embedding_service()
        self.clustering_service = get_clustering_service()
        self.llm_service = get_llm_service()
        
        # Category framework and utilities
        self.framework = CategoryFramework()
        self.text_preprocessor = get_text_preprocessor(aggressive=False)
        self.memory_monitor = get_memory_monitor()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize embedding service if not already done
        if not self.embedding_service.model:
            success = initialize_embedding_service(self.config)
            if not success:
                print("Warning: Embedding service initialization failed. Using fallback classification.")
        
        # Classification history for optimization
        self.classification_history = []
        
        # TUNABLE: Enhanced classification parameters  
        self.enhanced_params = {
            # Confidence combination weights
            "semantic_weight": 0.4,        # TUNE: Weight of semantic similarity in final confidence
            "framework_weight": 0.3,       # TUNE: Weight of framework matching
            "clustering_weight": 0.2,      # TUNE: Weight of clustering quality
            "llm_weight": 0.1,            # TUNE: Weight of LLM confidence
            
            # Quality thresholds
            "min_semantic_coherence": 0.3, # TUNE: Minimum cluster coherence for high confidence
            "min_framework_confidence": 0.7, # TUNE: Threshold for preferring framework over AI
            "max_processing_time_ms": 5000,  # TUNE: Maximum processing time before timeout
            
            # Fallback behavior
            "enable_semantic_fallback": True,  # TUNE: Use semantic analysis when framework fails
            "enable_clustering_validation": True, # TUNE: Validate results with clustering quality
        }
    
    def classify_tab_cluster(self, cluster_tabs: List[Dict]) -> ProcessingResult:
        """
        Classify a cluster of tabs using the full AI pipeline
        
        Args:
            cluster_tabs: List of tab dictionaries with title, domain, url, etc.
            
        Returns:
            ProcessingResult with category, subcategory, confidence, and metadata
        """
        if not cluster_tabs:
            return self._create_empty_result()
        
        # Start performance monitoring
        operation_start = self.performance_monitor.start_operation("classify_tab_cluster")
        result = ProcessingResult(success=False)
        
        try:
            # Stage 1: Data preparation and validation
            preparation_start = time.time()
            processed_data = self._prepare_tab_data(cluster_tabs)
            if not processed_data["valid"]:
                result.add_error("Invalid tab data provided")
                return result
            
            preparation_time = (time.time() - preparation_start) * 1000
            result.add_timing("data_preparation", preparation_time)
            
            # Stage 2: Semantic embedding generation
            embedding_start = time.time()
            embeddings_result = self._generate_semantic_embeddings(processed_data)
            embedding_time = (time.time() - embedding_start) * 1000
            result.add_timing("embedding_generation", embedding_time)
            
            # Stage 3: Clustering analysis (if multiple items)
            clustering_start = time.time()
            clustering_result = self._perform_clustering_analysis(
                embeddings_result, processed_data, len(cluster_tabs)
            )
            clustering_time = (time.time() - clustering_start) * 1000
            result.add_timing("clustering_analysis", clustering_time)
            
            # Stage 4: Framework matching
            framework_start = time.time()
            framework_result = self._enhanced_framework_matching(
                processed_data, embeddings_result, clustering_result
            )
            framework_time = (time.time() - framework_start) * 1000
            result.add_timing("framework_matching", framework_time)
            
            # Stage 5: AI classification (if framework confidence is low)
            ai_start = time.time()
            ai_result = self._ai_classification(
                processed_data, embeddings_result, clustering_result, framework_result
            )
            ai_time = (time.time() - ai_start) * 1000
            result.add_timing("ai_classification", ai_time)
            
            # Stage 6: Final decision and confidence calculation
            decision_start = time.time()
            final_result = self._make_final_decision(
                framework_result, ai_result, clustering_result, embeddings_result
            )
            decision_time = (time.time() - decision_start) * 1000
            result.add_timing("final_decision", decision_time)
            
            # Populate result
            result.success = True
            result.category = final_result["category"]
            result.subcategory = final_result["subcategory"]
            result.confidence = final_result["confidence"]
            result.method = final_result["method"]
            result.cluster_coherence = clustering_result.get("coherence", 0.0)
            result.semantic_similarity = embeddings_result.get("avg_similarity", 0.0)
            result.embeddings_shape = embeddings_result.get("shape")
            
            # Record performance
            total_time = self.performance_monitor.end_operation("classify_tab_cluster", operation_start)
            result.processing_time_ms = total_time
            
            # Store classification for learning
            self._record_classification(result, cluster_tabs, final_result)
            
            if self.config.enable_debug_logging:
                print(f"Classified {len(cluster_tabs)} tabs as '{result.category}/{result.subcategory}' "
                      f"with {result.confidence:.2f} confidence in {total_time:.1f}ms")
            
            return result
            
        except Exception as e:
            result.add_error(f"Classification failed: {str(e)}")
            print(f"Classification error: {e}")
            
            # Return fallback result
            return self._create_fallback_result(cluster_tabs)
        
        finally:
            # Cleanup and memory management
            if self.config.enable_garbage_collection:
                import gc
                gc.collect()
    
    def _prepare_tab_data(self, cluster_tabs: List[Dict]) -> Dict[str, Any]:
        """
        Prepare and validate tab data for processing
        
        Returns:
            Dictionary with cleaned data and validation status
        """
        try:
            # Extract basic information
            titles = [tab.get('title', '') for tab in cluster_tabs]
            urls = [tab.get('url', '') for tab in cluster_tabs]
            domains = [tab.get('domain', '') for tab in cluster_tabs]
            
            # Clean and preprocess text data
            cleaned_titles = [self.text_preprocessor.clean_text(title, 'title') for title in titles]
            cleaned_domains = [self.text_preprocessor.clean_text(domain, 'domain') for domain in domains]
            
            # Validate that we have meaningful data
            valid_titles = [t for t in cleaned_titles if t and len(t) > 2]
            valid_domains = [d for d in cleaned_domains if d and len(d) > 2]
            
            if not valid_titles and not valid_domains:
                return {"valid": False, "reason": "No meaningful text data found"}
            
            # Create combined text for analysis
            all_text = ' '.join(cleaned_titles + cleaned_domains)
            
            return {
                "valid": True,
                "original_titles": titles,
                "original_domains": domains,
                "original_urls": urls,
                "cleaned_titles": cleaned_titles,
                "cleaned_domains": cleaned_domains,
                "combined_text": all_text,
                "tab_count": len(cluster_tabs),
                "unique_domains": len(set(domains)),
                "text_length": len(all_text)
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"Data preparation failed: {str(e)}"}
    
    def _generate_semantic_embeddings(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate semantic embeddings for the tab data
        
        Returns:
            Dictionary with embeddings and related metadata
        """
        try:
            # Prepare texts for embedding
            texts_to_embed = []
            
            # Add cleaned titles (primary signal)
            for title in processed_data["cleaned_titles"]:
                if title and len(title) > 2:
                    texts_to_embed.append(title)
            
            # Add domain information (secondary signal)
            for domain in processed_data["cleaned_domains"]:
                if domain and len(domain) > 2:
                    texts_to_embed.append(domain)
            
            if not texts_to_embed:
                return {
                    "success": False,
                    "embeddings": None,
                    "reason": "No valid text for embedding"
                }
            
            # Generate embeddings
            embeddings = self.embedding_service.generate_embeddings(texts_to_embed)
            
            if embeddings.size == 0:
                return {
                    "success": False,
                    "embeddings": None,
                    "reason": "Embedding generation failed"
                }
            
            # Calculate average similarity within the cluster
            avg_similarity = self._calculate_average_similarity(embeddings)
            
            return {
                "success": True,
                "embeddings": embeddings,
                "texts": texts_to_embed,
                "shape": embeddings.shape,
                "avg_similarity": avg_similarity,
                "coherence": avg_similarity  # Alias for consistency
            }
            
        except Exception as e:
            return {
                "success": False,
                "embeddings": None,
                "reason": f"Embedding generation error: {str(e)}"
            }
    
    def _calculate_average_similarity(self, embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within embeddings"""
        if len(embeddings) < 2:
            return 1.0
        
        try:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
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
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _perform_clustering_analysis(self, embeddings_result: Dict[str, Any], 
                                   processed_data: Dict[str, Any], 
                                   cluster_size: int) -> Dict[str, Any]:
        """
        Perform clustering analysis to validate cluster quality
        
        Args:
            embeddings_result: Result from embedding generation
            processed_data: Processed tab data
            cluster_size: Size of the original cluster
            
        Returns:
            Dictionary with clustering analysis results
        """
        try:
            # Skip clustering for single items
            if cluster_size <= 1:
                return {
                    "success": True,
                    "coherence": 1.0,
                    "quality": "single_item",
                    "validation": "not_applicable"
                }
            
            # Skip clustering if embeddings failed
            if not embeddings_result.get("success") or embeddings_result.get("embeddings") is None:
                return {
                    "success": False,
                    "reason": "No embeddings available for clustering"
                }
            
            embeddings = embeddings_result["embeddings"]
            
            # Perform clustering to assess quality
            labels, metadata = self.clustering_service.cluster_embeddings(embeddings, self.config)
            
            # Calculate quality metrics
            quality_metrics = self.clustering_service.get_cluster_quality_metrics(embeddings, labels)
            
            # Determine if the original cluster is cohesive
            coherence = embeddings_result.get("avg_similarity", 0.0)
            
            # TUNABLE: Quality assessment thresholds
            if coherence >= self.enhanced_params["min_semantic_coherence"]:
                quality_assessment = "high"
            elif coherence >= 0.2:  # TUNABLE: Medium coherence threshold
                quality_assessment = "medium"
            else:
                quality_assessment = "low"
            
            return {
                "success": True,
                "labels": labels,
                "metadata": metadata,
                "quality_metrics": quality_metrics,
                "coherence": coherence,
                "quality": quality_assessment,
                "n_clusters_found": len(set(labels)),
                "validation": "cohesive" if coherence >= self.enhanced_params["min_semantic_coherence"] else "dispersed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "reason": f"Clustering analysis failed: {str(e)}",
                "coherence": 0.0,
                "quality": "unknown"
            }
    
    def _enhanced_framework_matching(self, processed_data: Dict[str, Any], 
                                   embeddings_result: Dict[str, Any],
                                   clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced framework matching with semantic similarity
        
        Combines traditional framework matching with semantic analysis
        """
        try:
            # Get basic framework match
            base_match = self._match_existing_category(
                processed_data["cleaned_titles"],
                processed_data["cleaned_domains"],
                []  # Empty history for now
            )
            
            # Enhance with semantic similarity if embeddings available
            if embeddings_result.get("success") and embeddings_result.get("embeddings") is not None:
                semantic_enhancement = self._calculate_semantic_framework_match(
                    embeddings_result, processed_data
                )
                
                # Combine framework and semantic scores
                enhanced_confidence = self._combine_framework_scores(base_match, semantic_enhancement)
                
                return {
                    "success": True,
                    "category": base_match.get("category", "Unknown"),
                    "confidence": enhanced_confidence,
                    "method": "framework_enhanced",
                    "base_match": base_match,
                    "semantic_enhancement": semantic_enhancement,
                    "cluster_quality": clustering_result.get("quality", "unknown")
                }
            else:
                # Fall back to basic framework matching
                return {
                    "success": True,
                    "category": base_match.get("category", "Unknown"),
                    "confidence": base_match.get("confidence", 0.0),
                    "method": "framework_basic",
                    "base_match": base_match,
                    "cluster_quality": clustering_result.get("quality", "unknown")
                }
                
        except Exception as e:
            return {
                "success": False,
                "reason": f"Framework matching failed: {str(e)}",
                "confidence": 0.0
            }
    
    def _calculate_semantic_framework_match(self, embeddings_result: Dict[str, Any], 
                                          processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic similarity with framework categories"""
        
        semantic_scores = {}
        embeddings = embeddings_result["embeddings"]
        
        try:
            # Calculate average embedding for the cluster
            cluster_embedding = np.mean(embeddings, axis=0)
            
            # Get representative texts for each framework category
            for category_name, category_info in self.framework.framework.items():
                # Create representative text for the category
                category_text = f"{category_info.get('description', '')} {' '.join(category_info.get('keywords', []))}"
                
                if category_text.strip():
                    # Generate embedding for category
                    category_embeddings = self.embedding_service.generate_embeddings([category_text])
                    
                    if len(category_embeddings) > 0:
                        # Calculate cosine similarity with safety checks
                        norm_cluster = np.linalg.norm(cluster_embedding)
                        norm_category = np.linalg.norm(category_embeddings[0])
                        
                        if norm_cluster == 0 or norm_category == 0:
                            similarity = 0.0
                        else:
                            similarity = np.dot(cluster_embedding, category_embeddings[0]) / (norm_cluster * norm_category)
                            # Clamp to valid range
                            similarity = max(-1.0, min(1.0, similarity))
                        
                        semantic_scores[category_name] = float(similarity)
            
            # Find best semantic match
            if semantic_scores:
                best_semantic = max(semantic_scores.items(), key=lambda x: x[1])
                return {
                    "best_category": best_semantic[0],
                    "best_score": best_semantic[1],
                    "all_scores": semantic_scores
                }
            else:
                return {"best_category": None, "best_score": 0.0, "all_scores": {}}
                
        except Exception as e:
            print(f"Semantic framework matching failed: {e}")
            return {"best_category": None, "best_score": 0.0, "all_scores": {}}
    
    def _combine_framework_scores(self, base_match: Dict[str, Any], 
                                semantic_enhancement: Dict[str, Any]) -> float:
        """Combine framework and semantic scores for final confidence"""
        
        base_confidence = base_match.get("confidence", 0.0)
        semantic_score = semantic_enhancement.get("best_score", 0.0)
        
        # TUNABLE: Scoring combination formula
        if semantic_score > 0.3:  # TUNABLE: Threshold for considering semantic score
            # Boost confidence if semantic analysis agrees
            if semantic_enhancement.get("best_category") == base_match.get("category"):
                combined_confidence = base_confidence * 0.7 + semantic_score * 0.3
            else:
                # Reduce confidence if semantic analysis disagrees
                combined_confidence = base_confidence * 0.8
        else:
            # No semantic boost
            combined_confidence = base_confidence
        
        return min(0.95, combined_confidence)  # Cap at 95%
    
    def _ai_classification(self, processed_data: Dict[str, Any], 
                          embeddings_result: Dict[str, Any],
                          clustering_result: Dict[str, Any],
                          framework_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-based classification using LLM service
        
        Used when framework confidence is low or as a validation step
        """
        try:
            # Prepare context for AI classification
            context = {
                "cluster_quality": clustering_result.get("quality", "unknown"),
                "semantic_coherence": clustering_result.get("coherence", 0.0),
                "unique_domains": processed_data.get("unique_domains", 0),
                "tab_count": processed_data.get("tab_count", 0)
            }
            
            # Use LLM service for classification
            ai_result = self.llm_service.generate_category_name(
                titles=processed_data["cleaned_titles"],
                domains=processed_data["cleaned_domains"],
                context=context
            )
            
            return {
                "success": True,
                "category": ai_result.get("category", "Unknown"),
                "subcategory": ai_result.get("subcategory", "General Activity"),
                "confidence": ai_result.get("confidence", 0.5),
                "method": ai_result.get("method", "ai_generated"),
                "evidence": ai_result.get("evidence", []),
                "analysis_metadata": ai_result.get("analysis_metadata", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "reason": f"AI classification failed: {str(e)}",
                "confidence": 0.0
            }
    
    def _make_final_decision(self, framework_result: Dict[str, Any],
                           ai_result: Dict[str, Any],
                           clustering_result: Dict[str, Any],
                           embeddings_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final classification decision by combining all analysis results
        
        Uses confidence scores and quality metrics to choose the best result
        """
        framework_confidence = framework_result.get("confidence", 0.0)
        ai_confidence = ai_result.get("confidence", 0.0) if ai_result.get("success") else 0.0
        cluster_quality = clustering_result.get("coherence", 0.0)
        
        # TUNABLE: Decision logic
        if framework_confidence >= self.enhanced_params["min_framework_confidence"]:
            # High confidence framework match
            chosen_result = {
                "category": framework_result["category"],
                "subcategory": self._generate_enhanced_subcategory(
                    framework_result["category"], ai_result, clustering_result
                ),
                "confidence": framework_confidence,
                "method": "framework_preferred",
                "primary_source": "framework",
                "validation": ai_result if ai_result.get("success") else None
            }
            
        elif ai_confidence > framework_confidence and ai_result.get("success"):
            # AI classification is more confident
            chosen_result = {
                "category": ai_result["category"],
                "subcategory": ai_result["subcategory"],
                "confidence": ai_confidence,
                "method": "ai_preferred",
                "primary_source": "ai",
                "validation": framework_result
            }
            
        elif framework_confidence > 0:
            # Use framework with moderate confidence
            chosen_result = {
                "category": framework_result["category"],
                "subcategory": self._generate_enhanced_subcategory(
                    framework_result["category"], ai_result, clustering_result
                ),
                "confidence": framework_confidence,
                "method": "framework_moderate",
                "primary_source": "framework",
                "validation": ai_result if ai_result.get("success") else None
            }
            
        else:
            # Last resort: use AI or fallback
            if ai_result.get("success"):
                chosen_result = {
                    "category": ai_result["category"],
                    "subcategory": ai_result["subcategory"],
                    "confidence": ai_confidence,
                    "method": "ai_fallback",
                    "primary_source": "ai"
                }
            else:
                chosen_result = {
                    "category": "Web Browsing",
                    "subcategory": "General Activity",
                    "confidence": 0.3,
                    "method": "final_fallback",
                    "primary_source": "fallback"
                }
        
        # Apply cluster quality adjustment
        if cluster_quality < 0.2:  # TUNABLE: Low quality threshold
            chosen_result["confidence"] *= 0.8  # Reduce confidence for low-quality clusters
            if "warnings" not in chosen_result:
                chosen_result["warnings"] = []
            chosen_result["warnings"].append("Low cluster coherence detected")
        
        return chosen_result
    
    def _generate_enhanced_subcategory(self, category: str, ai_result: Dict[str, Any], 
                                     clustering_result: Dict[str, Any]) -> str:
        """Generate enhanced subcategory using multiple sources"""
        
        # Try AI subcategory first
        if ai_result.get("success") and ai_result.get("subcategory"):
            return ai_result["subcategory"]
        
        # Fall back to cluster-based subcategory
        cluster_quality = clustering_result.get("quality", "unknown")
        
        if cluster_quality == "high":
            return "Deep Focus"
        elif cluster_quality == "medium":
            return "Multi-source Research"
        elif cluster_quality == "low":
            return "Exploratory Browsing"
        else:
            return "General Activity"

    # Keep existing methods with TUNABLE parameter comments
    def _match_existing_category(self, titles: List[str], domains: List[str], 
                               history: List[str]) -> Dict[str, Any]:
        """Enhanced category matching with tunable parameters"""
        
        # TUNABLE: These parameters significantly affect matching accuracy
        DOMAIN_WEIGHT = self.config.domain_weight      # TUNE: Weight of domain matching (0.0-1.0)
        KEYWORD_WEIGHT = self.config.keyword_weight    # TUNE: Weight of keyword matching (0.0-1.0) 
        SEMANTIC_WEIGHT = self.config.semantic_weight  # TUNE: Weight of semantic similarity (0.0-1.0)
        
        best_category = None
        best_confidence = 0.0
        
        # Calculate scores for each framework category
        for category_name, category_info in self.framework.framework.items():
            domain_score = self._calculate_domain_match(domains, category_info.get('domains', []))
            keyword_score = self._calculate_keyword_match(titles, category_info.get('keywords', []))
            
            # Combined score using tunable weights
            total_score = (domain_score * DOMAIN_WEIGHT + 
                          keyword_score * KEYWORD_WEIGHT)
            
            if total_score > best_confidence:
                best_confidence = total_score
                best_category = category_name
        
        return {
            "category": best_category or "Web Browsing",
            "confidence": min(0.9, best_confidence),  # Cap confidence
            "method": "framework_matching"
        }
    
    def _calculate_domain_match(self, domains: List[str], category_domains: List[str]) -> float:
        """Calculate domain matching score with tunable sensitivity"""
        
        if not domains or not category_domains:
            return 0.0
        
        matches = 0
        total_domains = len(domains)
        
        # TUNABLE: Domain matching sensitivity
        PARTIAL_MATCH_THRESHOLD = 0.7  # TUNE: Threshold for partial domain matching
        
        for domain in domains:
            for cat_domain in category_domains:
                if cat_domain in domain or domain in cat_domain:
                    matches += 1
                    break
        
        return matches / total_domains
    
    def _calculate_keyword_match(self, titles: List[str], category_keywords: List[str]) -> float:
        """Calculate keyword matching score with tunable parameters"""
        
        if not titles or not category_keywords:
            return 0.0
        
        # TUNABLE: Keyword matching parameters
        KEYWORD_DENSITY_WEIGHT = 0.7   # TUNE: Weight of keyword density vs variety
        KEYWORD_VARIETY_WEIGHT = 0.3   # TUNE: Weight of keyword variety vs density
        
        combined_text = ' '.join(titles).lower()
        total_words = len(combined_text.split())
        
        if total_words == 0:
            return 0.0
        
        # Count keyword matches
        keyword_matches = 0
        unique_keywords_matched = set()
        
        for keyword in category_keywords:
            import re
            matches = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text))
            if matches > 0:
                keyword_matches += matches
                unique_keywords_matched.add(keyword)
        
        # Calculate density and variety scores
        density_score = min(1.0, keyword_matches / total_words * 10)  # TUNABLE: Density scaling
        variety_score = len(unique_keywords_matched) / len(category_keywords)
        
        return (density_score * KEYWORD_DENSITY_WEIGHT + 
                variety_score * KEYWORD_VARIETY_WEIGHT)

    def _create_empty_result(self) -> ProcessingResult:
        """Create empty result for edge cases"""
        result = ProcessingResult(success=True)
        result.category = "Unknown"
        result.subcategory = "No Data"
        result.confidence = 0.0
        result.method = "empty_input"
        return result
    
    def _create_fallback_result(self, cluster_tabs: List[Dict]) -> ProcessingResult:
        """Create fallback result when all processing fails"""
        result = ProcessingResult(success=True)
        result.category = "Web Browsing"
        result.subcategory = "General Activity"
        result.confidence = 0.3
        result.method = "fallback"
        result.add_warning("Classification fell back to default category")
        return result
    
    def _record_classification(self, result: ProcessingResult, cluster_tabs: List[Dict], 
                             final_result: Dict[str, Any]):
        """Record classification for learning and optimization"""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "tab_count": len(cluster_tabs),
            "category": result.category,
            "subcategory": result.subcategory,
            "confidence": result.confidence,
            "method": result.method,
            "processing_time_ms": result.processing_time_ms,
            "cluster_coherence": result.cluster_coherence,
            "config_snapshot": {
                "domain_weight": self.config.domain_weight,
                "keyword_weight": self.config.keyword_weight,
                "semantic_weight": self.config.semantic_weight,
                "confidence_threshold": self.config.confidence_threshold
            }
        }
        
        self.classification_history.append(record)
        
        # Keep history manageable
        if len(self.classification_history) > 1000:  # TUNABLE: History size limit
            self.classification_history.pop(0)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        if not self.classification_history:
            return {"no_history": True}
        
        # Calculate statistics
        confidences = [r["confidence"] for r in self.classification_history]
        processing_times = [r["processing_time_ms"] for r in self.classification_history]
        coherences = [r["cluster_coherence"] for r in self.classification_history if r["cluster_coherence"] > 0]
        
        stats = {
            "total_classifications": len(self.classification_history),
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
            "avg_cluster_coherence": np.mean(coherences) if coherences else 0,
            "methods_used": list(set(r["method"] for r in self.classification_history)),
            "categories_assigned": list(set(r["category"] for r in self.classification_history)),
            "performance_trend": self._calculate_performance_trend()
        }
        
        return stats
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend from recent history"""
        
        if len(self.classification_history) < 10:
            return "insufficient_data"
        
        recent = self.classification_history[-5:]
        older = self.classification_history[-10:-5]
        
        recent_avg_confidence = np.mean([r["confidence"] for r in recent])
        older_avg_confidence = np.mean([r["confidence"] for r in older])
        
        if recent_avg_confidence > older_avg_confidence + 0.05:  # TUNABLE: Trend threshold
            return "improving"
        elif recent_avg_confidence < older_avg_confidence - 0.05:
            return "declining"
        else:
            return "stable"

# Global instance for backward compatibility
_classifier: Optional[SmartCategoryClassifier] = None

def get_category_classifier(config: Optional[ProcessingConfig] = None) -> SmartCategoryClassifier:
    """Get global category classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = SmartCategoryClassifier(config)
    return _classifier 