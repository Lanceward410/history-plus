#!/usr/bin/env python3
"""
Comprehensive Test Data Population Script
Fixes semantic search by populating the enhanced database with realistic test data
"""

import requests
import json
import time
import sys
import os
from datetime import datetime, timedelta

def create_comprehensive_test_data():
    """Create comprehensive test data for semantic search testing"""
    
    # Base timestamp (1 week ago)
    base_timestamp = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    
    # Comprehensive test data covering all categories and search scenarios
    test_entries = [
        # Development & Programming
        {
            'url': 'https://github.com/microsoft/vscode-extension-samples',
            'title': 'VS Code Extension Samples - Microsoft',
            'domain': 'github.com',
            'timestamp': base_timestamp + 1000,
            'category': 'development',
            'subcategory': 'coding',
            'engagement_score': 0.85,
            'time_on_page': 450000,  # 7.5 minutes
            'scroll_depth': 0.9,
            'interaction_count': 25,
            'page_type': 'documentation'
        },
        {
            'url': 'https://stackoverflow.com/questions/12345/how-to-build-chrome-extension',
            'title': 'How to build a Chrome extension? - Stack Overflow',
            'domain': 'stackoverflow.com',
            'timestamp': base_timestamp + 2000,
            'category': 'development',
            'subcategory': 'research',
            'engagement_score': 0.75,
            'time_on_page': 300000,  # 5 minutes
            'scroll_depth': 0.8,
            'interaction_count': 15,
            'page_type': 'question'
        },
        {
            'url': 'https://python.org/doc/tutorial/',
            'title': 'Python Tutorial - Official Documentation',
            'domain': 'python.org',
            'timestamp': base_timestamp + 3000,
            'category': 'development',
            'subcategory': 'learning',
            'engagement_score': 0.9,
            'time_on_page': 600000,  # 10 minutes
            'scroll_depth': 0.95,
            'interaction_count': 40,
            'page_type': 'tutorial'
        },
        
        # Machine Learning & AI
        {
            'url': 'https://course.fast.ai/Lessons/lesson1.html',
            'title': 'Practical Deep Learning for Coders - Lesson 1',
            'domain': 'course.fast.ai',
            'timestamp': base_timestamp + 4000,
            'category': 'learning',
            'subcategory': 'ai_ml',
            'engagement_score': 0.95,
            'time_on_page': 900000,  # 15 minutes
            'scroll_depth': 0.95,
            'interaction_count': 50,
            'page_type': 'course'
        },
        {
            'url': 'https://www.coursera.org/learn/machine-learning',
            'title': 'Machine Learning Course by Andrew Ng',
            'domain': 'coursera.org',
            'timestamp': base_timestamp + 5000,
            'category': 'learning',
            'subcategory': 'ai_ml',
            'engagement_score': 0.88,
            'time_on_page': 720000,  # 12 minutes
            'scroll_depth': 0.85,
            'interaction_count': 35,
            'page_type': 'course'
        },
        
        # Research & Academic
        {
            'url': 'https://scholar.google.com/scholar?q=machine+learning+research',
            'title': 'Machine Learning Research Papers - Google Scholar',
            'domain': 'scholar.google.com',
            'timestamp': base_timestamp + 6000,
            'category': 'research',
            'subcategory': 'academic',
            'engagement_score': 0.7,
            'time_on_page': 240000,  # 4 minutes
            'scroll_depth': 0.6,
            'interaction_count': 20,
            'page_type': 'scholar'
        },
        
        # Work & Productivity
        {
            'url': 'https://docs.google.com/document/d/12345',
            'title': 'Project Planning Document - Google Docs',
            'domain': 'docs.google.com',
            'timestamp': base_timestamp + 7000,
            'category': 'work',
            'subcategory': 'documentation',
            'engagement_score': 0.8,
            'time_on_page': 360000,  # 6 minutes
            'scroll_depth': 0.7,
            'interaction_count': 30,
            'page_type': 'document'
        },
        
        # Entertainment
        {
            'url': 'https://www.youtube.com/watch?v=abc123',
            'title': 'Introduction to Web Extensions - Tutorial',
            'domain': 'youtube.com',
            'timestamp': base_timestamp + 8000,
            'category': 'entertainment',
            'subcategory': 'video',
            'engagement_score': 0.65,
            'time_on_page': 180000,  # 3 minutes
            'scroll_depth': 0.5,
            'interaction_count': 10,
            'page_type': 'video'
        },
        
        # Social Media
        {
            'url': 'https://twitter.com/tech_news/status/12345',
            'title': 'Latest Tech News - Twitter',
            'domain': 'twitter.com',
            'timestamp': base_timestamp + 9000,
            'category': 'social',
            'subcategory': 'news',
            'engagement_score': 0.4,
            'time_on_page': 60000,  # 1 minute
            'scroll_depth': 0.3,
            'interaction_count': 5,
            'page_type': 'social'
        },
        
        # Shopping
        {
            'url': 'https://www.amazon.com/dp/B08N5WRWNW',
            'title': 'Programming Books - Amazon',
            'domain': 'amazon.com',
            'timestamp': base_timestamp + 10000,
            'category': 'shopping',
            'subcategory': 'books',
            'engagement_score': 0.6,
            'time_on_page': 120000,  # 2 minutes
            'scroll_depth': 0.6,
            'interaction_count': 12,
            'page_type': 'product'
        },
        
        # Additional development entries for better search coverage
        {
            'url': 'https://developer.mozilla.org/en-US/docs/Web/Extensions',
            'title': 'Browser Extensions - MDN Web Docs',
            'domain': 'developer.mozilla.org',
            'timestamp': base_timestamp + 11000,
            'category': 'development',
            'subcategory': 'documentation',
            'engagement_score': 0.82,
            'time_on_page': 480000,  # 8 minutes
            'scroll_depth': 0.85,
            'interaction_count': 28,
            'page_type': 'documentation'
        },
        {
            'url': 'https://www.npmjs.com/package/chrome-extension-manifest',
            'title': 'Chrome Extension Manifest - npm',
            'domain': 'npmjs.com',
            'timestamp': base_timestamp + 12000,
            'category': 'development',
            'subcategory': 'tools',
            'engagement_score': 0.68,
            'time_on_page': 180000,  # 3 minutes
            'scroll_depth': 0.6,
            'interaction_count': 12,
            'page_type': 'package'
        }
    ]
    
    return test_entries

def populate_database():
    """Populate the enhanced database with test data"""
    
    print("🚀 Populating enhanced database with test data...")
    
    # Create test data
    test_entries = create_comprehensive_test_data()
    
    # Send to enhanced backend
    try:
        response = requests.post(
            'http://localhost:5000/api/v1/enhanced/database/bulk-insert',
            headers={'Content-Type': 'application/json'},
            json={'entries': test_entries}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Successfully inserted {result['data']['inserted_count']} test entries")
                print(f"⏱️ Processing time: {result['data']['processing_time_ms']}ms")
                return True
            else:
                print(f"❌ Insert failed: {result.get('message')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        return False

def test_semantic_search():
    """Test semantic search with various queries"""
    
    print("\n🔍 Testing semantic search functionality...")
    
    # Test queries
    test_queries = [
        "python development",
        "machine learning tutorial",
        "chrome extension",
        "github repository",
        "stackoverflow question",
        "deep learning course",
        "programming tutorial",
        "web development",
        "AI research",
        "coding help"
    ]
    
    successful_tests = 0
    total_tests = len(test_queries)
    
    for query in test_queries:
        try:
            response = requests.post(
                'http://localhost:5000/api/v1/enhanced/semantic-search',
                headers={'Content-Type': 'application/json'},
                json={
                    'query': query,
                    'options': {'limit': 5, 'min_similarity': 0.1}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    results = result['data']['results']
                    print(f"✅ '{query}': Found {len(results)} results")
                    if results:
                        top_result = results[0]
                        print(f"   Top: {top_result.get('title', 'Unknown')}")
                        print(f"   URL: {top_result.get('url', 'Unknown')}")
                        print(f"   Similarity: {top_result.get('similarity', 0):.2f}")
                        successful_tests += 1
                    else:
                        print(f"   ⚠️ No results found for '{query}'")
                else:
                    print(f"❌ '{query}': {result.get('message')}")
            else:
                print(f"❌ '{query}': HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ '{query}': Error - {e}")
        
        time.sleep(0.1)  # Small delay between requests
    
    print(f"\n📊 Search Test Results: {successful_tests}/{total_tests} successful")
    return successful_tests > 0

def check_backend_health():
    """Check if the enhanced backend is running"""
    
    print("🏥 Checking backend health...")
    
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Backend is healthy: {health.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def main():
    """Main function to populate and test"""
    
    print("🎯 Semantic Search Test & Fix")
    print("=" * 50)
    
    # Step 1: Check backend health
    if not check_backend_health():
        print("\n❌ Enhanced backend is not running!")
        print("   Please start it with:")
        print("   cd backend && python enhanced_app.py")
        return False
    
    # Step 2: Populate database
    if populate_database():
        print("\n✅ Database populated successfully!")
        
        # Step 3: Test semantic search
        search_working = test_semantic_search()
        
        if search_working:
            print("\n🎉 Semantic search is now working!")
            print("\n📋 Next steps:")
            print("   1. Try searching in your dashboard")
            print("   2. Test with different queries")
            print("   3. Check real-time updates")
            print("   4. Test the Chrome extension integration")
        else:
            print("\n⚠️ Semantic search may still have issues")
            print("   Check the backend logs for errors")
        
        return True
        
    else:
        print("\n❌ Failed to populate database")
        print("   Check the backend logs for errors")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 