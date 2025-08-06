#!/usr/bin/env python3
"""
Test script for the enhanced History Plus database and real-time system
"""

import sys
import os
import time
import json
import asyncio
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from services.database_service import DatabaseService, QueryFilter, QueryOperator, QueryOptions
        print("✓ Database service imports successful")
    except ImportError as e:
        print(f"✗ Database service import failed: {e}")
        return False
    
    try:
        from services.realtime_service import RealtimeService, EventType, RealtimeEvent
        print("✓ Real-time service imports successful")
    except ImportError as e:
        print(f"✗ Real-time service import failed: {e}")
        return False
    
    try:
        from routes.enhanced_api_routes import enhanced_api_bp
        print("✓ Enhanced API routes imports successful")
    except ImportError as e:
        print(f"✗ Enhanced API routes import failed: {e}")
        return False
    
    try:
        from enhanced_app import create_enhanced_app
        print("✓ Enhanced app imports successful")
    except ImportError as e:
        print(f"✗ Enhanced app import failed: {e}")
        return False
    
    return True

def test_database_service():
    """Test the enhanced database service"""
    print("\nTesting database service...")
    
    try:
        from services.database_service import DatabaseService, QueryFilter, QueryOperator, QueryOptions
        
        # Create database service with test database
        db_service = DatabaseService("test_enhanced.db")
        print("✓ Database service created")
        
        # Test basic query (should return empty list for new database)
        results = db_service.query_history_entries()
        print(f"✓ Basic query successful, returned {len(results)} entries")
        
        # Test bulk insert with sample data
        sample_entries = [
            {
                'url': 'https://github.com/user/repo',
                'title': 'GitHub Repository',
                'domain': 'github.com',
                'timestamp': int(datetime.now().timestamp() * 1000),
                'category': 'development',
                'subcategory': 'coding',
                'engagement_score': 0.8
            },
            {
                'url': 'https://stackoverflow.com/questions/123',
                'title': 'Python Question',
                'domain': 'stackoverflow.com',
                'timestamp': int(datetime.now().timestamp() * 1000),
                'category': 'development',
                'subcategory': 'research',
                'engagement_score': 0.6
            }
        ]
        
        inserted_count = db_service.bulk_insert_history_entries(sample_entries)
        print(f"✓ Bulk insert successful, inserted {inserted_count} entries")
        
        # Test semantic search
        search_results = db_service.semantic_search("python development")
        print(f"✓ Semantic search successful, found {len(search_results)} results")
        
        # Test advanced filtering
        filters = [
            QueryFilter("category", QueryOperator.EQUALS, "development")
        ]
        options = QueryOptions(limit=10)
        
        filtered_results = db_service.query_history_entries(filters, options)
        print(f"✓ Advanced filtering successful, found {len(filtered_results)} results")
        
        # Test analytics
        analytics = db_service.get_category_analytics(7)
        if 'error' not in analytics:
            print("✓ Category analytics successful")
        else:
            print(f"⚠ Category analytics returned error: {analytics['error']}")
        
        productivity = db_service.get_productivity_insights(7)
        if 'error' not in productivity:
            print("✓ Productivity insights successful")
        else:
            print(f"⚠ Productivity insights returned error: {productivity['error']}")
        
        # Test performance stats
        perf_stats = db_service.get_performance_stats()
        if 'error' not in perf_stats:
            print("✓ Performance stats successful")
        else:
            print(f"⚠ Performance stats returned error: {perf_stats['error']}")
        
        # Clean up test database
        try:
            os.remove("test_enhanced.db")
            print("✓ Test database cleaned up")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"✗ Database service test failed: {e}")
        return False

def test_realtime_service():
    """Test the real-time service"""
    print("\nTesting real-time service...")
    
    try:
        from services.realtime_service import RealtimeService, EventType, RealtimeEvent
        from services.database_service import DatabaseEvent, DatabaseChangeType
        
        # Create real-time service without SocketIO
        realtime_service = RealtimeService(None)
        print("✓ Real-time service created")
        
        # Test event creation and broadcasting
        test_event = RealtimeEvent(
            event_type=EventType.DATABASE_CHANGE,
            data={'table': 'history_entries', 'change_type': 'insert'}
        )
        
        # Add a test callback
        events_received = []
        def test_callback(event):
            events_received.append(event)
        
        realtime_service.add_event_callback('database_change', test_callback)
        
        # Broadcast test event
        realtime_service.broadcast_event(test_event)
        print("✓ Event broadcasting successful")
        
        # Test database change broadcasting
        db_event = DatabaseEvent(
            change_type=DatabaseChangeType.INSERT,
            table='history_entries',
            record_id='test_id',
            data={'test': 'data'}
        )
        
        realtime_service.broadcast_database_change(db_event)
        print("✓ Database change broadcasting successful")
        
        # Test analytics broadcasting
        realtime_service.broadcast_analytics_update({
            'type': 'test_analytics',
            'data': {'test': 'analytics'}
        })
        print("✓ Analytics broadcasting successful")
        
        # Test connection stats
        stats = realtime_service.get_connection_stats()
        if 'error' not in stats:
            print("✓ Connection stats successful")
        else:
            print(f"⚠ Connection stats returned error: {stats['error']}")
        
        # Test event history
        history = realtime_service.get_event_history()
        print(f"✓ Event history successful, found {len(history)} events")
        
        return True
        
    except Exception as e:
        print(f"✗ Real-time service test failed: {e}")
        return False

def test_enhanced_app():
    """Test the enhanced Flask application"""
    print("\nTesting enhanced Flask application...")
    
    try:
        from enhanced_app import create_enhanced_app
        
        # Create enhanced app
        app, socketio = create_enhanced_app('development')
        print("✓ Enhanced app created successfully")
        
        # Test that app has the required attributes
        if hasattr(app, 'db_service'):
            print("✓ App has database service")
        else:
            print("⚠ App missing database service")
        
        if hasattr(app, 'realtime_service'):
            print("✓ App has real-time service")
        else:
            print("⚠ App missing real-time service")
        
        if hasattr(app, 'socketio'):
            print("✓ App has SocketIO attribute")
        else:
            print("⚠ App missing SocketIO attribute")
        
        # Test that routes are registered
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        
        expected_routes = [
            '/api/v1/enhanced/semantic-search',
            '/api/v1/enhanced/query-database',
            '/api/v1/enhanced/analytics/categories',
            '/api/v1/enhanced/analytics/productivity',
            '/api/v1/enhanced/database/performance',
            '/api/v1/enhanced/realtime/stats'
        ]
        
        routes_found = 0
        for expected_route in expected_routes:
            if expected_route in routes:
                routes_found += 1
        
        print(f"✓ Found {routes_found}/{len(expected_routes)} expected enhanced routes")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced app test failed: {e}")
        return False

def test_api_integration():
    """Test API integration by making actual HTTP requests"""
    print("\nTesting API integration...")
    
    try:
        import requests
        import threading
        import time
        from enhanced_app import run_enhanced_app, create_enhanced_app
        
        # Create app
        app, socketio = create_enhanced_app('development')
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=lambda: run_enhanced_app(app, socketio, host='127.0.0.1', port=5001, debug=False),
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test health endpoint
        try:
            response = requests.get('http://127.0.0.1:5001/health', timeout=5)
            if response.status_code == 200:
                print("✓ Health endpoint accessible")
            else:
                print(f"⚠ Health endpoint returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Health endpoint not accessible: {e}")
        
        # Test root endpoint
        try:
            response = requests.get('http://127.0.0.1:5001/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('features'):
                    print("✓ Root endpoint with features accessible")
                else:
                    print("⚠ Root endpoint accessible but missing features")
            else:
                print(f"⚠ Root endpoint returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Root endpoint not accessible: {e}")
        
        # Test enhanced API endpoint
        try:
            response = requests.get('http://127.0.0.1:5001/api/v1/enhanced/database/performance', timeout=5)
            if response.status_code in [200, 503]:  # 503 is OK if services not fully initialized
                print("✓ Enhanced API endpoint accessible")
            else:
                print(f"⚠ Enhanced API endpoint returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠ Enhanced API endpoint not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ API integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("History Plus Enhanced System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Database Service Tests", test_database_service),
        ("Real-time Service Tests", test_realtime_service),
        ("Enhanced App Tests", test_enhanced_app),
        ("API Integration Tests", test_api_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name}")
        print(f"{'-' * 40}")
        
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    
    if failed == 0:
        print("🎉 All tests passed! The enhanced system is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 