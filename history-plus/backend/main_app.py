"""
Main Flask application with real-time database services and LLM integration
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
import sys
from datetime import datetime

# Try to import Flask-SocketIO
try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("Warning: Flask-SocketIO not available. Install with: pip install flask-socketio")

from config import config
from routes.api_routes import api_bp
from routes.current_tasks import current_tasks_bp
from routes.enhanced_api_routes import enhanced_api_bp, init_services
from services.database_service import DatabaseService
from services.realtime_service import RealtimeService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    """Main application factory with real-time services and LLM integration"""
    
    # Determine config
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    logger.info(f"Creating main app with config: {config_name}")
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Add configuration
    app.config['MAX_BULK_INSERT_ENTRIES'] = int(os.environ.get('MAX_BULK_INSERT_ENTRIES', '1000'))
    app.config['ENABLE_REALTIME'] = os.environ.get('ENABLE_REALTIME', 'True').lower() == 'true'
    
    # Configure logging
    if not app.debug:
        logging.basicConfig(level=logging.INFO)
    
    # Setup CORS for Chrome extension and dashboard
    cors_origins = app.config['CORS_ORIGINS'].copy()
    cors_origins.extend([
        'chrome-extension://*',
        'moz-extension://*'  # Firefox support
    ])
    
    CORS(app, origins=cors_origins, supports_credentials=True)
    
    # Initialize SocketIO for real-time features (if available)
    socketio = None
    if SOCKETIO_AVAILABLE and app.config['ENABLE_REALTIME']:
        try:
            socketio = SocketIO(
                app, 
                cors_allowed_origins="*", 
                async_mode='threading',
                logger=False,  # Disable SocketIO logging to reduce noise
                engineio_logger=False
            )
            logger.info("SocketIO initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SocketIO: {e}")
            socketio = None
    else:
        logger.warning("SocketIO not available or disabled")
    
    # Initialize services
    try:
        # Initialize database service
        db_path = os.environ.get('DATABASE_PATH', 'history_plus_enhanced.db')
        db_service = DatabaseService(db_path)
        logger.info(f"Database service initialized with path: {db_path}")
        
        # Initialize real-time service
        realtime_service = RealtimeService(socketio)
        logger.info("Real-time service initialized")
        
        # Connect database events to real-time service
        db_service.add_event_listener('*', realtime_service.broadcast_database_change)
        logger.info("Database and real-time services connected")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Create fallback services
        db_service = None
        realtime_service = None
    
    # Initialize API routes with services
    if db_service and realtime_service:
        init_services(db_service, realtime_service)
        logger.info("API routes initialized")
    else:
        logger.warning("API routes initialized without services")
    
    # Register blueprints
    try:
        app.register_blueprint(api_bp, url_prefix=app.config['API_PREFIX'])
        app.register_blueprint(current_tasks_bp)
        app.register_blueprint(enhanced_api_bp, url_prefix=f"{app.config['API_PREFIX']}/enhanced")
        logger.info("All blueprints registered successfully")
    except Exception as e:
        logger.error(f"Error registering blueprints: {e}")
        raise
    
    # Root endpoint
    @app.route('/')
    def index():
        try:
            available_features = {
                'real_time': socketio is not None,
                'semantic_search': db_service is not None,
                'advanced_querying': db_service is not None,
                'analytics': db_service is not None,
                'database_service': db_service is not None,
                'websocket_support': SOCKETIO_AVAILABLE
            }
            
            endpoints = {
                'health': f"{app.config['API_PREFIX']}/health",
                'process_history': f"{app.config['API_PREFIX']}/process-history",
                'current_tasks': '/api/analyze-current-tasks'
            }
            
            # Add endpoints if available
            if db_service:
                advanced_endpoints = {
                    'semantic_search': f"{app.config['API_PREFIX']}/enhanced/semantic-search",
                    'query_database': f"{app.config['API_PREFIX']}/enhanced/query-database",
                    'category_analytics': f"{app.config['API_PREFIX']}/enhanced/analytics/categories",
                    'productivity_insights': f"{app.config['API_PREFIX']}/enhanced/analytics/productivity",
                    'database_performance': f"{app.config['API_PREFIX']}/enhanced/database/performance",
                    'bulk_insert': f"{app.config['API_PREFIX']}/enhanced/database/bulk-insert"
                }
                endpoints.update(advanced_endpoints)
            
            # Add real-time endpoints if available
            if socketio:
                realtime_endpoints = {
                    'realtime_stats': f"{app.config['API_PREFIX']}/enhanced/realtime/stats",
                    'realtime_events': f"{app.config['API_PREFIX']}/enhanced/realtime/events",
                    'websocket_endpoint': '/socket.io/'
                }
                endpoints.update(realtime_endpoints)
            
            # Determine which features are enabled
            features = []
            if available_features['semantic_search']:
                features.append("LLM Semantic Search")
            if available_features['real_time']:
                features.append("Real-time Updates")
            if available_features['advanced_querying']:
                features.append("Advanced Analytics")
            if available_features['database_service']:
                features.append("Database Integration")
            
            return jsonify({
                'message': 'History Plus Main Backend API',
                'version': app.config['API_VERSION'],
                'status': 'running',
                'features': available_features,
                'enabled_features': features,
                'endpoints': endpoints,
                'realtime': {
                    'available': socketio is not None,
                    'events': ['database_change', 'task_update', 'category_learned', 'semantic_search_result', 'analytics_update'],
                    'rooms': ['general', 'dashboard', 'search', 'analytics']
                } if socketio else None,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in root endpoint: {e}")
            return jsonify({
                'message': 'History Plus Main Backend API',
                'version': app.config.get('API_VERSION', '1.0.0'),
                'status': 'error',
                'error': str(e)
            }), 500
    
    # Global error handler
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception: {e}")
        return jsonify({
            'success': False,
            'message': 'An unexpected error occurred',
            'error': str(e) if app.debug else 'Internal server error'
        }), 500
    
    return app, socketio

def run_app(app, socketio, host='127.0.0.1', port=5000, debug=False):
    """Run the application with or without SocketIO"""
    print("\n🚀 History Plus Main Backend")
    print("=" * 50)
    
    if socketio:
        print("✓ Real-time features enabled (SocketIO)")
        print("✓ WebSocket support available")
        socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)
    else:
        print("⚠ Real-time features disabled (no SocketIO)")
        print("✓ Basic API functionality available")
        app.run(host=host, port=port, debug=debug, use_reloader=False)

# Create app instance for import
try:
    app, socketio = create_app()
    logger.info("App instance created for import")
except Exception as e:
    logger.error(f"Failed to create app instance: {e}")
    app, socketio = None, None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='History Plus Main Backend')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-realtime', action='store_true', help='Disable real-time features')
    
    args = parser.parse_args()
    
    if args.no_realtime:
        os.environ['ENABLE_REALTIME'] = 'False'
    
    if not app:
        app, socketio = create_app()
    
    print(f"\nStarting on http://{args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"Real-time features: {not args.no_realtime}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        run_app(app, socketio, host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1) 