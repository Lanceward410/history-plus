import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 'yes']
    
    # CORS settings for cross-origin requests
    CORS_ORIGINS = [
        'chrome-extension://*',  # Allow Chrome extension requests
        'http://localhost:5000', # Local backend requests
    ]
    
    # API settings
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'
    
    # Processing settings - TUNABLE PARAMETERS
    MAX_HISTORY_ENTRIES = int(os.environ.get('MAX_HISTORY_ENTRIES', '10000'))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
    SESSION_GAP_MINUTES = int(os.environ.get('SESSION_GAP_MINUTES', '15'))
    
    # Model settings - TUNABLE PARAMETERS
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    MIN_CLUSTER_SIZE = int(os.environ.get('MIN_CLUSTER_SIZE', '3'))
    
    # OpenAI settings (optional)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    ENABLE_LLM_FEATURES = OPENAI_API_KEY is not None
    
    # Cache settings
    CACHE_TIMEOUT = timedelta(hours=1)
    ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'True').lower() in ['true', '1', 'yes']

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32)

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 