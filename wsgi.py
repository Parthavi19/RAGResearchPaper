import os
import sys
import logging

# Set up logging before importing the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# Log environment info
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

# Check for required environment variables
required_env_vars = ['GOOGLE_API_KEY']
for var in required_env_vars:
    value = os.getenv(var)
    if value:
        logger.info(f"{var}: {'*' * len(value[:5])}... (length: {len(value)})")
    else:
        logger.error(f"{var}: NOT SET - This will cause errors!")

try:
    from app import app
    logger.info("Successfully imported Flask app")
    
    # Make the application available
    application = app
    
    # Log available routes
    logger.info("=== Available Routes ===")
    with app.app_context():
        for rule in app.url_map.iter_rules():
            logger.info(f"{rule.endpoint}: {rule.rule} {list(rule.methods)}")
    
    logger.info("WSGI application ready")
    
except Exception as e:
    logger.error(f"Failed to import Flask app: {e}")
    import traceback
    logger.error(traceback.format_exc())
    
    # Create a minimal error app
    from flask import Flask, jsonify
    
    error_app = Flask(__name__)
    
    @error_app.route('/')
    def error_home():
        return jsonify({
            'error': 'Application failed to start',
            'message': str(e),
            'status': 'error'
        }), 500
        
    @error_app.route('/health')
    def error_health():
        return jsonify({
            'status': 'unhealthy',
            'error': 'App import failed',
            'message': str(e)
        }), 503
    
    application = error_app
    logger.info("Created error application as fallback")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Running WSGI app directly on port {port}")
    application.run(host='0.0.0.0', port=port)
