import logging
from app import app

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more details
logger = logging.getLogger(__name__)

try:
    logger.debug("Starting Gunicorn application initialization")
    application = app
    logger.debug("Gunicorn application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gunicorn application: {str(e)}", exc_info=True)
    raise
