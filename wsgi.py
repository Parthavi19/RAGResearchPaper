import logging
import os
from app import app

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    logger.debug("Starting Gunicorn application initialization")
    application = app
    logger.debug(f"Gunicorn application initialized successfully, binding to PORT={os.environ.get('PORT', '8080')}")
except Exception as e:
    logger.error(f"Failed to initialize Gunicorn application: {str(e)}", exc_info=True)
    raise
