import logging
from app import app

# Configure logging for wsgi.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Initializing WSGI application")
application = app
logger.info("WSGI application initialized")
