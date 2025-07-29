import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = 2
threads = 4
timeout = 300  # Increased for model loading
loglevel = 'debug'
