import os

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Worker processes - CRITICAL: Use only 1 worker for ML models
workers = 1
worker_class = 'sync'
worker_connections = 1000

# Timeouts - Increased for model loading
timeout = 600  # 10 minutes for initial model loading
graceful_timeout = 120  # 2 minutes for graceful shutdown
keepalive = 2

# Worker lifecycle
max_requests = 100  # Restart workers periodically to prevent memory leaks
max_requests_jitter = 10

# Memory management
preload_app = False  # Don't preload to avoid model loading issues
worker_tmp_dir = '/dev/shm'

# Logging
loglevel = 'info'
accesslog = '-'
errorlog = '-'
capture_output = True
enable_stdio_inheritance = True

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

def on_starting(server):
    server.log.info("Gunicorn server is starting")

def when_ready(server):
    server.log.info("Gunicorn server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawning (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker aborted (pid: %s)", worker.pid)
