import os

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Worker processes - Single worker for ML models to avoid multiple model loads
workers = 1
worker_class = 'sync'
worker_connections = 1000

# Timeouts - Increased for model loading and Cloud Run
timeout = 600  # 10 minutes for initial model loading
graceful_timeout = 120  # 2 minutes for graceful shutdown
keepalive = 5  # Increased for Cloud Run's HTTP/2 keepalive

# Worker lifecycle
max_requests = 100  # Restart workers to prevent memory leaks
max_requests_jitter = 10

# Memory management
preload_app = False  # Avoid loading ML models multiple times
worker_tmp_dir = '/dev/shm'

# Logging - Explicit paths for Cloud Run
loglevel = 'info'
accesslog = '-'  # stdout for Cloud Run
errorlog = '-'   # stdout for Cloud Run
capture_output = True
enable_stdio_inheritance = True

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Gunicorn hooks for debugging
def on_starting(server):
    server.log.info("Gunicorn server is starting")

def when_ready(server):
    server.log.info("Gunicorn server is ready. Spawning workers")

def pre_fork(server, worker):
    server.log.info(f"Worker spawning (pid: {worker.pid})")

def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def worker_int(worker):
    worker.log.info(f"Worker received INT or QUIT signal (pid: {worker.pid})")

def worker_abort(worker):
    worker.log.info(f"Worker aborted (pid: {worker.pid})")
