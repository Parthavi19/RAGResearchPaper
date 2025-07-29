import os

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Worker processes
workers = 1  # Reduced due to memory-intensive ML models
worker_class = 'sync'  # Use sync workers instead of threads for stability
worker_connections = 1000

# Timeouts
timeout = 300  # 5 minutes for model loading and processing
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Memory management
preload_app = True  # Preload app to share model memory
worker_tmp_dir = '/dev/shm'  # Use RAM for tmp files

# Logging
loglevel = 'info'
accesslog = '-'
errorlog = '-'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
