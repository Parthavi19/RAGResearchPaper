import multiprocessing

workers = min(multiprocessing.cpu_count() * 2 + 1, 4)
threads = 2
worker_class = 'gthread'
timeout = 120
max_requests = 1000
max_requests_jitter = 100
loglevel = 'info'
errorlog = '-'
accesslog = '-'
preload_app = True
bind = '0.0.0.0:8080'
