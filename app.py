import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pipelined_research_rag import PipelinedResearchPaperRAG
import logging
import concurrent.futures
from threading import Lock
import time
from dotenv import load_dotenv  # MISSING in your original import

# Load environment variables from .env file (local only)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check for required environment variables
required_env_vars = ['GOOGLE_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global RAG and threading config
rag_instances = {}
processing_lock = Lock()
query_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Environment configs
API_KEY = os.environ['GOOGLE_API_KEY']
QDRANT_URL = os.environ['QDRANT_URL']
QDRANT_API_KEY = os.environ['QDRANT_API_KEY']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_session_id(request):
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = f"session_{int(time.time())}_{hash(request.remote_addr) % 10000}"
    return session_id

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Received upload request")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    if file.getbuffer().nbytes > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'File exceeds 10MB'}), 400

    session_id = get_session_id(request)
    try:
        with processing_lock:
            if session_id in rag_instances:
                try:
                    rag_instances[session_id].cleanup()
                except Exception:
                    pass
                del rag_instances[session_id]

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)

        rag = PipelinedResearchPaperRAG(
            api_key=API_KEY,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            chunk_size=600,
            overlap=50,
            max_workers=4
        )
        start_time = time.time()
        rag.load_research_paper(file_path)
        processing_time = time.time() - start_time

        with processing_lock:
            rag_instances[session_id] = rag

        os.remove(file_path)

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'session_id': session_id,
            'processing_time': round(processing_time, 2),
            'chunks_created': len(rag.current_data.chunks) if rag.current_data else 0,
            'metadata': rag.paper_metadata
        }), 200

    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Error during upload: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '').strip()
    session_id = data.get('session_id') or get_session_id(request)

    if not question:
        return jsonify({'error': 'Please provide a valid question'}), 400

    with processing_lock:
        rag = rag_instances.get(session_id)

    if not rag:
        return jsonify({'error': 'No document loaded. Please upload a PDF first.', 'session_id': session_id}), 400

    try:
        start_time = time.time()
        future = query_executor.submit(rag.query, question, 10)
        result = future.result(timeout=120)
        result['performance'] = {
            'query_time': round(time.time() - start_time, 2),
            'session_id': session_id
        }
        return jsonify(result), 200
    except concurrent.futures.TimeoutError:
        return jsonify({'error': 'Query timeout'}), 408
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/batch_query', methods=['POST'])
def batch_query():
    data = request.get_json()
    questions = data.get('questions', [])
    session_id = data.get('session_id') or get_session_id(request)

    if not questions or not isinstance(questions, list):
        return jsonify({'error': 'Please provide a list of questions'}), 400

    with processing_lock:
        rag = rag_instances.get(session_id)

    if not rag:
        return jsonify({'error': 'No document loaded. Please upload a PDF first.', 'session_id': session_id}), 400

    start_time = time.time()
    futures = [(i, q, query_executor.submit(rag.query, q.strip(), 10)) for i, q in enumerate(questions) if q.strip()]

    results = []
    for i, question, future in futures:
        try:
            result = future.result(timeout=60)
            results.append({'index': i, 'question': question, 'result': result})
        except Exception as e:
            results.append({'index': i, 'question': question, 'error': str(e)})

    return jsonify({
        'results': results,
        'performance': {
            'total_time': round(time.time() - start_time, 2),
            'average_time_per_query': round((time.time() - start_time) / len(questions), 2),
            'session_id': session_id
        }
    }), 200

@app.route('/status', methods=['GET'])
def status():
    session_id = get_session_id(request)
    with processing_lock:
        session_exists = session_id in rag_instances
    return jsonify({
        'session_id': session_id,
        'session_active': session_exists,
        'system_status': 'healthy',
        'vector_storage': 'qdrant'
    })

@app.route('/health', methods=['GET'])
def health_check():
    from qdrant_client import QdrantClient
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qdrant_client.get_collections()
        return jsonify({'status': 'healthy'}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    data = request.get_json() or {}
    session_id = data.get('session_id') or get_session_id(request)
    cleanup_all = data.get('cleanup_all', False)

    with processing_lock:
        if cleanup_all:
            for sid, rag in rag_instances.items():
                try: rag.cleanup()
                except: pass
            rag_instances.clear()
            return jsonify({'message': 'All sessions cleaned'}), 200
        elif session_id in rag_instances:
            try:
                rag_instances[session_id].cleanup()
                del rag_instances[session_id]
                return jsonify({'message': f'Session {session_id} cleaned'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return jsonify({'message': 'No active session'}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Graceful shutdown
import atexit
def cleanup_on_exit():
    with processing_lock:
        for _, rag in rag_instances.items():
            try: rag.cleanup()
            except: pass
        rag_instances.clear()
    query_executor.shutdown(wait=True)

atexit.register(cleanup_on_exit)
