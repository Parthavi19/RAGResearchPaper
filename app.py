import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pipelined_research_rag import PipelinedResearchPaperRAG
import logging
import asyncio
import concurrent.futures
from threading import Lock
import time

# Load environment variables from .env file (optional for local testing)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Use DEBUG for initial deployment troubleshooting
logger = logging.getLogger(__name__)

# Check for required environment variables
required_env_vars = ['GOOGLE_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Use absolute path for Cloud Run
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Ensure upload folder exists
try:
    logger.debug(f"Creating upload folder: {app.config['UPLOAD_FOLDER']}")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create upload folder: {str(e)}", exc_info=True)
    raise

# Global variables for RAG instances and processing
rag_instances = {}
processing_lock = Lock()
query_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Configuration
API_KEY = os.environ.get('GOOGLE_API_KEY')
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')

logger.debug("Flask app initialized successfully")

def allowed_file(filename):
    """Check if the file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_session_id(request):
    """Get or create a session ID for the request."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = f"session_{int(time.time())}_{hash(request.remote_addr) % 10000}"
    return session_id

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and initialize RAG with pipelined processing."""
    logger.debug("Received upload request")
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Validate file size
    if file.getbuffer().nbytes > app.config['MAX_CONTENT_LENGTH']:
        logger.error("File size exceeds 10MB limit")
        return jsonify({'error': 'File size exceeds 10MB limit'}), 400
    
    session_id = get_session_id(request)
    
    try:
        with processing_lock:
            # Clean up any existing RAG instance for this session
            if session_id in rag_instances:
                try:
                    rag_instances[session_id].cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up previous RAG instance: {e}", exc_info=True)
                del rag_instances[session_id]
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        try:
            file.save(file_path)
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {e}", exc_info=True)
            return jsonify({'error': f"Failed to save file: {str(e)}"}), 500
        
        # Initialize pipelined RAG
        logger.info(f"Initializing pipelined RAG for session {session_id}")
        rag = PipelinedResearchPaperRAG(
            api_key=API_KEY,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            chunk_size=600,
            overlap=50,
            max_workers=4
        )
        
        # Process document using pipeline
        start_time = time.time()
        rag.load_research_paper(file_path)
        processing_time = time.time() - start_time
        
        # Store RAG instance
        with processing_lock:
            rag_instances[session_id] = rag
        
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove uploaded file {file_path}: {e}", exc_info=True)
        
        logger.info(f"Successfully loaded {file_path} in {processing_time:.2f}s for session {session_id}")
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'session_id': session_id,
            'processing_time': round(processing_time, 2),
            'chunks_created': len(rag.current_data.chunks) if rag.current_data else 0,
            'metadata': rag.paper_metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing file for session {session_id}: {e}", exc_info=True)
        # Clean up on error
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove file {file_path}: {e}", exc_info=True)
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query():
    """Handle query with pipelined processing and parallel execution."""
    data = request.get_json()
    question = data.get('question', '')
    session_id = data.get('session_id') or get_session_id(request)
    
    if not question.strip():
        logger.error("Empty question provided")
        return jsonify({'error': 'Please provide a valid question'}), 400
    
    # Check if RAG instance exists for this session
    with processing_lock:
        rag = rag_instances.get(session_id)
    
    if rag is None:
        logger.error(f"No document loaded for session {session_id}")
        return jsonify({
            'error': 'No document loaded for this session. Please upload a PDF first.',
            'session_id': session_id
        }), 400
    
    try:
        # Process query using pipelined RAG with timeout
        start_time = time.time()
        
        # Submit query to thread pool for parallel processing
        future = query_executor.submit(rag.query, question, 10)
        
        try:
            result = future.result(timeout=120)  # 2 minute timeout
            query_time = time.time() - start_time
            
            # Add performance metrics
            result['performance'] = {
                'query_time': round(query_time, 2),
                'session_id': session_id
            }
            
            logger.info(f"Query processed in {query_time:.2f}s for session {session_id}")
            return jsonify(result), 200
            
        except concurrent.futures.TimeoutError:
            logger.error(f"Query timeout for session {session_id}")
            return jsonify({'error': 'Query processing timeout. Please try again.'}), 408
            
    except Exception as e:
        logger.error(f"Error processing query for session {session_id}: {e}", exc_info=True)
        return jsonify({'error': f"Error processing query: {str(e)}"}), 500

@app.route('/batch_query', methods=['POST'])
def batch_query():
    """Handle multiple queries in parallel using pipelining."""
    data = request.get_json()
    questions = data.get('questions', [])
    session_id = data.get('session_id') or get_session_id(request)
    
    if not questions or not isinstance(questions, list):
        logger.error("Invalid or empty questions list")
        return jsonify({'error': 'Please provide a list of questions'}), 400
    
    # Check if RAG instance exists for this session
    with processing_lock:
        rag = rag_instances.get(session_id)
    
    if rag is None:
        logger.error(f"No document loaded for session {session_id}")
        return jsonify({
            'error': 'No document loaded for this session. Please upload a PDF first.',
            'session_id': session_id
        }), 400
    
    try:
        start_time = time.time()
        
        # Process queries in parallel using thread pool
        futures = []
        for i, question in enumerate(questions):
            if question.strip():
                future = query_executor.submit(rag.query, question.strip(), 10)
                futures.append((i, question, future))
        
        # Collect results
        results = []
        for i, question, future in futures:
            try:
                result = future.result(timeout=60)  # 1 minute per query
                results.append({
                    'index': i,
                    'question': question,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Error processing batch query {i}: {e}", exc_info=True)
                results.append({
                    'index': i,
                    'question': question,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        response = {
            'results': results,
            'performance': {
                'total_time': round(total_time, 2),
                'average_time_per_query': round(total_time / len(questions), 2),
                'session_id': session_id,
                'queries_processed': len(results)
            }
        }
        
        logger.info(f"Batch query processed {len(results)} queries in {total_time:.2f}s")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing batch query for session {session_id}: {e}", exc_info=True)
        return jsonify({'error': f"Error processing batch query: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get system status and active sessions."""
    session_id = get_session_id(request)
    
    with processing_lock:
        active_sessions = list(rag_instances.keys())
        session_exists = session_id in rag_instances
    
    return jsonify({
        'session_id': session_id,
        'session_active': session_exists,
        'total_active_sessions': len(active_sessions),
        'system_status': 'healthy',
        'pipeline_features': {
            'parallel_processing': True,
            'batch_queries': True,
            'vector_storage': 'qdrant',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }
    }), 200

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    """Clean up a specific session or all sessions."""
    data = request.get_json() or {}
    session_id = data.get('session_id') or get_session_id(request)
    cleanup_all = data.get('cleanup_all', False)
    
    try:
        with processing_lock:
            if cleanup_all:
                # Clean up all sessions
                for sid, rag in list(rag_instances.items()):
                    try:
                        rag.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up session {sid}: {e}", exc_info=True)
                rag_instances.clear()
                message = "All sessions cleaned up successfully"
            else:
                # Clean up specific session
                if session_id in rag_instances:
                    try:
                        rag_instances[session_id].cleanup()
                        del rag_instances[session_id]
                        message = f"Session {session_id} cleaned up successfully"
                    except Exception as e:
                        logger.error(f"Error cleaning up session {session_id}: {e}", exc_info=True)
                        return jsonify({'error': f"Error cleaning up session: {str(e)}"}), 500
                else:
                    message = f"Session {session_id} not found or already cleaned up"
        
        return jsonify({'message': message}), 200
        
    except Exception as e:
        logger.error(f"Error in cleanup: {e}", exc_info=True)
        return jsonify({'error': f"Error in cleanup: {str(e)}"}), 500

@app.route('/results', methods=['GET'])
def results():
    """Render results page (for manual navigation, if needed)."""
    return render_template('results.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint."""
    with processing_lock:
        active_sessions = len(rag_instances)
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qdrant_client.get_collections()
        return jsonify({
            'status': 'healthy',
            'active_sessions': active_sessions,
            'initialized': True,
            'port': os.environ.get('PORT', '8080')
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Cleanup on app shutdown
import atexit

def cleanup_on_exit():
    """Clean up all RAG instances on application shutdown."""
    logger.info("Cleaning up RAG instances on shutdown...")
    with processing_lock:
        for session_id, rag in rag_instances.items():
            try:
                rag.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}", exc_info=True)
        rag_instances.clear()
    
    # Shutdown thread pool
    query_executor.shutdown(wait=True)

atexit.register(cleanup_on_exit)

# Remove local development block for Cloud Run
# if __name__ == '__main__':
#     logger.info("Starting pipelined RAG application...")
#     app.run(host='0.0.0.0', port=8080, debug=True)
