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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for RAG instances and processing
rag_instances = {}
processing_lock = Lock()
query_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Configuration
API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyAVzwqMt0edserFCtiGHlb5g2iOkxZf2SA")
QDRANT_URL = os.environ.get('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY', 'your-qdrant-api-key')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    session_id = get_session_id(request)
    
    try:
        with processing_lock:
            # Clean up any existing RAG instance for this session
            if session_id in rag_instances:
                try:
                    rag_instances[session_id].cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up previous RAG instance: {e}")
                del rag_instances[session_id]
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
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
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove uploaded file: {e}")
        
        logger.info(f"Successfully loaded {file_path} in {processing_time:.2f}s for session {session_id}")
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'session_id': session_id,
            'processing_time': round(processing_time, 2),
            'chunks_created': len(rag.current_data.chunks) if rag.current_data else 0,
            'metadata': rag.paper_metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing file for session {session_id}: {e}")
        # Clean up on error
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query():
    """Handle query with pipelined processing and parallel execution."""
    data = request.get_json()
    question = data.get('question', '')
    session_id = data.get('session_id') or get_session_id(request)
    
    if not question.strip():
        return jsonify({'error': 'Please provide a valid question'}), 400
    
    # Check if RAG instance exists for this session
    with processing_lock:
        rag = rag_instances.get(session_id)
    
    if rag is None:
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
        logger.error(f"Error processing query for session {session_id}: {e}")
        return jsonify({'error': f"Error processing query: {str(e)}"}), 500

@app.route('/batch_query', methods=['POST'])
def batch_query():
    """Handle multiple queries in parallel using pipelining."""
    data = request.get_json()
    questions = data.get('questions', [])
    session_id = data.get('session_id') or get_session_id(request)
    
    if not questions or not isinstance(questions, list):
        return jsonify({'error': 'Please provide a list of questions'}), 400
    
    # Check if RAG instance exists for this session
    with processing_lock:
        rag = rag_instances.get(session_id)
    
    if rag is None:
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
                logger.error(f"Error processing batch query {i}: {e}")
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
        logger.error(f"Error processing batch query for session {session_id}: {e}")
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
                        logger.warning(f"Error cleaning up session {sid}: {e}")
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
                        logger.error(f"Error cleaning up session {session_id}: {e}")
                        return jsonify({'error': f"Error cleaning up session: {str(e)}"}), 500
                else:
                    message = f"Session {session_id} not found or already cleaned up"
        
        return jsonify({'message': message}), 200
        
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")
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
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': active_sessions,
        'pipeline_enabled': True,
        'features': {
            'parallel_processing': True,
            'batch_queries': True,
            'session_management': True,
            'vector_storage': True
        }
    }), 200

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
                logger.error(f"Error cleaning up session {session_id}: {e}")
        rag_instances.clear()
    
    # Shutdown thread pool
    query_executor.shutdown(wait=True)

atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    # For local development only
    logger.info("Starting pipelined RAG application...")
    app.run(host='0.0.0.0', port=8080, debug=True)
