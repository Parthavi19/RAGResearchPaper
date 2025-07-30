import os
import sys
import time
import logging
import threading
import traceback
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Enhanced logging setup for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger.info(f"Environment check - GOOGLE_API_KEY present: {bool(GOOGLE_API_KEY)}")

# Session state
rag_instances = {}
processing_lock = threading.Lock()

# Test imports at startup
RAG_AVAILABLE = False
IMPORT_ERRORS = {}

def test_imports():
    """Test all critical imports at startup"""
    global RAG_AVAILABLE, IMPORT_ERRORS
    
    imports_to_test = [
        ('PyPDF2', 'PyPDF2'),
        ('transformers', 'transformers'), 
        ('torch', 'torch'),
        ('google.generativeai', 'google-generativeai'),
        ('qdrant_client', 'qdrant-client'),
        ('numpy', 'numpy')
    ]
    
    all_imports_ok = True
    
    for module_name, package_name in imports_to_test:
        try:
            __import__(module_name)
            logger.info(f"✓ {package_name} imported successfully")
        except ImportError as e:
            logger.error(f"✗ {package_name} import failed: {e}")
            IMPORT_ERRORS[package_name] = str(e)
            all_imports_ok = False
        except Exception as e:
            logger.error(f"✗ {package_name} error: {e}")
            IMPORT_ERRORS[package_name] = str(e)
            all_imports_ok = False
    
    # Test RAG module specifically
    try:
        from pipelined_research_rag import PipelinedResearchPaperRAG
        logger.info("✓ PipelinedResearchPaperRAG imported successfully")
        RAG_AVAILABLE = True
    except ImportError as e:
        logger.error(f"✗ PipelinedResearchPaperRAG import failed: {e}")
        IMPORT_ERRORS['pipelined_research_rag'] = str(e)
        all_imports_ok = False
    except Exception as e:
        logger.error(f"✗ PipelinedResearchPaperRAG error: {e}")
        IMPORT_ERRORS['pipelined_research_rag'] = str(e)
        all_imports_ok = False
    
    if all_imports_ok:
        logger.info("🎉 All imports successful - RAG system ready")
    else:
        logger.warning("⚠️  Some imports failed - RAG system may not work properly")

# Test imports at startup
test_imports()

# Lazy import for RAG to avoid startup delay
def get_rag_instance(collection_name):
    """Lazily import and initialize RAG with comprehensive error handling"""
    try:
        if not RAG_AVAILABLE:
            raise ImportError("RAG system not available - check import errors")
            
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        logger.info(f"Creating RAG instance for collection: {collection_name}")
        from pipelined_research_rag import PipelinedResearchPaperRAG
        return PipelinedResearchPaperRAG(collection_name)
        
    except ImportError as e:
        logger.error(f"RAG import error: {e}")
        raise Exception(f"RAG system not available: {e}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise Exception(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"Failed to create RAG instance: {e}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to initialize RAG system: {e}")

# HTML template as string
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RAG Research Paper Bot</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .loading { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
  <div class="container mx-auto px-4 py-8">
    <div class="max-w-4xl mx-auto">
      <!-- Header -->
      <div class="text-center mb-8">
        <h1 class="text-4xl font-bold text-gray-800 mb-2">RAG Research Paper Bot</h1>
        <p class="text-gray-600">Upload a PDF research paper and ask questions about it</p>
      </div>

      <!-- System Status -->
      <div id="systemStatus" class="mb-6 hidden"></div>

      <!-- Main Interface -->
      <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
        <!-- Status -->
        <div id="status" class="mb-6 p-4 rounded-lg text-center bg-blue-50 text-blue-800">
          <span class="font-semibold">Initializing...</span> - Checking system status
        </div>

        <!-- Session ID -->
        <div class="mb-6">
          <label class="block text-sm font-medium text-gray-700 mb-2">Session ID (optional)</label>
          <input type="text" id="sessionId" placeholder="Auto-generated if empty" 
                 class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
        </div>

        <!-- Upload Section -->
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-6 hover:border-blue-400 transition-colors">
          <form id="uploadForm" enctype="multipart/form-data">
            <div class="text-center">
              <svg class="mx-auto h-12 w-12 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
              <input type="file" id="fileInput" accept=".pdf" class="hidden" required />
              <button type="button" onclick="document.getElementById('fileInput').click()" 
                      class="mb-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-md border">
                Choose PDF File
              </button>
              <p class="text-sm text-gray-500 mb-4">or drag and drop</p>
              <div id="fileName" class="text-sm text-gray-700 mb-4 hidden"></div>
              <button type="submit" id="uploadBtn" 
                      class="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-md transition-colors">
                Upload and Process
              </button>
            </div>
          </form>
        </div>

        <!-- Query Section -->
        <div class="border border-gray-200 rounded-lg p-6">
          <form id="queryForm">
            <label class="block text-sm font-medium text-gray-700 mb-2">Ask a Question</label>
            <div class="flex gap-2">
              <input type="text" id="queryInput" placeholder="What is the main contribution of this paper?" 
                     class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500" required />
              <button type="submit" id="queryBtn" 
                      class="bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-6 rounded-md transition-colors">
                Ask
              </button>
            </div>
          </form>
        </div>
      </div>

      <!-- Results -->
      <div id="result" class="hidden"></div>
    </div>
  </div>

  <script>
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const resultDiv = document.getElementById('result');
    const sessionIdInput = document.getElementById('sessionId');
    const statusDiv = document.getElementById('status');
    const uploadBtn = document.getElementById('uploadBtn');
    const queryBtn = document.getElementById('queryBtn');
    const fileNameDiv = document.getElementById('fileName');
    const systemStatusDiv = document.getElementById('systemStatus');

    function setStatus(message, type = 'info') {
      const colors = {
        info: 'bg-blue-50 text-blue-800',
        success: 'bg-green-50 text-green-800',
        error: 'bg-red-50 text-red-800',
        warning: 'bg-yellow-50 text-yellow-800'
      };
      statusDiv.className = `mb-6 p-4 rounded-lg text-center ${colors[type]}`;
      statusDiv.innerHTML = `<span class="font-semibold">${type.toUpperCase()}</span> - ${message}`;
    }

    function setLoading(isLoading) {
      uploadBtn.disabled = isLoading;
      queryBtn.disabled = isLoading;
      
      if (isLoading) {
        uploadBtn.innerHTML = '<span class="loading">Processing...</span>';
        queryBtn.innerHTML = '<span class="loading">Processing...</span>';
      } else {
        uploadBtn.textContent = 'Upload and Process';
        queryBtn.textContent = 'Ask';
      }
    }

    function showSystemStatus(systemInfo) {
      let statusHtml = '';
      
      if (!systemInfo.rag_available) {
        statusHtml = `
          <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <h3 class="text-red-800 font-semibold mb-2">⚠️ System Issues Detected</h3>
            <p class="text-red-700 mb-2">The RAG system is not fully operational. Missing dependencies:</p>
            <ul class="text-red-600 text-sm list-disc list-inside">
              ${Object.entries(systemInfo.import_errors || {}).map(([pkg, err]) => 
                `<li><strong>${pkg}:</strong> ${err}</li>`
              ).join('')}
            </ul>
          </div>
        `;
      } else if (!systemInfo.google_api_configured) {
        statusHtml = `
          <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <h3 class="text-yellow-800 font-semibold">⚠️ Configuration Issue</h3>
            <p class="text-yellow-700">Google API key is not configured. Please check server configuration.</p>
          </div>
        `;
      }
      
      if (statusHtml) {
        systemStatusDiv.innerHTML = statusHtml;
        systemStatusDiv.classList.remove('hidden');
      }
    }

    // File input change handler
    fileInput.addEventListener('change', function() {
      if (this.files[0]) {
        fileNameDiv.textContent = `Selected: ${this.files[0].name}`;
        fileNameDiv.classList.remove('hidden');
      } else {
        fileNameDiv.classList.add('hidden');
      }
    });

    // Upload form handler
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (!fileInput.files[0]) {
        setStatus('Please select a PDF file', 'error');
        return;
      }

      const sessionId = sessionIdInput.value.trim() || crypto.randomUUID();
      sessionIdInput.value = sessionId;

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('session_id', sessionId);

      setStatus('Uploading and processing document...', 'info');
      setLoading(true);

      try {
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();
        
        if (!response.ok) {
          throw new Error(result.error || `HTTP ${response.status}: Upload failed`);
        }

        setStatus('Document processed successfully! You can now ask questions.', 'success');
        
        resultDiv.innerHTML = `
          <div class="bg-white rounded-lg shadow-lg p-6">
            <h3 class="text-lg font-semibold text-green-800 mb-2">✅ Upload Complete</h3>
            <div class="text-sm text-gray-600 space-y-1">
              <p><strong>File:</strong> ${fileInput.files[0].name}</p>
              <p><strong>Session:</strong> ${sessionId}</p>
              <p><strong>Processing Time:</strong> ${result.processing_time || 'N/A'}s</p>
            </div>
          </div>
        `;
        resultDiv.classList.remove('hidden');

      } catch (err) {
        console.error('Upload error:', err);
        setStatus(`Upload failed: ${err.message}`, 'error');
        
        resultDiv.innerHTML = `
          <div class="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 class="text-red-800 font-semibold">❌ Upload Failed</h3>
            <p class="text-red-700 mt-1">${err.message}</p>
          </div>
        `;
        resultDiv.classList.remove('hidden');
      } finally {
        setLoading(false);
      }
    });

    // Query form handler
    queryForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const sessionId = sessionIdInput.value.trim();
      const question = queryInput.value.trim();

      if (!sessionId) {
        setStatus('Please upload a document first', 'error');
        return;
      }

      setStatus('Processing your question...', 'info');
      setLoading(true);

      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, session_id: sessionId })
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(result.error || `HTTP ${response.status}: Query failed`);
        }

        setStatus('Question answered successfully!', 'success');
        
        resultDiv.innerHTML = `
          <div class="bg-white rounded-lg shadow-lg p-6">
            <div class="border-l-4 border-blue-500 pl-4 mb-4">
              <h3 class="font-semibold text-gray-800">Question:</h3>
              <p class="text-gray-700">${question}</p>
            </div>
            <div class="border-l-4 border-green-500 pl-4 mb-4">
              <h3 class="font-semibold text-gray-800">Answer:</h3>
              <div class="text-gray-700 whitespace-pre-wrap">${result.answer}</div>
            </div>
            <div class="text-xs text-gray-500 border-t pt-2">
              Response time: ${result.performance?.query_time || 'N/A'}s
            </div>
          </div>
        `;
        resultDiv.classList.remove('hidden');
        
        queryInput.value = '';

      } catch (err) {
        console.error('Query error:', err);
        setStatus(`Query failed: ${err.message}`, 'error');
        
        resultDiv.innerHTML = `
          <div class="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 class="text-red-800 font-semibold">❌ Query Failed</h3>
            <p class="text-red-700 mt-1">${err.message}</p>
          </div>
        `;
        resultDiv.classList.remove('hidden');
      } finally {
        setLoading(false);
      }
    });

    // Test server connection and system status on load
    async function checkSystemStatus() {
      try {
        const [healthResponse, importsResponse] = await Promise.all([
          fetch('/health'),
          fetch('/debug')
        ]);
        
        const healthData = await healthResponse.json();
        const importsData = await importsResponse.json();
        
        if (healthData.status === 'healthy' && importsData.rag_available) {
          setStatus('System ready - upload a PDF to get started', 'success');
        } else {
          setStatus('System issues detected - check configuration', 'warning');
          showSystemStatus(importsData);
        }
        
      } catch (err) {
        setStatus('Unable to connect to server', 'error');
        console.error('System check failed:', err);
      }
    }
    
    // Initialize
    checkSystemStatus();
  </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Home page"""
    logger.info("Home page requested")
    return render_template_string(INDEX_HTML)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'active_sessions': len(rag_instances),
            'google_api_configured': bool(GOOGLE_API_KEY),
            'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
            'rag_available': RAG_AVAILABLE
        }), 200
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/debug', methods=['GET'])
def debug_environment():
    """Debug endpoint to check environment setup"""
    try:
        env_status = {
            'GOOGLE_API_KEY': bool(GOOGLE_API_KEY),
            'QDRANT_URL': bool(os.getenv("QDRANT_URL")),
            'QDRANT_API_KEY': bool(os.getenv("QDRANT_API_KEY")),
            'PORT': os.getenv('PORT', '8080'),
            'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
            'upload_folder_writable': os.access(UPLOAD_FOLDER, os.W_OK) if os.path.exists(UPLOAD_FOLDER) else False,
        }
        
        return jsonify({
            'environment': env_status,
            'import_errors': IMPORT_ERRORS,
            'rag_available': RAG_AVAILABLE,
            'python_version': sys.version,
            'working_directory': os.getcwd()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-imports', methods=['GET'])
def test_imports_endpoint():
    """Test all critical imports"""
    results = {}
    
    # Test each import individually
    imports_to_test = [
        ('os', 'os'),
        ('flask', 'Flask'),
        ('dotenv', 'python-dotenv'),
        ('PyPDF2', 'PyPDF2'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('google.generativeai', 'google-generativeai'),
        ('qdrant_client', 'qdrant-client'),
        ('numpy', 'numpy')
    ]
    
    for module_name, package_name in imports_to_test:
        try:
            __import__(module_name)
            results[package_name] = "✓ Available"
        except ImportError as e:
            results[package_name] = f"✗ Missing: {e}"
        except Exception as e:
            results[package_name] = f"✗ Error: {e}"
    
    # Test RAG module specifically
    try:
        from pipelined_research_rag import PipelinedResearchPaperRAG
        results['pipelined_research_rag'] = "✓ Available"
    except ImportError as e:
        results['pipelined_research_rag'] = f"✗ Import Error: {e}"
    except Exception as e:
        results['pipelined_research_rag'] = f"✗ Error: {e}"
    
    return jsonify({
        'imports': results,
        'google_api_configured': bool(GOOGLE_API_KEY),
        'rag_available': RAG_AVAILABLE,
        'import_errors': IMPORT_ERRORS
    }), 200

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF with comprehensive error handling"""
    start_time = time.time()
    
    try:
        logger.info("=== Upload Request Started ===")
        
        # Check if RAG is available first
        if not RAG_AVAILABLE:
            logger.error("RAG system not available")
            return jsonify({
                'error': 'RAG system not available. Check server configuration.',
                'details': 'Required dependencies may be missing',
                'import_errors': IMPORT_ERRORS
            }), 503
        
        # Validate request
        if 'file' not in request.files:
            logger.error("No file in request.files")
            return jsonify({'error': 'No file provided'}), 400
            
        if 'session_id' not in request.form:
            logger.error("No session_id in request.form")
            return jsonify({'error': 'No session_id provided'}), 400
        
        file = request.files['file']
        session_id = request.form['session_id']
        
        logger.info(f"Processing upload - File: {file.filename}, Session: {session_id}")

        # Validate file
        if not file or file.filename == '':
            logger.error("Empty file provided")
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are supported'}), 400

        # Check environment
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not configured")
            return jsonify({'error': 'Server configuration error: Missing Google API key'}), 500

        # Create RAG instance lazily
        logger.info("Creating RAG instance...")
        with processing_lock:
            if session_id not in rag_instances:
                logger.info(f"Creating new RAG instance for session {session_id}")
                try:
                    collection_name = f"collection_{session_id.replace('-', '_')}"
                    logger.info(f"Collection name: {collection_name}")
                    
                    rag = get_rag_instance(collection_name)
                    rag_instances[session_id] = rag
                    logger.info(f"RAG instance created for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"RAG creation failed: {e}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'error': f'Failed to initialize system: {str(e)}',
                        'type': type(e).__name__
                    }), 500
            else:
                logger.info(f"Using existing RAG instance for session {session_id}")
                rag = rag_instances[session_id]

        # Save file
        logger.info("Saving uploaded file...")
        try:
            filename = secure_filename(f"{session_id}_{int(time.time())}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise Exception("File was not saved properly")
                
            file_size = os.path.getsize(file_path)
            logger.info(f"File saved: {file_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"File save failed: {e}")
            return jsonify({
                'error': f'Failed to save file: {str(e)}',
                'type': type(e).__name__
            }), 500

        # Process paper
        logger.info("Starting paper processing...")
        try:
            processing_start = time.time()
            metadata = rag.load_research_paper(file_path)
            processing_time = time.time() - processing_start
            
            logger.info(f"Paper processing completed in {processing_time:.2f} seconds")
            
            # Clean up file
            try:
                os.remove(file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
            
            total_time = time.time() - start_time
            logger.info(f"Upload completed successfully in {total_time:.2f} seconds")
            
            return jsonify({
                'message': 'Document uploaded and processed successfully',
                'session_id': session_id,
                'processing_time': round(processing_time, 2),
                'total_time': round(total_time, 2),
                'metadata': metadata
            }), 200
            
        except Exception as e:
            logger.error(f"Paper processing failed: {e}")
            logger.error(traceback.format_exc())
            
            try:
                os.remove(file_path)
            except:
                pass
                
            return jsonify({
                'error': f'Failed to process document: {str(e)}',
                'type': type(e).__name__
            }), 500

    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/query', methods=['POST'])
def query():
    """Process query with comprehensive error handling"""
    start_time = time.time()
    
    try:
        logger.info("=== Query Request Started ===")
        
        # Check if RAG is available
        if not RAG_AVAILABLE:
            return jsonify({
                'error': 'RAG system not available',
                'import_errors': IMPORT_ERRORS
            }), 503
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON data'}), 400
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return jsonify({'error': 'Invalid JSON format'}), 400
            
        question = data.get('question', '').strip()
        session_id = data.get('session_id', '').strip()

        if not question:
            return jsonify({'error': 'Question is required'}), 400
            
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        logger.info(f"Query - Session: {session_id}, Question: {question[:100]}...")

        # Get RAG instance
        rag = rag_instances.get(session_id)
        if not rag:
            logger.warning(f"Session {session_id} not found")
            return jsonify({'error': 'Session not found. Please upload a document first.'}), 404

        # Process query
        try:
            logger.info("Processing query...")
            query_start = time.time()
            answer = rag.query(question)
            query_time = time.time() - query_start
            
            total_time = time.time() - start_time
            logger.info(f"Query completed in {query_time:.2f} seconds")

            return jsonify({
                'answer': answer,
                'performance': {
                    'query_time': round(query_time, 2),
                    'total_time': round(total_time, 2),
                    'session_id': session_id
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f'Query failed: {str(e)}',
                'type': type(e).__name__
            }), 500

    except Exception as e:
        logger.error(f"Unexpected query error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/clear/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    """Clear session"""
    try:
        logger.info(f"Clearing session: {session_id}")
        
        with processing_lock:
            rag = rag_instances.pop(session_id, None)

        if not rag:
            return jsonify({'error': 'Session not found'}), 404

        try:
            if hasattr(rag, 'cleanup'):
                rag.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
        
        logger.info(f"Session {session_id} cleared successfully")
        return jsonify({'message': 'Session cleared successfully'}), 200
        
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        return jsonify({'error': f'Clear failed: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Page not found', 'url': request.url}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({'error': 'An unexpected error occurred'}), 500

# Log routes at app initialization
logger.info("=== Available Routes ===")
for rule in app.url_map.iter_rules():
    logger.info(f"{rule.endpoint}: {rule.rule} {list(rule.methods)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {os.environ.get('FLASK_DEBUG', 'False')}")
    logger.info(f"RAG Available: {RAG_AVAILABLE}")
    if IMPORT_ERRORS:
        logger.warning(f"Import Errors: {IMPORT_ERRORS}")
    app.run(host="0.0.0.0", port=port, debug=False)
