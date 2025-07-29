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

# Import RAG after logging setup with timeout protection
def safe_import_rag():
    """Safely import RAG with timeout"""
    try:
        logger.info("Importing PipelinedResearchPaperRAG...")
        from pipelined_research_rag import PipelinedResearchPaperRAG
        logger.info("Successfully imported PipelinedResearchPaperRAG")
        return PipelinedResearchPaperRAG
    except Exception as e:
        logger.error(f"Failed to import PipelinedResearchPaperRAG: {e}")
        logger.error(traceback.format_exc())
        raise

# Import with error handling
PipelinedResearchPaperRAG = safe_import_rag()

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

# HTML template as string (since templates folder might not exist)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RAG Bot</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex flex-col items-center justify-start py-10 px-4">
  <div class="bg-white shadow-lg rounded-lg p-6 max-w-xl w-full">
    <h1 class="text-2xl font-bold mb-4 text-center">RAG Bot</h1>

    <!-- Status indicator -->
    <div id="status" class="mb-4 p-2 rounded text-center bg-blue-100 text-blue-800">
      Ready to upload documents
    </div>

    <!-- Session ID Input -->
    <input type="text" id="sessionId" placeholder="Session ID (auto-generated if empty)" class="border p-2 mb-4 w-full">

    <!-- Upload Form -->
    <form id="uploadForm" class="mb-4" enctype="multipart/form-data" method="POST">
      <input type="file" id="fileInput" accept=".pdf" class="mb-2" required />
      <button type="submit" id="uploadBtn" class="bg-blue-500 text-white px-4 py-2 rounded w-full">Upload Document</button>
    </form>

    <!-- Query Form -->
    <form id="queryForm" class="mb-4">
      <input type="text" id="queryInput" placeholder="Enter your question..." class="border p-2 w-full mb-2" required />
      <button type="submit" id="queryBtn" class="bg-green-500 text-white px-4 py-2 rounded w-full">Ask Question</button>
    </form>

    <!-- Results -->
    <div id="result" class="mt-6"></div>
  </div>

  <script>
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const resultContent = document.getElementById('result');
    const sessionIdInput = document.getElementById('sessionId');
    const statusDiv = document.getElementById('status');
    const uploadBtn = document.getElementById('uploadBtn');
    const queryBtn = document.getElementById('queryBtn');

    function setStatus(message, type = 'info') {
      const colors = {
        info: 'bg-blue-100 text-blue-800',
        success: 'bg-green-100 text-green-800',
        error: 'bg-red-100 text-red-800',
        warning: 'bg-yellow-100 text-yellow-800'
      };
      statusDiv.className = `mb-4 p-2 rounded text-center ${colors[type]}`;
      statusDiv.textContent = message;
    }

    function setLoading(loading) {
      uploadBtn.disabled = loading;
      queryBtn.disabled = loading;
      if (loading) {
        uploadBtn.textContent = 'Processing...';
        queryBtn.textContent = 'Processing...';
      } else {
        uploadBtn.textContent = 'Upload Document';
        queryBtn.textContent = 'Ask Question';
      }
    }

    // Upload document
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (fileInput.files.length === 0) {
        setStatus('Please select a PDF file to upload', 'error');
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
          throw new Error(result.error || 'Upload failed');
        }

        setStatus('Document uploaded successfully! You can now ask questions.', 'success');
        resultContent.innerHTML = `
          <div class="bg-green-50 border border-green-200 rounded p-4">
            <h3 class="font-semibold text-green-800">Upload Complete</h3>
            <p class="text-sm text-green-600 mt-1">${result.message}</p>
            <p class="text-xs text-gray-500 mt-2">Session: ${sessionId}</p>
          </div>
        `;
      } catch (err) {
        console.error('Upload error:', err);
        setStatus('Upload failed. Please try again.', 'error');
        resultContent.innerHTML = `<div class="bg-red-50 border border-red-200 rounded p-4 text-red-800">Error: ${err.message}</div>`;
      } finally {
        setLoading(false);
      }
    });

    // Query the uploaded document
    queryForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const sessionId = sessionIdInput.value.trim();
      const queryInputText = queryInput.value.trim();

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
          body: JSON.stringify({ question: queryInputText, session_id: sessionId })
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(result.error || 'Query failed');
        }

        setStatus('Question answered successfully!', 'success');
        resultContent.innerHTML = `
          <div class="bg-gray-50 border rounded p-4">
            <h3 class="font-semibold text-gray-800 mb-2">Question:</h3>
            <p class="text-gray-700 mb-4">${queryInputText}</p>
            <h3 class="font-semibold text-gray-800 mb-2">Answer:</h3>
            <div class="text-gray-700 mb-4 whitespace-pre-wrap">${result.answer}</div>
            <div class="text-xs text-gray-500 border-t pt-2">
              Response time: ${result.performance?.query_time?.toFixed(2) || "?"} seconds
            </div>
          </div>
        `;
        
        // Clear the query input
        queryInput.value = '';
        
      } catch (err) {
        console.error('Query error:', err);
        setStatus('Query failed. Please try again.', 'error');
        resultContent.innerHTML = `<div class="bg-red-50 border border-red-200 rounded p-4 text-red-800">Error: ${err.message}</div>`;
      } finally {
        setLoading(false);
      }
    });
  </script>
</body>
</html>
"""

# Home route
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

# Health check route (simple and fast)
@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify({
            'status': 'healthy',
            'active_sessions': len(rag_instances),
            'google_api_configured': bool(GOOGLE_API_KEY)
        }), 200
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

# Upload PDF route
@app.route('/upload', methods=['POST'])
def upload_pdf():
    start_time = time.time()
    
    try:
        logger.info("=== Upload Request Started ===")
        
        # Validate request format
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

        # Create or get RAG instance
        logger.info("Getting RAG instance...")
        with processing_lock:
            if session_id not in rag_instances:
                logger.info(f"Creating new RAG instance for session {session_id}")
                try:
                    # Clean collection name for Qdrant
                    collection_name = f"collection_{session_id.replace('-', '_')}"
                    logger.info(f"Collection name: {collection_name}")
                    
                    # Create RAG instance - ONLY pass collection_name
                    rag = PipelinedResearchPaperRAG(collection_name)
                    rag_instances[session_id] = rag
                    logger.info(f"RAG instance created for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"RAG creation failed: {e}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Failed to initialize system: {str(e)}'}), 500
            else:
                logger.info(f"Using existing RAG instance for session {session_id}")
                rag = rag_instances[session_id]

        # Save uploaded file
        logger.info("Saving uploaded file...")
        try:
            filename = secure_filename(f"{session_id}_{int(time.time())}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Verify file exists and has content
            if not os.path.exists(file_path):
                raise Exception("File was not saved successfully")
                
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception("Uploaded file is empty")
                
            logger.info(f"File saved: {file_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"File save failed: {e}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

        # Process the paper
        logger.info("Starting paper processing...")
        try:
            processing_start = time.time()
            metadata = rag.load_research_paper(file_path)
            processing_time = time.time() - processing_start
            
            logger.info(f"Paper processing completed in {processing_time:.2f} seconds")
            
            # Clean up uploaded file
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
            
            # Cleanup on failure
            try:
                os.remove(file_path)
            except:
                pass
                
            return jsonify({'error': f'Failed to process document: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Query route
@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    
    try:
        logger.info("=== Query Request Started ===")
        
        # Parse JSON request
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
            return jsonify({'error': f'Query failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected query error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Clear session route
@app.route('/clear/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    try:
        logger.info(f"Clearing session: {session_id}")
        
        with processing_lock:
            rag = rag_instances.pop(session_id, None)

        if not rag:
            return jsonify({'error': 'Session not found'}), 404

        # Cleanup
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
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
