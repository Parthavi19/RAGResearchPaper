import os
import sys
import time
import logging
import threading
import traceback
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
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

# Import RAG after logging setup
try:
    from pipelined_research_rag import PipelinedResearchPaperRAG
    logger.info("Successfully imported PipelinedResearchPaperRAG")
except Exception as e:
    logger.error(f"Failed to import PipelinedResearchPaperRAG: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

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

# Test imports and dependencies
try:
    import google.generativeai as genai
    logger.info("Successfully imported google.generativeai")
except Exception as e:
    logger.error(f"Failed to import google.generativeai: {e}")

try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
except Exception as e:
    logger.error(f"Failed to import torch: {e}")

try:
    from transformers import AutoTokenizer, AutoModel
    logger.info("Successfully imported transformers")
except Exception as e:
    logger.error(f"Failed to import transformers: {e}")

try:
    from qdrant_client import QdrantClient
    logger.info("Successfully imported qdrant_client")
except Exception as e:
    logger.error(f"Failed to import qdrant_client: {e}")

try:
    import PyPDF2
    logger.info("Successfully imported PyPDF2")
except Exception as e:
    logger.error(f"Failed to import PyPDF2: {e}")

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

    <!-- Session ID Input -->
    <input type="text" id="sessionId" placeholder="Enter session ID (or leave blank)" class="border p-2 mb-4 w-full">

    <!-- Upload Form -->
    <form id="uploadForm" class="mb-4" enctype="multipart/form-data" method="POST">
      <input type="file" id="fileInput" accept=".pdf" class="mb-2" required />
      <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded">Upload Document</button>
    </form>

    <!-- Query Form -->
    <form id="queryForm" class="mb-4">
      <input type="text" id="queryInput" placeholder="Enter your question..." class="border p-2 w-full mb-2" required />
      <button type="submit" class="bg-green-500 text-white px-4 py-2 rounded">Ask</button>
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

    // Upload document
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (fileInput.files.length === 0) {
        resultContent.innerHTML = `<p class="text-red-500">Please select a file to upload.</p>`;
        return;
      }

      const sessionId = sessionIdInput.value.trim() || crypto.randomUUID();
      sessionIdInput.value = sessionId;

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('session_id', sessionId);

      resultContent.innerHTML = `<p class="text-blue-500">Uploading and processing...</p>`;

      try {
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();
        
        if (!response.ok) {
          throw new Error(result.error || 'Upload failed');
        }

        resultContent.innerHTML = `<p class="text-green-600 font-semibold">${result.message}</p>`;
      } catch (err) {
        console.error('Upload error:', err);
        resultContent.innerHTML = `<p class="text-red-500">Upload failed: ${err.message}</p>`;
      }
    });

    // Query the uploaded document
    queryForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const sessionId = sessionIdInput.value.trim();
      const queryInputText = queryInput.value.trim();

      if (!sessionId) {
        resultContent.innerHTML = `<p class="text-red-500">Please upload a document first or enter a session ID.</p>`;
        return;
      }

      resultContent.innerHTML = `<p class="text-blue-500">Processing query...</p>`;

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

        resultContent.innerHTML = `
          <div class="bg-gray-50 p-4 rounded">
            <p><strong>Question:</strong> ${queryInputText}</p>
            <p><strong>Answer:</strong> ${result.answer}</p>
            <p class="text-sm text-gray-600 mt-2">
              <strong>Session:</strong> ${sessionId} | 
              <strong>Query Time:</strong> ${result.performance?.query_time?.toFixed(2) || "?"} seconds
            </p>
          </div>
        `;
      } catch (err) {
        console.error('Query error:', err);
        resultContent.innerHTML = `<p class="text-red-500">Query failed: ${err.message}</p>`;
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

# Upload PDF route
@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        logger.info("Upload endpoint called")
        
        # Validate request
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file provided'}), 400
            
        if 'session_id' not in request.form:
            logger.error("No session_id in request")
            return jsonify({'error': 'No session_id provided'}), 400
        
        file = request.files['file']
        session_id = request.form['session_id']
        
        logger.info(f"Processing file: {file.filename} for session: {session_id}")

        if not file or file.filename == '':
            logger.error("Empty file provided")
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are supported'}), 400

        # Check if Google API key is available
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            return jsonify({'error': 'Server configuration error: Missing Google API key'}), 500

        logger.info(f"Google API key found: {google_api_key[:10]}...")

        # Create RAG instance
        with processing_lock:
            if session_id not in rag_instances:
                logger.info(f"Creating new RAG instance for session {session_id}")
                try:
                    collection_name = f"collection_{session_id}"
                    logger.info(f"Initializing RAG with collection: {collection_name}")
                    
                    rag = PipelinedResearchPaperRAG(collection_name=collection_name)
                    rag_instances[session_id] = rag
                    logger.info(f"RAG instance created successfully for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to create RAG instance: {e}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Failed to initialize RAG system: {str(e)}'}), 500
            else:
                logger.info(f"Using existing RAG instance for session {session_id}")
                rag = rag_instances[session_id]

        # Save file
        try:
            filename = secure_filename(f"{session_id}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File saved to: {file_path}")
            
            # Verify file was saved
            if not os.path.exists(file_path):
                raise Exception(f"File was not saved properly: {file_path}")
                
            file_size = os.path.getsize(file_path)
            logger.info(f"Saved file size: {file_size} bytes")
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

        # Process paper
        try:
            logger.info("Starting paper processing...")
            metadata = rag.load_research_paper(file_path)
            logger.info("Paper processing completed successfully")
            
            # Clean up the uploaded file
            try:
                os.remove(file_path)
                logger.info("Temporary file cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file: {cleanup_error}")
            
            return jsonify({
                'message': 'Paper uploaded and indexed successfully.',
                'session_id': session_id,
                'collection_name': rag.collection_name,
                'paper_metadata': metadata
            }), 200
            
        except Exception as e:
            logger.error(f"Failed to process paper: {e}")
            logger.error(traceback.format_exc())
            
            # Clean up on failure
            try:
                os.remove(file_path)
            except:
                pass
                
            return jsonify({'error': f'Failed to process paper: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected error in upload_pdf: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

# Query route
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        question = data.get('question')
        session_id = data.get('session_id')

        if not question or not session_id:
            logger.error("Missing question or session_id in request")
            return jsonify({'error': 'Missing question or session_id'}), 400

        rag = rag_instances.get(session_id)
        if not rag:
            logger.warning(f"Session ID {session_id} not found")
            return jsonify({'error': 'Invalid session_id. Please upload a document first.'}), 404

        try:
            start_time = time.time()
            result = rag.query(question)
            duration = time.time() - start_time

            return jsonify({
                'answer': result,
                'performance': {
                    'query_time': round(duration, 2),
                    'session_id': session_id
                }
            }), 200
        except Exception as e:
            logger.exception("Error during query processing")
            return jsonify({'error': f'Query processing failed: {str(e)}'}), 500

    except Exception as e:
        logger.exception("Unexpected error in query")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Clear session route
@app.route('/clear/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    try:
        with processing_lock:
            rag = rag_instances.pop(session_id, None)

        if not rag:
            logger.warning(f"Tried to clear non-existent session {session_id}")
            return jsonify({'error': 'Session not found'}), 404

        try:
            # Clean up Qdrant collection if possible
            if hasattr(rag, 'qdrant_client') and hasattr(rag, 'collection_name'):
                try:
                    rag.qdrant_client.delete_collection(rag.collection_name)
                except:
                    pass  # Ignore cleanup errors
            
            return jsonify({'message': 'Session cleared successfully.'}), 200
        except Exception as e:
            logger.exception("Cleanup failed")
            return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

    except Exception as e:
        logger.exception("Unexpected error in clear_session")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Health check
@app.route('/health', methods=['GET'])
def health():
    try:
        # Simple health check
        return jsonify({
            'status': 'healthy',
            'active_sessions': len(rag_instances),
            'google_api_key_configured': bool(GOOGLE_API_KEY)
        }), 200
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error")
    return jsonify({'error': 'Internal server error'}), 500

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
