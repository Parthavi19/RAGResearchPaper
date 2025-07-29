import os
import time
import logging
import threading
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from pipelined_research_rag import PipelinedResearchPaperRAG
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Environment config
API_KEY = os.getenv("API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Session storage
rag_instances = {}
processing_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html', sessions=list(rag_instances.keys()))

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get('file')
    session_id = request.form.get('session_id')

    logging.debug(f"Received upload - session_id: {session_id}, file: {file.filename if file else 'None'}")

    if not file or not session_id:
        return jsonify({'error': 'Missing file or session_id'}), 400

    with processing_lock:
        if session_id not in rag_instances:
            rag = PipelinedResearchPaperRAG(
                api_key=API_KEY,
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY,
                chunk_size=600,
                overlap=50,
                max_workers=4
            )
            rag_instances[session_id] = rag
        else:
            rag = rag_instances[session_id]

    filename = secure_filename(f"{session_id}_{file.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    logging.debug(f"File saved to: {file_path}")

    try:
        metadata = rag.load_research_paper(file_path)
        logging.debug(f"Metadata loaded: {metadata}")
        return jsonify({
            'message': 'Paper uploaded and indexed successfully.',
            'session_id': session_id,
            'collection_name': rag.collection_name,
            'paper_metadata': metadata
        }), 200
    except Exception as e:
        logging.exception("Failed to process paper.")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id')

    logging.debug(f"Query received - session_id: {session_id}, question: {question}")

    if not question or not session_id:
        return jsonify({'error': 'Missing question or session_id'}), 400

    rag = rag_instances.get(session_id)
    if not rag:
        return jsonify({'error': 'Invalid session_id'}), 404

    try:
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=rag.max_workers) as executor:
            future = executor.submit(rag.query, question)
            result = future.result(timeout=60)
        duration = time.time() - start_time

        return jsonify({
            'answer': result,
            'performance': {
                'query_time': round(duration, 2),
                'session_id': session_id
            }
        }), 200
    except Exception as e:
        logging.exception("Error during query.")
        return jsonify({'error': str(e)}), 500

@app.route('/clear/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    with processing_lock:
        rag = rag_instances.pop(session_id, None)

    if not rag:
        return jsonify({'error': 'Session not found'}), 404

    try:
        rag.cleanup()
        return jsonify({'message': 'Session cleared successfully.'}), 200
    except Exception as e:
        logging.exception("Cleanup failed.")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        return jsonify({'status': 'healthy'}), 200
    except UnexpectedResponse as e:
        logging.warning(f"Qdrant health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
