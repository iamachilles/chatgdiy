from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv
import openai
import assemblyai as aai
from werkzeug.utils import secure_filename
import mimetypes
import tempfile
import gc
import psutil
from threading import Lock

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure processing settings
MAX_MEMORY = 512 * 1024 * 1024  # 512MB
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'mpeg', 'mpga', 'oga', 'webm'}

# Thread safety
processing_lock = Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({"status": "API is running"})

@app.route('/health')
def health_check():
    """Check if the service is running and its memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    status = {
        'status': 'healthy',
        'memory_usage_mb': memory_info.rss / 1024 / 1024,
        'cpu_percent': process.cpu_percent(),
        'working_directory': os.getcwd(),
    }
    
    return jsonify(status)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Basic request validation
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
        
    # Get API details from request
    service = request.form.get('service')
    api_key = request.form.get('api_key')
    
    if not service or not api_key:
        return jsonify({"error": "Missing service or API key"}), 400

    return jsonify({"message": "Transcription endpoint ready", "service": service})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    if not data or 'text' not in data or 'api_key' not in data:
        return jsonify({"error": "Missing required fields"}), 400

    return jsonify({"message": "Summarize endpoint ready"})

# For local development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))