import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import assemblyai as aai
from werkzeug.utils import secure_filename
import mimetypes
from pydub import AudioSegment
import gc
from threading import Lock
import resource
import psutil
import tempfile
import sys
import requests
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__, static_folder='../public', static_url_path='')
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Memory management settings
MAX_MEMORY = 512 * 1024 * 1024  # 512MB
CHUNK_SIZE = 10 * 1024 * 1024   # 10MB chunks
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

# Thread safety
processing_lock = Lock()

# Add ALLOWED_EXTENSIONS that was missing
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'mpeg', 'mpga', 'oga', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Update the root route to serve index.html
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health_check():
    """Check if the service is running and its memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    status = {
        'status': 'healthy',
        'memory_usage_mb': memory_info.rss / 1024 / 1024,
        'cpu_percent': process.cpu_percent(),
        'working_directory': os.getcwd()
    }
    
    return jsonify(status)

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    logger.info("Test endpoint called")
    return jsonify({
        "status": "ok",
        "message": "Test endpoint working",
        "environment": os.environ.get('VERCEL_ENV', 'unknown'),
        "python_version": sys.version
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    logger.info("Transcribe endpoint called")
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        service = request.form.get('service')
        api_key = request.form.get('api_key')

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name

        try:
            if service == 'openai':
                # Use requests to call OpenAI API
                headers = {
                    'Authorization': f'Bearer {api_key}'
                }
                files = {
                    'file': open(temp_filename, 'rb'),
                    'model': (None, 'whisper-1')
                }
                response = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, files=files)
                response.raise_for_status()
                result = response.json()['text']
            elif service == 'assemblyai':
                aai.settings.api_key = api_key
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(temp_filename)
                result = transcript.text
            else:
                return jsonify({"error": "Invalid service selected"}), 400

            return jsonify({"transcription": result})

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info(f"Cleaned up temp file: {temp_filename}")

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize():
    logger.info("Summarize endpoint called")
    try:
        data = request.json
        if not data or 'text' not in data or 'api_key' not in data:
            logger.error("Missing required fields in request")
            return jsonify({"error": "Missing required fields"}), 400

        client = OpenAI(api_key=data['api_key'])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following text concisely:"},
                {"role": "user", "content": data['text']}
            ],
            max_completion_tokens=500
        )
        
        summary = response.choices[0].message.content.strip()
        return jsonify({"summary": summary})
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))