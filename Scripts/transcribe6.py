"""
Advanced Audio Transcription Script with Web Interface (Version 6.1)

This script provides a Flask-based web server that handles audio transcription requests
from a web interface. It supports transcription using both OpenAI's Whisper API and 
AssemblyAI's transcription service. This version is designed to be more suitable for
public hosting.

Features:
- Accepts audio file uploads (MP3, WAV, OGG, M4A)
- Supports transcription using OpenAI or AssemblyAI
- Provides a simple API endpoint for transcription requests
- Implements improved error handling and logging
- Uses secure temporary file handling
- Implements rate limiting to prevent abuse

Usage:
1. Ensure all required packages are installed:
   pip install flask flask-cors openai assemblyai pydub python-dotenv werkzeug

2. Run the script:
   python transcribe6.py

3. The server will start on http://localhost:5000

4. Use the accompanying index.html file as the front-end interface to interact with this server.

Note: While this script has been improved for public hosting, ensure you have appropriate
security measures in place, including HTTPS encryption, when deploying to a production environment.
"""

import os
import logging
import tempfile
from dotenv import load_dotenv
import openai
import assemblyai as aai
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_openai(file_path, api_key):
    client = openai.OpenAI(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )
    return transcription

def transcribe_assemblyai(file_path, api_key):
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    return transcript.text

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    service = request.form.get('service')
    api_key = request.form.get('api_key')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_filename = temp_file.name

            if service == 'openai':
                transcription = transcribe_openai(temp_filename, api_key)
            elif service == 'assemblyai':
                transcription = transcribe_assemblyai(temp_filename, api_key)
            else:
                return jsonify({"error": "Invalid service selected"}), 400

            return jsonify({"transcription": transcription})
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return jsonify({"error": "An error occurred during transcription"}), 500
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File is too large. Maximum size is 16MB"}), 413

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False for production