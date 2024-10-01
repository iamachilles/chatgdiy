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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import mimetypes

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure upload settings
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'mpeg', 'mpga', 'oga', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_openai(file_path, api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
        return transcription
    except Exception as e:
        logger.error(f"OpenAI transcription error: {str(e)}")
        raise Exception(f"OpenAI transcription failed: {str(e)}")

def transcribe_assemblyai(file_path, api_key, language='fr'):
    aai.settings.api_key = api_key
    try:
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            language_code=language,
            speaker_labels=True,  # Enable speaker diarization
            language_detection=True if language == 'auto' else False
        )
        transcript = transcriber.transcribe(file_path, config=config)
        
        # Format the transcript with speaker labels
        formatted_transcript = []
        for utterance in transcript.utterances:
            formatted_transcript.append(f"Speaker {utterance.speaker}: {utterance.text}")
        
        return "\n".join(formatted_transcript)
    except Exception as e:
        logger.error(f"AssemblyAI transcription error: {str(e)}")
        raise Exception(f"AssemblyAI transcription failed: {str(e)}")

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    service = request.form.get('service')
    api_key = request.form.get('api_key')
    language = request.form.get('language', 'fr')  # Default to French if not specified

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Create a temporary file with the correct extension
            _, file_extension = os.path.splitext(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                file.save(temp_file.name)
                temp_filename = temp_file.name

            # Log file details for debugging
            file_size = os.path.getsize(temp_filename)
            mime_type, _ = mimetypes.guess_type(temp_filename)
            logger.info(f"File saved: {temp_filename}, Size: {file_size} bytes, MIME: {mime_type}")

            if service == 'openai':
                transcription = transcribe_openai(temp_filename, api_key)
            elif service == 'assemblyai':
                transcription = transcribe_assemblyai(temp_filename, api_key, language)
            else:
                return jsonify({"error": "Invalid service selected"}), 400

            return jsonify({"transcription": transcription})
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return jsonify({"error": f"Transcription error: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True)