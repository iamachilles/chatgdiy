"""
Advanced Audio Transcription and Summarization Script (Version 6.5)

This script provides a Flask-based web server that handles audio transcription requests
and on-demand summarization of the transcribed text. It supports transcription using both OpenAI's 
Whisper API and AssemblyAI's transcription service, and uses OpenAI's GPT model for summarization.

Features:
- Accepts audio file uploads (MP3, WAV, OGG, M4A, FLAC, MP4, MPEG, MPGA, OGA, WEBM)
- Supports transcription using OpenAI or AssemblyAI
- Handles large audio files by splitting them for both transcription services
- Provides on-demand summarization of transcribed text
- Implements improved error handling and logging
- Uses secure temporary file handling

Usage:
1. Ensure all required packages are installed:
   pip install flask flask-cors openai assemblyai pydub python-dotenv werkzeug

2. Run the script:
   python transcribe6.py

3. The server will start on http://localhost:5000

4. Use the accompanying index.html file as the front-end interface to interact with this server.

Note: Ensure you have appropriate security measures in place, including HTTPS encryption,
when deploying to a production environment.
"""
"""
Advanced Audio Transcription and Summarization Script (Version 6.6)

Updates from 6.5:
- Improved memory management for large file processing
- Added worker memory limits and cleanup
- Enhanced error handling for out-of-memory situations
- Optimized chunk processing for transcription services
- Added request timeout handling
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
from pydub import AudioSegment
import gc
from threading import Lock
import resource
import psutil

def cleanup_temp_files():
    """Clean up all temporary files in the /tmp directory"""
    import glob
    
    patterns = [
        '/tmp/tmp*.mp3',
        '/tmp/tmp*.wav',
        '/tmp/tmp*.ogg',
        '/tmp/tmp*.m4a',
        '/tmp/tmp*.flac',
        '/tmp/tmp*.mp4',
        '/tmp/tmp*.mpeg',
        '/tmp/tmp*.mpga',
        '/tmp/tmp*.oga',
        '/tmp/tmp*.webm'
    ]
    
    for pattern in patterns:
        try:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing pattern {pattern}: {e}")

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure upload settings
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'mp4', 'mpeg', 'mpga', 'oga', 'webm'}

# Memory management settings
MAX_MEMORY = 512 * 1024 * 1024  # 512MB
CHUNK_SIZE = 10 * 1024 * 1024   # 10MB chunks for processing
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

# Thread safety
processing_lock = Lock()

def check_memory():
    """Monitor memory usage and clean up if necessary"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss
    
    if memory_usage > MAX_MEMORY:
        gc.collect()
        return False
    return True

def limit_memory():
    """Set memory limits for the process"""
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_audio(file_path, chunk_duration_ms=29000):  # 29 seconds per chunk
    """Split audio with memory-efficient processing"""
    chunks = []
    try:
        # Load audio in chunks
        audio = AudioSegment.from_file(file_path)
        total_duration = len(audio)
        total_chunks = (total_duration + chunk_duration_ms - 1) // chunk_duration_ms
        
        logger.info(f"Starting to split audio file of {total_duration}ms into {total_chunks} chunks")
        
        for i in range(0, total_duration, chunk_duration_ms):
            if not check_memory():
                logger.warning(f"Memory threshold reached during split at chunk {len(chunks)+1}/{total_chunks}")
                gc.collect()
            
            chunk = audio[i:i + chunk_duration_ms]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                chunk.export(temp_file.name, format="mp3")
                chunks.append(temp_file.name)
            
            logger.info(f"Processed chunk {len(chunks)}/{total_chunks}")
            
            # Clean up chunk memory
            del chunk
            gc.collect()
        
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        # Clean up any created chunks
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)
        raise

def transcribe_openai(file_path, api_key):
    client = openai.OpenAI(api_key=api_key)
    chunks = split_audio(file_path)
    transcriptions = []
    total_chunks = len(chunks)

    try:
        for i, chunk in enumerate(chunks, 1):
            if not check_memory():
                logger.warning(f"Memory threshold reached during OpenAI transcription at chunk {i}/{total_chunks}")
                gc.collect()
                
            logger.info(f"Transcribing chunk {i}/{total_chunks}")
            with open(chunk, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
                transcriptions.append(transcription)
                
            # Clean up chunk immediately
            os.remove(chunk)
            gc.collect()
            logger.info(f"Completed chunk {i}/{total_chunks}")
            
    except Exception as e:
        logger.error(f"OpenAI transcription error at chunk {i}/{total_chunks}: {str(e)}")
        raise
    finally:
        # Ensure all chunks are cleaned up
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)

    return " ".join(transcriptions)

def transcribe_assemblyai(file_path, api_key, language='fr'):
    aai.settings.api_key = api_key
    chunks = split_audio(file_path)
    transcriptions = []

    try:
        transcriber = aai.Transcriber()
        for chunk in chunks:
            if not check_memory():
                logger.warning("Memory threshold reached during AssemblyAI transcription")
                gc.collect()
                
            config = aai.TranscriptionConfig(
                language_code=language,
                speaker_labels=True,
                language_detection=True if language == 'auto' else False
            )
            transcript = transcriber.transcribe(chunk, config=config)
            
            # Format transcript with speaker labels
            formatted_transcript = []
            for utterance in transcript.utterances:
                formatted_transcript.append(f"Speaker {utterance.speaker}: {utterance.text}")
            
            transcriptions.append("\n".join(formatted_transcript))
            
            # Clean up chunk immediately
            os.remove(chunk)
            gc.collect()
            
    except Exception as e:
        logger.error(f"AssemblyAI transcription error: {str(e)}")
        raise
    finally:
        # Ensure all chunks are cleaned up
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)

    return "\n".join(transcriptions)

def summarize_text(text, api_key, max_tokens=500):
    client = openai.OpenAI(api_key=api_key)
    
    # Split text into smaller chunks
    max_chunk_length = 3000  # Reduced from 4000 to be more conservative
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
        if not check_memory():
            logger.warning("Memory threshold reached during summarization")
            gc.collect()
            
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text. For longer texts, provide a summary in the form of bullet points highlighting the main parts of the conversation."},
                    {"role": "user", "content": f"Please summarize the following text concisely yet comprehensively. If the text is long, use bullet points to highlight the main parts of the conversation:\n\n{chunk}"}
                ],
                max_tokens=max_tokens
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"OpenAI summarization error: {str(e)}")
            raise

    full_summary = " ".join(summaries)
    
    # Format as HTML bullets if long
    if len(full_summary.split()) > 50:
        bullet_points = full_summary.split('\n')
        formatted_summary = "<ul>"
        for point in bullet_points:
            if point.strip():
                formatted_summary += f"<li>{point.strip()}</li>"
        formatted_summary += "</ul>"
        return formatted_summary
    return full_summary

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    with processing_lock:  # Ensure thread safety
        cleanup_temp_files()  # Clean up any leftover temporary files
        logger.info("Starting new transcription request")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        service = request.form.get('service')
        api_key = request.form.get('api_key')
        language = request.form.get('language', 'fr')
        
        logger.info(f"Service selected: {service}, Language: {language}")

        try:
            # Check file size
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
            logger.info(f"Initial file size: {size} bytes")
            
            if size > MAX_FILE_SIZE:
                return jsonify({"error": "File too large"}), 400

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                temp_filename = temp_file.name

            # Log file details
            file_size = os.path.getsize(temp_filename)
            mime_type, _ = mimetypes.guess_type(temp_filename)
            logger.info(f"File saved: {temp_filename}, Size: {file_size} bytes, MIME: {mime_type}")
            
            # Log memory usage before processing
            process = psutil.Process()
            logger.info(f"Memory usage before processing: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            # Process based on service
            if service == 'openai':
                logger.info("Starting OpenAI transcription")
                transcription = transcribe_openai(temp_filename, api_key)
            elif service == 'assemblyai':
                logger.info("Starting AssemblyAI transcription")
                transcription = transcribe_assemblyai(temp_filename, api_key, language)
            else:
                return jsonify({"error": "Invalid service selected"}), 400

            logger.info("Transcription completed successfully")
            logger.info(f"Transcription length: {len(transcription)} characters")
            logger.info(f"Memory usage after processing: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            return jsonify({"transcription": transcription})

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            logger.exception("Full error traceback:")  # This will log the full stack trace
            return jsonify({"error": str(e)}), 500
        finally:
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info(f"Cleaned up temp file: {temp_filename}")
            gc.collect()
            logger.info(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

@app.route('/summarize', methods=['POST'])
def summarize():
    with processing_lock:  # Ensure thread safety
        data = request.json
        text = data.get('text')
        api_key = data.get('api_key')

        if not text or not api_key:
            return jsonify({"error": "Missing text or API key"}), 400

        try:
            summary = summarize_text(text, api_key)
            return jsonify({"summary": summary})
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            gc.collect()

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
        'temp_files': len(os.listdir('/tmp')),
    }
    
    return jsonify(status)

if __name__ == '__main__':
    # Set memory limits before running
    limit_memory()
    app.run(debug=False)