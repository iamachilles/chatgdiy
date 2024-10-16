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

def split_audio(file_path, chunk_size=25*1024*1024):  # 25 MB in bytes
    audio = AudioSegment.from_file(file_path)
    chunks = []
    duration = len(audio)
    chunk_duration = chunk_size / (audio.frame_rate * audio.sample_width * audio.channels) * 1000  # in milliseconds

    for i in range(0, duration, int(chunk_duration)):
        chunk = audio[i:i+int(chunk_duration)]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            chunk.export(temp_file.name, format="mp3")
            chunks.append(temp_file.name)

    return chunks

def transcribe_openai(file_path, api_key):
    client = openai.OpenAI(api_key=api_key)
    chunks = split_audio(file_path)
    transcriptions = []

    try:
        for chunk in chunks:
            with open(chunk, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
                transcriptions.append(transcription)
    except Exception as e:
        logger.error(f"OpenAI transcription error: {str(e)}")
        raise Exception(f"OpenAI transcription failed: {str(e)}")
    finally:
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
            config = aai.TranscriptionConfig(
                language_code=language,
                speaker_labels=True,
                language_detection=True if language == 'auto' else False
            )
            transcript = transcriber.transcribe(chunk, config=config)
            
            # Format the transcript with speaker labels
            formatted_transcript = []
            for utterance in transcript.utterances:
                formatted_transcript.append(f"Speaker {utterance.speaker}: {utterance.text}")
            
            transcriptions.append("\n".join(formatted_transcript))
    except Exception as e:
        logger.error(f"AssemblyAI transcription error: {str(e)}")
        raise Exception(f"AssemblyAI transcription failed: {str(e)}")
    finally:
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)

    return "\n".join(transcriptions)

def summarize_text(text, api_key, max_tokens=500):
    client = openai.OpenAI(api_key=api_key)
    
    # Split the text into chunks if it's too long
    max_chunk_length = 4000  # Adjust this value based on the model's token limit
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in chunks:
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
            raise Exception(f"OpenAI summarization failed: {str(e)}")
    
    full_summary = " ".join(summaries)
    
    # If the summary is long, format it as HTML bullet points
    if len(full_summary.split()) > 50:  # Adjust this threshold as needed
        bullet_points = full_summary.split('\n')
        formatted_summary = "<ul>"
        for point in bullet_points:
            if point.strip():
                formatted_summary += f"<li>{point.strip()}</li>"
        formatted_summary += "</ul>"
        return formatted_summary
    else:
        return full_summary

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
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/summarize', methods=['POST'])
def summarize():
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

if __name__ == '__main__':
    app.run(debug=True)