"""
Enhanced Audio Transcription Script (Version 2)

This script transcribes an MP3 audio file using OpenAI's Whisper API and processes the transcription into a readable format with improved features.

Features:
- Splits long audio files into manageable chunks for better processing
- Transcribes audio using OpenAI's Whisper API (unlike Version 1, which used local processing)
- Processes the transcription to include accurate timestamps
- Implements a more sophisticated sentence segmentation algorithm
- Saves the transcribed text to a file with improved formatting

Improvements over Version 1:
- Uses OpenAI's API for potentially more accurate transcription
- Handles longer audio files by splitting them into chunks
- Provides more accurate timestamps for each transcribed segment
- Implements smarter sentence segmentation for improved readability

Usage:
python transcribe2.py

Before running:
1. Ensure you have a .env file with your OpenAI API key:
   OPENAI_API_KEY=your_openai_api_key

2. Install required libraries:
   pip install python-dotenv openai pydub

3. Modify the file_path variable in the main() function to point to your MP3 file.

The script will process the audio file and save the transcription as a text file in the same directory.

Note: This script requires an active internet connection to access the OpenAI API.

For more detailed instructions or troubleshooting, refer to the README.md file in the project repository.
"""

import os
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
import math

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def split_audio(file_path, chunk_duration_ms=1200000):  # 20 minutes in milliseconds
    audio = AudioSegment.from_mp3(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i+chunk_duration_ms]
        temp_file = f"temp_chunk_{i//chunk_duration_ms}.mp3"
        chunk.export(temp_file, format="mp3")
        chunks.append((temp_file, i/1000))  # Store start time in seconds
    return chunks

def transcribe_audio(file_path):
    print(f"Transcribing {file_path}...")
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="srt"
        )
    return transcription

import re

def process_transcription(transcription, chunk_start_time, max_interval=30):
    blocks = transcription.split('\n\n')
    processed_lines = []
    current_sentence = ""
    current_timestamp = None
    last_timestamp = 0

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            time_range = lines[1].split(' --> ')[0]
            current_time = time_to_seconds(time_range) + chunk_start_time
            text = ' '.join(lines[2:])
            
            if current_timestamp is None:
                current_timestamp = current_time
                last_timestamp = current_time

            current_sentence += text + " "
            
            if current_time - last_timestamp >= max_interval or re.search(r'[.!?]\s*$', text):
                processed_lines.append(f"[{seconds_to_time(current_timestamp)}] {current_sentence.strip()}")
                current_sentence = ""
                current_timestamp = None
                last_timestamp = current_time

    if current_sentence:
        processed_lines.append(f"[{seconds_to_time(current_timestamp)}] {current_sentence.strip()}")

    return '\n'.join(processed_lines)

def time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

def seconds_to_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def process_episode(file_path):
    chunks = split_audio(file_path)
    full_transcript = ""
    
    for i, (chunk, start_time) in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        transcription = transcribe_audio(chunk)
        transcript = process_transcription(transcription, start_time)
        full_transcript += transcript + "\n\n"
        os.remove(chunk)  # Clean up the temporary chunk file

    output_file = f"transcript_full_{os.path.splitext(os.path.basename(file_path))[0]}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_transcript)
    
    print(f"Full transcription completed. Saved to {output_file}")

def main():
    file_path = "/Users/achille/Documents/Projets/chatgdiy/episodes/Yann.mp3"
    process_episode(file_path)

if __name__ == "__main__":
    main()
