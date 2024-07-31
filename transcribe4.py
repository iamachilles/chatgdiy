"""
Advanced Audio Transcription and Speaker Diarization Script (Version 4)

This script transcribes an MP3 audio file using OpenAI's Whisper API and performs advanced speaker diarization using pyannote.audio, with significant improvements over previous versions.

Features:
- Splits long audio files into manageable chunks
- Transcribes audio using OpenAI's Whisper API
- Performs advanced speaker diarization with smoothing and clustering
- Processes the transcription to include timestamps and clustered speaker labels
- Saves the transcribed and diarized text to a file

Improvements over Version 3:
- Implements improved diarization with smoothing to reduce rapid speaker changes
- Adds speaker clustering to assign consistent labels across the entire audio
- Enhances the integration of transcription and diarization results
- Improves error handling and provides more detailed logging

Usage:
python transcribe4.py

Before running:
1. Ensure you have a .env file with your OpenAI API key and Hugging Face access token:
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_TOKEN=your_huggingface_token

2. Install required libraries:
   pip install python-dotenv openai pydub torch pyannote.audio scipy scikit-learn

3. Modify the file_path variable in the main() function to point to your MP3 file.

The script will process the audio file, perform transcription and advanced diarization, and save the result as a text file in the same directory.

Note: 
- This script requires an active internet connection to access the OpenAI API and download the diarization model.
- The pyannote.audio library may require additional setup. Refer to its documentation for details.
- This script uses more advanced techniques and may take longer to process audio files compared to previous versions.

For more detailed instructions or troubleshooting, refer to the README.md file in the project repository.
"""

import os
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
import math
import torch
import sys
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio import Audio

import numpy as np
import functools
from scipy.signal import medfilt
from sklearn.cluster import KMeans

# Patch numpy.NAN
np.NAN = np.nan

# Patch the reconstruct method
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

original_reconstruct = SpeakerDiarization.reconstruct

@functools.wraps(original_reconstruct)
def patched_reconstruct(self, *args, **kwargs):
    global np
    original_NAN = np.NAN
    np.NAN = np.nan
    try:
        return original_reconstruct(self, *args, **kwargs)
    finally:
        np.NAN = original_NAN

SpeakerDiarization.reconstruct = patched_reconstruct

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize pyannote pipeline
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token="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")
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    sys.exit(1)

audio = Audio()

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
    
    # Get the duration of the audio file
    audio = AudioSegment.from_mp3(file_path)
    audio_duration = len(audio) / 1000  # Duration in seconds
    
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="srt"
        )
    
    # Check if there's a gap at the beginning
    first_timestamp = time_to_seconds(transcription.split('\n')[1].split(' --> ')[0])
    if first_timestamp > 1:  # If the first transcription starts after 1 second
        # Add a placeholder line at the beginning
        placeholder = f"1\n00:00:00,000 --> {seconds_to_time(first_timestamp)}\n[Beginning of audio]\n\n"
        transcription = placeholder + transcription
    
    print(f"Raw transcription: {transcription[:100]}...")
    return transcription

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

            if text == "[Beginning of audio]":
                processed_lines.append(f"[{seconds_to_time(current_timestamp)}] {text}")
                current_sentence = ""
                current_timestamp = None
                continue

            current_sentence += text + " "
            
            if current_time - last_timestamp >= max_interval or text.strip().endswith(('.', '!', '?')):
                processed_lines.append(f"[{seconds_to_time(current_timestamp)}] {current_sentence.strip()}")
                current_sentence = ""
                current_timestamp = None
                last_timestamp = current_time

    if current_sentence:
        processed_lines.append(f"[{seconds_to_time(current_timestamp)}] {current_sentence.strip()}")

    return '\n'.join(processed_lines)

def time_to_seconds(time_str):
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))
    except ValueError:
        print(f"Invalid time format: {time_str}")
        return 0
    
def seconds_to_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

from pyannote.audio import Pipeline

def diarize_audio(file_path, start_time=0, duration=None):
    print(f"Diarizing {file_path}...")
    
    # Use the latest pre-trained pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token="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")
    
    if duration:
        excerpt = Segment(start=start_time, end=start_time + duration)
        waveform, sample_rate = audio.crop(file_path, excerpt)
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    else:
        diarization = pipeline(file_path)
    
    return diarization

def integrate_diarization(transcript, diarization, chunk_start_time, min_turn_duration=1.0):
    diarized_lines = []
    transcript_lines = transcript.split('\n')
    
    # Smooth diarization results
    smoothed_diarization = smooth_diarization(diarization, min_turn_duration)
    
    # Cluster speakers
    speaker_clusters = cluster_speakers(smoothed_diarization)
    
    for turn, _, speaker in smoothed_diarization.itertracks(yield_label=True):
        start_time = turn.start + chunk_start_time
        end_time = turn.end + chunk_start_time
        cluster_id = speaker_clusters[speaker]
        
        matching_lines = []
        for line in transcript_lines:
            line_time = time_to_seconds(line[1:13])  # Extract timestamp from line
            if start_time <= line_time < end_time:
                matching_lines.append(line)
        
        if not matching_lines:
            continue
        
        for line in matching_lines:
            diarized_lines.append(f"Speaker {cluster_id}: {line}")
    
    if not diarized_lines:
        print("No diarized lines produced. Returning original transcript.")
        return transcript
    
    return "\n".join(diarized_lines)

def smooth_diarization(diarization, min_turn_duration):
    # Convert diarization to a list of (start, end, speaker) tuples
    turns = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    
    # Sort turns by start time
    turns.sort(key=lambda x: x[0])
    
    # Merge short turns
    smoothed_turns = []
    current_turn = turns[0]
    for turn in turns[1:]:
        if turn[0] - current_turn[1] < min_turn_duration and turn[2] == current_turn[2]:
            current_turn = (current_turn[0], turn[1], current_turn[2])
        else:
            smoothed_turns.append(current_turn)
            current_turn = turn
    smoothed_turns.append(current_turn)
    
    # Create a new Annotation object with smoothed turns
    smoothed_diarization = diarization.empty()
    for start, end, speaker in smoothed_turns:
        smoothed_diarization[Segment(start, end)] = speaker
    
    return smoothed_diarization

def cluster_speakers(diarization, num_clusters=2):
    # Extract speaker embeddings (this is a placeholder, you'd need to implement this)
    embeddings = extract_speaker_embeddings(diarization)
    
    # Perform clustering (e.g., using K-means)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(embeddings)
    
    # Create a mapping from original speaker labels to cluster IDs
    unique_speakers = list(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
    return {speaker: f"SPEAKER_{clusters[i]:02d}" for i, speaker in enumerate(unique_speakers)}

def extract_speaker_embeddings(diarization):
    # This is a placeholder function. You'd need to implement this based on the
    # specific features of your audio and diarization output.
    # It should return a list of feature vectors, one for each unique speaker in the diarization.
    # For now, we'll return random embeddings
    unique_speakers = list(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
    return np.random.rand(len(unique_speakers), 10)  # 10-dimensional random embeddings

def process_episode(file_path):
    chunks = split_audio(file_path)
    full_transcript = ""
    
    for i, (chunk, start_time) in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        transcription = transcribe_audio(chunk)
        print(f"Transcription length: {len(transcription)}")
        transcript = process_transcription(transcription, start_time)
        print(f"Processed transcript length: {len(transcript)}")
        
        chunk_duration = AudioSegment.from_mp3(chunk).duration_seconds
        diarization = diarize_audio(chunk, start_time=0, duration=chunk_duration)
        
        if diarization is not None:
            print("Diarization result:")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"Speaker {speaker}: {turn.start:.2f} - {turn.end:.2f}")
            diarized_transcript = integrate_diarization(transcript, diarization, start_time)
            print(f"Diarized transcript length: {len(diarized_transcript)}")
        else:
            diarized_transcript = transcript
            print("Diarization failed, using original transcript")
        
        full_transcript += diarized_transcript + "\n\n"
        os.remove(chunk)

    print(f"Full transcript length: {len(full_transcript)}")
    output_file = f"transcript_full_{os.path.splitext(os.path.basename(file_path))[0]}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_transcript)
    
    print(f"Full transcription with diarization completed. Saved to {output_file}")
    
def main():
    file_path = "/Users/achille/Documents/Projets/chatgdiy/macron_episode__cut.mp3"
    process_episode(file_path)

if __name__ == "__main__":
    main()