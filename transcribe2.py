import os
from dotenv import load_dotenv
import openai
from pydub import AudioSegment

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def trim_audio(file_path, duration_ms=600000):  # Default to 10 minutes (600000 ms)
    audio = AudioSegment.from_mp3(file_path)
    trimmed_audio = audio[:duration_ms]
    temp_file = "temp_trimmed_audio.mp3"
    trimmed_audio.export(temp_file, format="mp3")
    return temp_file

def transcribe_audio(file_path):
    print(f"Transcribing {file_path}...")
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="srt"
        )
    return process_transcription(transcription)

import re

def process_transcription(transcription, max_interval=30):  # max_interval in seconds
    blocks = transcription.split('\n\n')
    processed_lines = []
    current_sentence = ""
    current_timestamp = ""
    last_timestamp = 0

    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            time_range = lines[1].split(' --> ')[0]
            current_time = time_to_seconds(time_range)
            text = ' '.join(lines[2:])
            
            if not current_timestamp:
                current_timestamp = time_range.replace(',', '.')
                last_timestamp = current_time

            current_sentence += text + " "
            
            # Check if we need to add a timestamp (due to max_interval or sentence end)
            if current_time - last_timestamp >= max_interval or re.search(r'[.!?]\s*$', text):
                processed_lines.append(f"[{current_timestamp}] {current_sentence.strip()}")
                current_sentence = ""
                current_timestamp = ""
                last_timestamp = current_time

    # Add any remaining text
    if current_sentence:
        processed_lines.append(f"[{current_timestamp}] {current_sentence.strip()}")

    return '\n'.join(processed_lines)

def time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

def process_episode(file_path, duration_minutes=10):
    duration_ms = duration_minutes * 60 * 1000  # Convert minutes to milliseconds
    temp_file = trim_audio(file_path, duration_ms)
    
    transcript = transcribe_audio(temp_file)
    
    output_file = f"transcript_{duration_minutes}min_{os.path.splitext(os.path.basename(file_path))[0]}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    os.remove(temp_file)  # Clean up the temporary file
    print(f"Transcription of first {duration_minutes} minutes completed. Saved to {output_file}")

def main():
    file_path = "/Users/achille/Documents/Projets/chatgdiy/macron__extract.mp3"
    process_episode(file_path, duration_minutes=20)  # Process first 20 minutes

if __name__ == "__main__":
    main()
