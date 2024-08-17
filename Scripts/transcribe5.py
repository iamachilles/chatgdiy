"""
Advanced Audio Transcription and Speaker Diarization Script (Version 5)

This script transcribes an audio file using the AssemblyAI API, performing both transcription and speaker diarization. It offers significant improvements and simplifications over previous versions.

Features:
- Transcribes audio files using AssemblyAI's advanced speech recognition technology
- Performs speaker diarization to distinguish between different speakers
- Supports multiple languages (configurable via command-line argument)
- Formats the transcript with timestamps and speaker labels
- Saves the transcribed and diarized text to a file
- Provides detailed logging and progress reporting

Improvements in Version 5:
- Utilizes AssemblyAI's API for both transcription and diarization, simplifying the process
- Eliminates the need for audio file splitting and separate diarization steps
- Improves transcript formatting for better readability
- Adds language support through command-line arguments
- Enhances error handling and logging

Usage:
python transcribe5.py /path/to/your/audio_file.mp3 [-l LANGUAGE_CODE] [-v]

Options:
  -l LANGUAGE_CODE, --language LANGUAGE_CODE    Specify the language code (e.g., 'fr' for French, 'en' for English)
  -v, --verbose                                 Increase output verbosity

Before running:
1. Ensure you have a .env file with your AssemblyAI API key:
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key

2. Install required libraries:
   pip install python-dotenv assemblyai tqdm argparse

The script will process the audio file, perform transcription and diarization, and save the result as a text file in the same directory.

Note: 
- This script requires an active internet connection to access the AssemblyAI API.
- Processing time may vary depending on the length of the audio file and the API's current load.

For more detailed instructions or troubleshooting, refer to the README.md file in the project repository.
"""


import os
import logging
from dotenv import load_dotenv
import assemblyai as aai
import argparse
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AssemblyAI
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY", "80c94a33424344b99a5228ba3b36bac7")

def transcribe_and_diarize(file_path, language_code="fr"):
    logger.info(f"Transcribing and diarizing: {file_path}")
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        language_code=language_code
    )
    
    transcript = transcriber.transcribe(file_path, config=config)
    
    return transcript

def format_transcript(transcript, max_words_per_segment=30):
    formatted_transcript = ""
    current_speaker = None
    current_segment = ""
    current_start_time = None

    for utterance in transcript.utterances:
        words = utterance.text.split()
        start_time = utterance.start

        if current_speaker != utterance.speaker or len(current_segment.split()) + len(words) > max_words_per_segment:
            if current_segment:
                formatted_transcript += f"[{format_time(current_start_time)}] Speaker {current_speaker}: {current_segment.strip()}\n"
            current_speaker = utterance.speaker
            current_segment = ""
            current_start_time = start_time

        current_segment += " " + utterance.text

    # Add the last segment
    if current_segment:
        formatted_transcript += f"[{format_time(current_start_time)}] Speaker {current_speaker}: {current_segment.strip()}\n"

    return formatted_transcript

def format_time(milliseconds):
    seconds = int(milliseconds / 1000)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_episode(file_path, language_code):
    logger.info(f"Processing episode: {file_path}")
    
    try:
        transcript = transcribe_and_diarize(file_path, language_code)
        formatted_transcript = format_transcript(transcript)
        
        output_file = f"transcript_full_{os.path.splitext(os.path.basename(file_path))[0]}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)
        
        logger.info(f"Full transcription completed. Saved to {output_file}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        logger.exception("Exception details:")

def main():
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio files using AssemblyAI")
    parser.add_argument("file_path", help="Path to the audio file")
    parser.add_argument("-l", "--language", default="fr", help="Language code (e.g., 'fr' for French, 'en' for English)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting audio processing")
    try:
        process_episode(args.file_path, args.language)
        logger.info("Audio processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()