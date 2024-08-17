"""
Local Audio Transcription Script (Whisper Model)

This script transcribes an MP3 audio file using the Whisper model from OpenAI, running locally on your machine.

Features:
- Transcribes audio using the Whisper model locally (no API calls)
- Saves the transcribed text to a file
- Includes logging for debugging purposes

Usage:
python transcribe.py

Before running:
1. Install required libraries:
   pip install whisper

2. Modify the audio_file variable in the main() function to point to your MP3 file.

The script will process the audio file locally and save the transcription as 'transcript.txt' in the same directory.
It also generates a 'transcribe_debug.log' file for debugging purposes.

Note: 
- This script uses the 'base' Whisper model running locally. You can change this in the script if needed.
- The Whisper model will be downloaded the first time you run the script, but subsequent runs will use the local copy.
- No internet connection is required after the initial model download.

For more detailed instructions or troubleshooting, refer to the README.md file in the project repository.
"""

import whisper
import os
import logging
import time

logging.basicConfig(filename='transcribe_debug.log', level=logging.DEBUG)

def transcribe_audio(audio_file_path):
    logging.debug(f"Starting transcription of {audio_file_path}")
    try:
        # Load the Whisper model
        logging.debug("Loading Whisper model")
        model = whisper.load_model("base")
        logging.debug("Whisper model loaded successfully")
        
        # Perform the transcription
        logging.debug("Beginning transcription")
        start_time = time.time()
        result = model.transcribe(audio_file_path)
        end_time = time.time()
        logging.debug(f"Transcription completed in {end_time - start_time:.2f} seconds")
        
        # Get the transcript
        transcript = result["text"]
        return transcript
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise

def save_transcript(transcript, output_file):
    logging.debug(f"Saving transcript to {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        logging.debug("Transcript saved successfully")
    except Exception as e:
        logging.error(f"Error saving transcript: {str(e)}")
        raise

def main():
    # Path to your audio file
    audio_file = "macron__extract.mp3"
    
    logging.debug(f"Starting main function with audio file: {audio_file}")
    
    # Check if the audio file exists
    if not os.path.exists(audio_file):
        error_msg = f"Error: Audio file not found at {audio_file}"
        print(error_msg)
        logging.error(error_msg)
        return
    
    print("Starting transcription...")
    try:
        transcript = transcribe_audio(audio_file)
        # Save the transcript
        output_file = "transcript.txt"
        save_transcript(transcript, output_file)
        print(f"Transcription completed. Saved to {output_file}")
        logging.debug("Main function completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
