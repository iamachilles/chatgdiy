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
