import whisper
import os

def transcribe_audio(audio_file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Perform the transcription
    result = model.transcribe(audio_file_path)

    # Get the transcript
    transcript = result["text"]

    return transcript

def save_transcript(transcript, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)

def main():
    # Path to your audio file
    audio_file = "macron_episode.mp3"
    
    # Check if the audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        return

    print("Starting transcription...")
    transcript = transcribe_audio(audio_file)

    # Save the transcript
    output_file = "transcript.txt"
    save_transcript(transcript, output_file)
    print(f"Transcription completed. Saved to {output_file}")

if __name__ == "__main__":
    main()
