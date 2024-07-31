import os
from dotenv import load_dotenv
import openai
from pyannote.audio import Pipeline
import torch
import sys

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

from huggingface_hub import login
login(HUGGINGFACE_TOKEN)

from huggingface_hub import HfApi

api = HfApi()
try:
    user_info = api.whoami(token=HUGGINGFACE_TOKEN)
    print(f"Authenticated as: {user_info['name']}")
    print(f"Token permissions: {user_info['auth'].get('accessToken', {}).get('role', 'Unknown')}")
except Exception as e:
    print(f"Error verifying token: {str(e)}")
    sys.exit(1)

# Initialize the diarization pipeline
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.0", 
                                        use_auth_token=HUGGINGFACE_TOKEN)
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
except Exception as e:
    print(f"Error initializing pipeline: {str(e)}")
    print("Please make sure you have accepted the user conditions for 'pyannote/speaker-diarization' at https://huggingface.co/pyannote/speaker-diarization")
    print("Also, ensure your Hugging Face token has the necessary permissions.")
    sys.exit(1)

def diarize_audio(file_path):
    print(f"Diarizing {file_path}...")
    try:
        diarization = pipeline(file_path)
        return diarization
    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        return None

# ... (rest of your functions remain the same)

def process_episode(file_path, duration_minutes=10):
    print(f"Processing {file_path}...")
    
    diarization = diarize_audio(file_path)
    if diarization is None:
        print("Diarization failed. Exiting.")
        return

    transcription = transcribe_audio(file_path)
    
    processed_transcript = process_transcription(transcription, diarization)
    
    output_file = f"transcript_{duration_minutes}min_{os.path.splitext(os.path.basename(file_path))[0]}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(processed_transcript)
    
    print(f"Transcription with speaker diarization completed. Saved to {output_file}")

def main():
    file_path = "/Users/achille/Documents/Projets/chatgdiy/macron__extract.mp3"
    process_episode(file_path, duration_minutes=1)  # Process the extract

if __name__ == "__main__":
    main()
