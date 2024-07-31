# ChatGDIY

ChatGDIY is an application aimed at allowing users to interact with content from the French podcast "Generation Do It Yourself". The goal is to enable users to ask questions to the host (Matthieu) or guests and receive sourced answers.

## Current State

The project has made significant progress in transcribing and processing podcast episodes using various AI technologies.

## Implemented Features

* Multiple transcription methods using both local and API-based OpenAI Whisper models
* Speaker diarization to identify different speakers in the audio
* Timeline extraction from Spotify podcast descriptions
* Scripts to process audio files and generate text transcripts with timestamps and speaker labels
* Advanced speaker clustering and smoothing techniques
* Logging system for debugging and progress tracking

## Scripts Overview

1. `transcribe.py`: Local audio transcription using Whisper model
2. `transcribe2.py`: Enhanced transcription using OpenAI's Whisper API with improved timestamp accuracy
3. `transcribe3.py`: Transcription with basic speaker diarization
4. `transcribe4.py`: Advanced transcription with improved speaker diarization, smoothing, and clustering
5. `spotifyfindshow.py`: Utility to find Spotify show IDs
6. `spotifygetepisode.py`: Extract episode details and timeline from Spotify podcasts

## Challenges and Limitations

1. **Processing Time**: While improved, transcription and diarization of full episodes (1 to 4 hours each) still require significant processing time.
2. **Output Quality**:
   * Speaker diarization accuracy may vary depending on audio quality and number of speakers.
   * Some transcription errors may still be present in the generated text.
3. **Scalability**: Processing the entire podcast archive (400+ episodes) remains time-consuming.

## Next Steps

1. Optimize processing speed for full-length episodes.
2. Implement error handling and recovery mechanisms for long-running processes.
3. Develop a system to manage and update transcriptions incrementally as new episodes are released.
4. Create a data storage and indexing system for efficient retrieval of transcribed content.
5. Implement a Retrieval-Augmented Generation (RAG) feature to use the transcripts as a knowledge base for the chatbot.
6. Begin development of the chatbot interface for user interactions.
7. Implement post-processing techniques to further clean up transcripts and improve readability.

## Technical Stack

* Python 3.12
* OpenAI Whisper for speech-to-text conversion (both local and API versions)
* pyannote.audio for speaker diarization
* FFmpeg for audio processing
* Spotify API for podcast metadata retrieval

### Python Libraries Used

* openai
* pydub
* torch
* pyannote.audio
* scipy
* scikit-learn
* python-dotenv
* requests
* aiohttp
* numpy
* argparse
* logging

## Setup and Usage

1. Clone the repository
2. Install required dependencies:

pip install -r requirements.txt

3. Set up environment variables in a `.env` file:

OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_SHOW_ID=your_spotify_show_id

4. Run the desired script, e.g.:

python transcribe4.py

For detailed usage instructions for each script, refer to the docstrings at the beginning of each file.

## Contributing

Contributions to ChatGDIY are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.