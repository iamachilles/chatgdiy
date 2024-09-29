# ChatGDIY

ChatGDIY is an ambitious project aimed at allowing users to interact with content from the French podcast "Generation Do It Yourself". The ultimate goal is to enable users to ask questions to the host (Matthieu) or guests and receive sourced answers.

## Project Evolution

### Initial Development
The project initially focused on transcribing and processing podcast episodes using various AI technologies, with significant progress in transcription accuracy and processing efficiency.

### Current State and Latest Developments
Building on our initial progress, we've now implemented a crucial intermediate step: a web-based audio transcription tool. This new feature serves as a practical stepping stone towards our larger goal of podcast content interaction.

#### New Feature: Audio Transcription Web App
We've developed and deployed a web application that allows users to transcribe audio files using either OpenAI's Whisper API or AssemblyAI's transcription service.

**Web App URL:** [https://transcribe-audio-0r2y.onrender.com/](https://transcribe-audio-0r2y.onrender.com/)

##### Key Components:
1. `transcribe6.py`: A Flask-based backend that handles file uploads and transcription requests.
2. `index.html`: A user-friendly frontend interface for file upload and transcription.
3. Render Deployment: The application is hosted on Render, providing easy access to users.

## Implemented Features

* Web-based audio transcription using OpenAI and AssemblyAI services (New)
* File upload functionality with support for various audio formats (New)
* Error handling and user feedback mechanisms (New)
* Secure handling of API keys and temporary file storage (New)
* Multiple transcription methods using both local and API-based models
* Speaker diarization to identify different speakers in the audio
* Timeline extraction from Spotify podcast descriptions
* Scripts to process audio files and generate text transcripts with timestamps and speaker labels
* Advanced speaker clustering and smoothing techniques
* Logging system for debugging and progress tracking
* Integration with AssemblyAI for improved transcription and diarization

## Scripts Overview

1. `transcribe.py`: Local audio transcription using Whisper model
2. `transcribe2.py`: Enhanced transcription using OpenAI's Whisper API with improved timestamp accuracy
3. `transcribe3.py`: Transcription with basic speaker diarization
4. `transcribe4.py`: Advanced transcription with improved speaker diarization, smoothing, and clustering using OpenAI's Whisper API and pyannote.audio
5. `transcribe5.py`: Streamlined transcription and diarization using AssemblyAI's API
6. `transcribe6.py`: New Flask-based web application for audio transcription (Latest addition)
7. `spotifyfindshow.py`: Utility to find Spotify show IDs
8. `spotifygetepisode.py`: Extract episode details and timeline from Spotify podcasts

## Challenges and Limitations

1. **Processing Time**: While improved with `transcribe5` and `transcribe6`, processing full episodes (1 to 4 hours each) still requires significant time.
2. **Output Quality**:
   * Speaker diarization accuracy may vary depending on audio quality and number of speakers.
   * Some transcription errors may still be present in the generated text.
3. **Scalability**: Processing the entire podcast archive (400+ episodes) remains time-consuming.
4. **API Dependency**: The new web app relies on external API services, which may have usage limits or costs associated with high-volume processing.

## Next Steps

1. Integrate the transcription tool with podcast episode processing
2. Implement a system to manage and update transcriptions for podcast episodes
3. Develop a data storage and indexing system for efficient retrieval of transcribed content
4. Create a Retrieval-Augmented Generation (RAG) feature to use the transcripts as a knowledge base
5. Begin development of the chatbot interface for user interactions with podcast content
6. Implement post-processing techniques to further clean up transcripts and improve readability
7. Explore ways to combine the strengths of different transcription methods for optimal results

## Technical Stack

* Python 3.12
* Flask for web application framework
* OpenAI Whisper API for speech-to-text conversion
* AssemblyAI API for alternative speech-to-text conversion
* pyannote.audio for speaker diarization
* FFmpeg for audio processing
* Spotify API for podcast metadata retrieval
* Render for web application hosting
* HTML/CSS/JavaScript for frontend interface

## Python Libraries Used

* flask
* flask-cors
* openai
* assemblyai
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
* tqdm

## Getting Started with the Web App

To use the transcription tool, visit [https://transcribe-audio-0r2y.onrender.com/](https://transcribe-audio-0r2y.onrender.com/) and follow these steps:

1. Upload an audio file (supported formats: MP3, WAV, OGG, M4A, etc.)
2. Choose between OpenAI and AssemblyAI for transcription
3. Enter your API key for the selected service
4. Click "Transcribe" and wait for the results

## Setup and Usage for Development

1. Clone the repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key
   HUGGINGFACE_TOKEN=your_huggingface_token
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   SPOTIFY_SHOW_ID=your_spotify_show_id
   ```
4. Run the desired script, e.g.:
   ```
   python transcribe6.py
   ```

For detailed usage instructions for each script, refer to the docstrings at the beginning of each file.

## Contributing

Contributions to ChatGDIY are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.