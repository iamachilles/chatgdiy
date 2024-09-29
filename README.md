# ChatGDIY

ChatGDIY is an application aimed at allowing users to interact with content from the French podcast "Generation Do It Yourself". The goal is to enable users to ask questions to the host (Matthieu) or guests and receive sourced answers.

## Current State

The project has made significant progress in transcribing and processing podcast episodes using various AI technologies, with recent improvements in transcription accuracy and processing efficiency.

## Implemented Features

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
6. `spotifyfindshow.py`: Utility to find Spotify show IDs
7. `spotifygetepisode.py`: Extract episode details and timeline from Spotify podcasts

### Transition from transcribe4 to transcribe5

The project has evolved from `transcribe4` to `transcribe5` to address several challenges and improve overall performance:

1. **Simplified Processing**: `transcribe5` uses AssemblyAI's API for both transcription and diarization, eliminating the need for separate processing steps and reducing complexity.
2. **Improved Efficiency**: By leveraging AssemblyAI's advanced technology, `transcribe5` offers faster processing times for full-length episodes.
3. **Enhanced Accuracy**: AssemblyAI's state-of-the-art models provide improved transcription accuracy and more reliable speaker diarization.
4. **Multilingual Support**: `transcribe5` adds easy language configuration, making it more versatile for potential future use with non-French podcasts.
5. **Reduced Dependencies**: The new version reduces the number of required libraries and simplifies the setup process.

## Challenges and Limitations

1. **Processing Time**: While improved with `transcribe5`, processing full episodes (1 to 4 hours each) still requires significant time.
2. **Output Quality**:
   * Speaker diarization accuracy may vary depending on audio quality and number of speakers.
   * Some transcription errors may still be present in the generated text.
3. **Scalability**: Processing the entire podcast archive (400+ episodes) remains time-consuming.
4. **API Dependency**: `transcribe5` relies on external API services, which may have usage limits or costs associated with high-volume processing.

## Next Steps

1. Optimize processing speed for full-length episodes.
2. Implement error handling and recovery mechanisms for long-running processes.
3. Develop a system to manage and update transcriptions incrementally as new episodes are released.
4. Create a data storage and indexing system for efficient retrieval of transcribed content.
5. Implement a Retrieval-Augmented Generation (RAG) feature to use the transcripts as a knowledge base for the chatbot.
6. Begin development of the chatbot interface for user interactions.
7. Implement post-processing techniques to further clean up transcripts and improve readability.
8. Explore ways to combine the strengths of different transcription methods for optimal results.

## Technical Stack

* Python 3.12
* AssemblyAI API for speech-to-text conversion and speaker diarization
* OpenAI Whisper for alternative speech-to-text conversion (both local and API versions)
* pyannote.audio for alternative speaker diarization
* FFmpeg for audio processing
* Spotify API for podcast metadata retrieval

## Python Libraries Used

* assemblyai
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
* tqdm

## Setup and Usage

1. Clone the repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_TOKEN=your_huggingface_token
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   SPOTIFY_SHOW_ID=your_spotify_show_id
   ASSEMBLYAI_API_KEY=your_assembly_api_key
   ```
4. Run the desired script, e.g.:
   ```
   python transcribe5.py /path/to/audio_file.mp3 -l fr
   ```

For detailed usage instructions for each script, refer to the docstrings at the beginning of each file.

## Contributing

Contributions to ChatGDIY are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
