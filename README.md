# ChatGDIY

ChatGDIY is an application aimed at allowing users to interact with content from the French podcast "Generation Do It Yourself". The goal is to enable users to ask questions to the host (Matthieu) or guests and receive sourced answers.

## Current State

The project is in its initial stages, focusing on transcribing podcast episodes using AI technology.

### Implemented Features

- Basic transcription functionality using OpenAI's Whisper model
- Script to process audio files and generate text transcripts
- Logging system for debugging and progress tracking

### Challenges and Limitations

1. **Processing Time**: Transcription of a 1-minute audio file takes 5-10 minutes, making it impractical for full episodes (1.5 hours each) and the entire podcast archive (400+ episodes).

2. **Output Quality**:
   - The current output is plain text without speaker identification or timestamps.
   - Some transcription errors are present in the generated text.

3. **Scalability**: The current approach is not scalable for processing the entire podcast archive in a reasonable timeframe.

### Next Steps

1. Implement speaker diarization and timestamp extraction.
2. Explore methods to improve transcription accuracy.
3. Develop strategies for efficient processing of the full podcast archive:
   - Consider cloud-based or distributed processing solutions.
   - Investigate parallel processing techniques.
   - Explore incremental processing of new episodes.
4. Plan data storage, indexing, and retrieval systems for the chatbot functionality.
5. Research and implement post-processing techniques to clean up transcripts.

## Technical Stack

- Python 3.12
- OpenAI Whisper for speech-to-text conversion
- FFmpeg for audio processing