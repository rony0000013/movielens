# Movie Analysis RAG Application

## Overview
This is a sophisticated web application that uses AI technologies to analyze movies, extract key points, and provide intelligent insights using Retrieval Augmented Generation (RAG).

## Features
- Movie file upload and audio extraction
- AssemblyAI-powered transcription and key point extraction
- ChromaDB vector storage for semantic search
- AI-powered query response system using Google's Gemini model

## Prerequisites
- Python 3.8+
- API Keys:
  * AssemblyAI API Key
  * Google API Key (for Gemini)

## Setup Instructions

1. Clone the repository
```bash
git clone <repository_url>
cd movie-analysis-rag-app
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure API Keys
- Create a `.env` file in the root directory
- Add your API keys:
  ```
  ASSEMBLYAI_API_KEY=your_assemblyai_api_key
  GOOGLE_API_KEY=your_gemini_api_key
  ```

5. Run the application
```bash
flask run
```

## Usage
1. Upload a movie file
2. The application will process the audio and extract key points
3. Use the query interface to ask questions about the movie

## Technologies Used
- Flask (Web Framework)
- AssemblyAI (Audio Transcription)
- ChromaDB (Vector Database)
- Google Gemini (LLM)
- Sentence Transformers (Embeddings)

## Limitations
- Supports video file uploads
- Requires stable internet connection
- API rate limits may apply

## Future Improvements
- Direct integration with SambaNova Llama 3.1 405B model
- Enhanced frontend with real-time processing updates
- Batch processing for large movie files
- More robust error handling
- Caching mechanisms for faster responses
