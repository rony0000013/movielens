# 🎬 MovieLens 📸

## Overview

This is a sophisticated web application that uses AI technologies to analyze movies, extract key points, and provide intelligent insights using Retrieval Augmented Generation (RAG).

## Features

- Movie file upload and audio extraction
- AssemblyAI-powered transcription and key point extraction
- ChromaDB vector storage for semantic search
- AI-powered query response system using SambaNova's Llama model

## Prerequisites

- Python 3.11+
- API Keys:
  - AssemblyAI API Key
  - Google API Key (for Gemini)
  - SambaNova API Key
  - Cohere API Key

## Setup Instructions

1. Clone the repository

```bash
git clone <repository_url>
cd movielens
```

2. Create a virtual environment

```bash
uv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies

```bash
uv add -r requirements.txt
```

4. Configure API Keys

- Create a `.env` file in the root directory
- Add your API keys:

  .env file
  ```[.env]
  ASSEMBLYAI_API_KEY=<your_assemblyai_api_key>
  SAMBANOVA_API_KEY=<your_sambanova_api_key>
  GOOGLE_API_KEY=<your_google_api_key>
  COHERE_API_KEY=<your_cohere_api_key>
  SAMBANOVA_MODEL="Meta-Llama-3.1-70B-Instruct"
  COHERE_MODEL="embed-multilingual-v3.0"
  ```

  .steamlit/secrets.toml file
  ```[.steamlit/secrets.toml]
  SERVER_URL="http://localhost:8000"
  ```

5. Run the application

```bash
uv run fastapi run main.py
```

## Usage

1. Upload a movie file
2. The application will process the audio and extract key points
3. Use the query interface and voice to ask questions about the movie

## Technologies Used

- FastAPI (Web Framework)
- AssemblyAI (Audio Transcription)
- ChromaDB (Vector Database)
- Google Gemini (LLM)
- Cohere (Embeddings)
- SambaNova (LLM)

## Future Improvements

- Enhanced user interface frontend with real-time processing updates
- Batch processing for large movie files
- More robust error handling
- Caching mechanisms for faster responses
