import os, uuid
import chromadb
import assemblyai as aai
import openai
import cohere
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(debug=True)

# Initialize AssemblyAI Client
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
aai_config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano, auto_chapters=True)
transcriber = aai.Transcriber(config=aai_config)


# Initialize Cohere Client
cohere_client = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))
client = openai.OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient()
movie_collection = chroma_client.create_collection(name="movie_summaries", get_or_create=True)


def extract_audio_summary(movie_file_path):
    """
    Extract audio summary using AssemblyAI
    """
    audio_file_path = extract_audio_from_video(movie_file_path, "temp/temp_audio.wav")
    print("Starting transcription...")
    # Upload audio file to AssemblyAI
    try:
        transcript = transcriber.transcribe(audio_file_path, aai_config)
    
        print("Transcription complete!")
        
        # Process and store key points in ChromaDB
        for point in transcript.chapters:
            # Use Cohere model to encode the summary
            response = cohere_client.embed(model=os.getenv('COHERE_MODEL'),
                                        texts=[point.summary],
                                        input_type="search_document",
                                        embedding_types=["float"],
                                        truncate='NONE')
            embedding = response.embeddings.float[0]
            logging.info(embedding)
            movie_collection.add(
                embeddings=[embedding],
                documents=[point.summary],
                metadatas=[{
                    'start_time': point.start,
                    'end_time': point.end,
                    'headline': point.headline,
                    'transcript_id': transcript.id
                }],
                ids=[str(uuid.uuid4())]
            )

        # Remove temporary audio file
        os.remove(audio_file_path)
        
        return transcript
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def query_movie_summary(query):
    """
    Query movie summary using vector similarity search and Cohere
    """
    try:
        # Convert query to embedding
        response = cohere_client.embed(model=os.getenv('COHERE_MODEL'),
                                        texts=[query],
                                        input_type='search_query',
                                        embedding_types=['float'],
                                        truncate='NONE')
        query_embedding = response.embeddings.float[0]
        
        print("Embedding complete!")
        # Perform similarity search
        results = movie_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        if len(results["documents"]) == 0: 
            return {"llm_response": "No results found"}

        
        print("Similarity search complete!")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Provide a concise and accurate response based only on the information given in the movie summary points. Also output the timestamps of the key points if provided."},
            {"role": "user", "content": f"{results['documents']}"},
            {"role": "user", "content": f"{results['metadatas']}"},
            {"role": "user", "content": f"Please answer this query: {query}"}
        ]


        response = client.chat.completions.create(
            model=os.getenv('SAMBANOVA_MODEL'),
            messages=messages,
            temperature=0.5,
            top_p=0.1
        )
        print("Content generation complete!")
        return {
            'llm_response': response.choices[0].message.content.strip(),
            'query_results': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_audio_from_video(video_path, audio_output_path):
    """
    Extracts audio from a video file and saves it as an audio file.

    :param video_path: Path to the video file.
    :param audio_output_path: Path where the extracted audio will be saved.
    """

    print("Saving audio...", audio_output_path)
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio.write_audiofile(audio_output_path)

    return audio_output_path




@app.post('/analyze_movie')
async def analyze_movie(movie_file: UploadFile = File(...)):
    """
    Endpoint to analyze a movie file
    """
    try:
        movie_file_path = f"temp/temp_{movie_file.filename}"
        with open(movie_file_path, 'wb') as buffer:
            buffer.write(await movie_file.read())
        print("Starting movie analysis...")

        transcript = extract_audio_summary(movie_file_path)
        os.remove(movie_file_path)
        print("Movie analysis complete!")

        return JSONResponse(content={
            "message": "Movie analyzed successfully",
            "transcript": [
                {
                    "text": point.summary,
                    "headline": point.headline,
                    "start_time": point.start,
                    "end_time": point.end
                } for point in transcript.chapters
            ]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post('/query_movie')
async def query_movie(query: str | None = None, audio_file: UploadFile | None = None):
    """
    Endpoint to query movie summary
    """
    try:
        if audio_file:
            audio_file_path = f"temp/temp_{audio_file.filename}"
            with open(audio_file_path, 'wb') as buffer:
                buffer.write(await audio_file.read())
            transcript = transcriber.transcribe(audio_file_path)
            os.remove(audio_file_path)
            results = query_movie_summary(transcript.text)
            return JSONResponse(content=results)

        elif query:
            results = query_movie_summary(query)
            return JSONResponse(content=results)

        return JSONResponse(content={"message": "No audio file or query provided"}, status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/collection")
async def get_collection():
    return JSONResponse(content=movie_collection.get())

@app.delete("/collection")
async def clear_collection():
    global movie_collection
    try:
        # Delete existing collection if it exists
        chroma_client.delete_collection(name="movie_collection")
    except ValueError:
        # Collection doesn't exist, that's fine
        pass
    
    # Create new collection
    movie_collection = chroma_client.create_collection(name="movie_collection")
    return {"message": "Collection cleared"}