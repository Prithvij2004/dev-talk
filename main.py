from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()

app = FastAPI()

ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
aai.settings.api_key = ASSEMBLY_API_KEY
config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best, summary_model=aai.SummarizationModel.informative, summarization=True, summary_type=aai.SummarizationType.bullets)

class TranscriptionResult(BaseModel):
    text: str
    summary: str

@app.post("/transcribe/", response_model=TranscriptionResult)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file uploaded by the client.

    Args:
    - file (UploadFile): The audio file uploaded by the client.

    Returns:
    - TranscriptionResult: The transcription result of the audio file.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Create a transcriber
        transcriber = aai.Transcriber(config=config)

        # Transcribe the audio file
        transcript = transcriber.transcribe(file.file)

        if transcript.status == "completed":
            return TranscriptionResult(text=transcript.text, summary=transcript.summary)
        else:
            raise HTTPException(status_code=500, detail="Transcription failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/items/")
async def create_item(item: Item):
    return item
