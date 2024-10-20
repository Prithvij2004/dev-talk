from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
import os
from transriber import transcribe_a, summarize_transcript
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class TranscriptionResult(BaseModel):
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
        # transcribe the audio file
        transcript = transcribe_a(file.file)
        # summarize the transcript
        # summary = summarize_transcript(transcript)
        return TranscriptionResult(summary=transcript["srt"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
