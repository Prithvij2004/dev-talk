import assemblyai as aai
from together import Together
from dotenv import load_dotenv
import os

from typing import cast

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

def transcribe_a(file):
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file)
    srt = transcript.export_subtitles_srt(chars_per_caption=32)
    result = {
        "transcript": transcript.text,
        "srt": srt,
    }
    return result

def summarize_transcript(transcript):
    client = Together(api_key=together_api_key)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        messages=[
            {"role": "system",
                "content": """
                You are a note maker for software developers and engineers. Store all the important decisions in concise way.
                No need for a long sentance giving too much context. Just a short summary of the important decisions.
                Make the notes first person, this is a personal note or memory book that stores the users previous thoughts and decisions.
                If their is something about decisions, always add those with detailed reason they are mentioning in the text in concise way.
                Only summarise based of on the text submitted and don't assume things.

                Try make the thought process more clear. Like how we reached to the final steps.
                Example: Starting from blah... till reaching blah..
            """},
            {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript}"},
        ],
        temperature=0.5,
        repetition_penalty=1.5,
    )
    summary = response.choices[0].message.content #type: ignore
    return summary
