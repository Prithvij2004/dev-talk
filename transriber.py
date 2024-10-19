import assemblyai as aai
from together import Together
from dotenv import load_dotenv
import os

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

def transcribe_a(file):
    config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file)
    return transcript

def summarize_transcript(transcript):
    client = Together(api_key=together_api_key)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        messages=[
            {"role": "system",
                "content": """You are a helpful travel guide.

                If their is something about decisions, always add those with detailed reason they are mentioning in the text in concise way. Only summarise based of on the text submitted and don't assume things.

                Try make the thought process more clear. Like how we reached to the final steps.
                Example: Starting from blah... till reaching blah..

                The oupput should in markdown format and don't include escpace sequence backslash n component in the ouput.
            """},
            {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript}"},
        ],
        temperature=0.5,
        repetition_penalty=1.5,
    )
    summary = response.choices[0].message.content
    return summary
