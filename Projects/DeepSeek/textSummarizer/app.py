
#This app uses fastapi to create a web API for text summarization using DeepSeek AI.


from fastapi import FastAPI #fastapi is a web framework for building APIs
from pydantic import BaseModel #pydantic is a data validation library
import requests #requests is a library for making HTTP requests

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"

class SummarizeRequest(BaseModel):
    text: str


@app.post("/summarize/")
def summarize_text(req: SummarizeRequest):
    payload = {"model": "deepseek-r1", "prompt": f"Summarize:\n\n{req.text}", "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No summary generated.")
    except requests.RequestException as e:
        return {"error": f"Failed to reach Ollama at {OLLAMA_URL}: {e}"}

# Run with: uvicorn app:app --reload
