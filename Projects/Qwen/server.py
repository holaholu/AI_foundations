#Qwen 2.5 is a state-of-the-art AI model created by Alibaba Cloud. It is a large language model that has been trained on a vast corpus of text data, including books, websites, and other sources of information. Qwen 2.5 is designed to understand and generate human-like text, making it a powerful tool for a wide range of applications, such as natural language processing, machine translation, and text generation.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel #<-- for data validation
import ollama 

# Initialize FastAPI app
app = FastAPI()

#Enable CORS to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], #<-- allow all HTTP methods
    allow_headers=["*"], #<-- allow all headers
)


class ChatRequest(BaseModel): #<-- for data validation
    message: str #<-- message to be sent to the AI model

@app.post("/chat") #<-- endpoint for chat
async def chat(request: ChatRequest): #<-- request body of class ChatRequest
    try:
        print(request.message)
        response = ollama.chat(model="qwen2.5", messages=[{"role": "user", "content": request.message}]) #<-- send message to AI model
        return {"response": response["message"]["content"]} #<-- return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) #<-- raise exception

@app.get("/") #home endpoint
def home():
    return {"message": "Qwen 2.5 Chatbot API is running"}    





# Run with: uvicorn server:app --reload