from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models
class Question(BaseModel):
    question: str

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="mixtral-8x7b-32768",
    api_key=os.getenv("GROQ_API_KEY")
)
output_parser = StrOutputParser()

# Create a default system message
DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant. Be concise and clear in your responses. 
Maintain context of the conversation to provide relevant and coherent answers."""

@app.get("/")
async def read_root():
    return {"message": "Welcome to LangChain API with Groq"}

@app.post("/ask")
async def get_answer(question: Question):
    try:
        # Create a simple prompt with just the system message and user question
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEFAULT_SYSTEM_MESSAGE),
            ("user", question.question)
        ])
        
        # Create chain and invoke
        chain = prompt | llm | output_parser
        
        # Get response with proper parameter passing
        response = chain.invoke({"question": question.question})
        
        return {
            "answer": response
        }
        
    except Exception as e:
        # Log the error (you might want to use proper logging here)
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
