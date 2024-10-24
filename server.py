from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Explicitly specify allowed methods
    allow_headers=["*"],
)

# Pydantic model for request body
class Question(BaseModel):
    question: str

# Initialize prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please response to the user queries"),
    ("user", "Question:{question}")
])

# Initialize Groq LLM and chain
llm = ChatGroq(
    temperature=0.7,
    model_name="mixtral-8x7b-32768",
    api_key=os.getenv("GROQ_API_KEY")
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.get("/")
async def read_root():
    return {"message": "Welcome to LangChain API with Groq"}

@app.post("/ask/")  # Note the trailing slash
async def get_answer(question: Question):
    try:
        response = chain.invoke({"question": question.question})
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
