from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Replace with your Groq API key

app = FastAPI()

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
    model_name="mixtral-8x7b-32768",  # You can also use "llama2-70b-4096"
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.get("/")
def read_root():
    return {"message": "Welcome to LangChain API with Groq"}

@app.post("/ask")
async def get_answer(question: Question):
    try:
        response = chain.invoke({"question": question.question})
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)