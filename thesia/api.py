from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import RAGSystem

app = FastAPI()
rag = RAGSystem()

class Query(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    rag.initialize()

@app.post("/query")
async def query(query: Query):
    try:
        result = rag.query(query.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))