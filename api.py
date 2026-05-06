from fastapi import FastAPI
from pydantic import BaseModel
from app import chain

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    response = chain.invoke({"input": req.question})
    return {"answer": response["answer"]}