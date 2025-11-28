from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.main_agent import ask_medical_agent
from src.config import APP_ENV


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


app = FastAPI(
    title="Multi-Tool Medical AI Agent",
    description=(
        "An API that routes medical questions either to dataset-specific tools "
        "(Heart, Cancer, Diabetes) or to a web search tool for general medical knowledge."
    ),
    version="0.1.0",
)

# Static assets (plain HTML/CSS/JS served by FastAPI)
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def read_root():
    """
    Serve the static HTML UI.
    """
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.post("/ask", response_model=AskResponse)
def ask_agent(payload: AskRequest):
    """
    Main endpoint to interact with the medical agent.

    Example request body:
    {
      "question": "What are the symptoms of diabetes?"
    }
    """
    answer = ask_medical_agent(payload.question)
    return AskResponse(question=payload.question, answer=answer)
