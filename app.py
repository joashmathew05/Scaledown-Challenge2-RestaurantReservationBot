"""
Bella Roma AI Restaurant Reservation Bot — FastAPI Application.
Serves the frontend, handles chat routing between RAG and Booking engines.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel

from rag_engine import RAGEngine
from booking_engine import BookingEngine


# Load environment variables from .env file
load_dotenv()

# Directory paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Global engine references
rag_engine: RAGEngine | None = None
booking_engine: BookingEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engines on startup."""
    global rag_engine, booking_engine
    print("🍕 Initializing Bella Roma AI engines...")
    rag_engine = RAGEngine()
    booking_engine = BookingEngine()
    print("✅ Engines ready. Bella Roma is open for business!")
    yield
    print("👋 Bella Roma is closing. Arrivederci!")


# Create FastAPI application
app = FastAPI(
    title="Bella Roma AI",
    description="AI-powered restaurant reservation and menu chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Request model for chat endpoint
class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""
    message: str


# Booking-related keywords for intent detection
BOOKING_KEYWORDS = [
    "book",
    "reserve",
    "reservation",
    "available",
    "availability",
    "table",
    "seat",
    "booking",
]


def is_booking_intent(message: str) -> bool:
    """
    Determine if a message is related to booking/reservations.

    Args:
        message: The user's input message.

    Returns:
        True if the message contains booking-related keywords.
    """
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in BOOKING_KEYWORDS)


@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Handle incoming chat messages.

    Routes to booking engine for reservation queries,
    or to RAG engine for menu-related questions.

    Args:
        chat_request: The incoming chat message.

    Returns:
        JSON response with the bot's reply.
    """
    message = chat_request.message.strip()

    if not message:
        return JSONResponse(
            content={"response": "🍕 Please type a message! I'm here to help with our menu and reservations."}
        )

    # Route to appropriate engine
    if is_booking_intent(message):
        response_text = booking_engine.handle_message(message)
    else:
        response_text = rag_engine.query(message)

    return JSONResponse(content={"response": response_text})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "restaurant": "Bella Roma", "engines": {
        "rag": rag_engine is not None,
        "booking": booking_engine is not None,
    }}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
