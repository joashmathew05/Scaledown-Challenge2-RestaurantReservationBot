# 🍕 Bella Roma — AI Restaurant Reservation Bot

A modern, AI-powered restaurant reservation and menu assistant built with **FastAPI**, **LangChain**, **Groq (LLaMA 3.3 70B)**, **HuggingFace Embeddings**, and **FAISS**. Features an elegant café-themed chat interface with a subtle pizza-pattern aesthetic.

---

## Overview

**Bella Roma AI** is a full-stack web application that combines:

- **RAG (Retrieval-Augmented Generation)** for intelligent menu Q&A
- **Rule-based booking logic** for table reservations
- **A polished, café-themed chat UI** with responsive design

Users can ask about menu items, dietary options, pricing, and make table reservations — all through natural language conversation.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)             │
│              Elegant café-themed chat UI              │
│         Scattered pizza SVG background icons          │
└──────────────────┬───────────────────────────────────┘
                   │  POST /chat  { "message": "..." }
                   ▼
┌──────────────────────────────────────────────────────┐
│                 FastAPI Backend (app.py)              │
│          Intent Detection (keyword routing)           │
│                                                      │
│   ┌─────────────────┐     ┌────────────────────┐     │
│   │  Booking Engine  │     │    RAG Engine       │    │
│   │  (rule-based)    │     │  (LangChain+FAISS) │    │
│   │                  │     │                    │     │
│   │  • check avail.  │     │  • OpenAI Embed.   │    │
│   │  • book table    │     │  • FAISS search    │    │
│   │  • suggest alt.  │     │  • GPT-4o-mini     │    │
│   └─────────────────┘     └────────────────────┘     │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │              Data Layer (JSON)                │   │
│   │  menu.json • compressed_menu.json             │   │
│   │  availability.json                            │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## RAG (Retrieval-Augmented Generation)

The RAG engine powers menu-related conversations:

1. **Data Loading** — Compressed menu descriptions are loaded from `compressed_menu.json`
2. **Embedding** — Text chunks are embedded locally using HuggingFace's `all-MiniLM-L6-v2` model (free, no API key needed)
3. **Vector Store** — Embeddings are stored in a FAISS index for fast similarity search
4. **Retrieval** — User queries are matched against the top-4 most relevant menu chunks
5. **Generation** — Groq's `llama-3.3-70b-versatile` generates a response using only the retrieved context
6. **Guardrails** — A strict system prompt ensures the bot only answers from the menu data

---

## Booking Logic

The booking engine handles reservations with structured rules:

- **Availability Check** — Validates date and time against `availability.json`
- **Table Booking** — Decrements available tables on successful reservation
- **Smart Suggestions** — If a slot is full, suggests alternative times on the same day; if the day is full, suggests alternative dates
- **Input Parsing** — Extracts dates (YYYY-MM-DD), times (HH:MM), and guest counts from natural language
- **Default Date** — Falls back to 2026-02-20 if no date is specified
- **State Persistence** — Availability updates persist throughout the application runtime

---

## UI Design

The interface features an **elegant café aesthetic** inspired by premium bakery and restaurant websites:

- **Color Palette** — Olive greens, warm cream backgrounds, and accent gold
- **Chat Bubbles** — Cream-colored bot messages (left) and olive-green user messages (right)
- **Pizza Pattern** — 10 scattered pizza SVG icons with low opacity (5–8%) behind the chat, using absolute positioning with varied sizes and rotations
- **Typography** — Inter font family for clean readability
- **Animations** — Smooth fade-in for messages, bouncing typing indicator dots
- **Quick Actions** — Pre-set buttons for common queries (Vegan Options, Full Menu, Book Table, Availability)
- **Responsive** — Full-width on mobile, centered card on desktop

### Creative Feature

> *Themed restaurant chatbot with subtle pizza-pattern aesthetic* — The scattered pizza SVG icons create an immersive restaurant atmosphere without distracting from the conversation, blending form and function.

---

## How to Run

### Prerequisites

- Python 3.10+
- A free Groq API key → [Get one here](https://console.groq.com/keys)

### Setup

```bash
# Clone the repository
git clone https://github.com/joashmathew05/Scaledown-Challenge2-RestaurantReservationBot.git
cd Scaledown-Challenge2-RestaurantReservationBot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your Groq API key
```

### Run the Application

```bash
python app.py
```

Or with Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000**

---

## Example Queries

### Menu Questions (RAG)
| Query | Expected Behavior |
|-------|------------------|
| "What vegan options do you have?" | Lists vegan menu items with prices |
| "Tell me about the Margherita Pizza" | Describes the dish with details |
| "What's the cheapest item?" | Identifies Tomato Soup at $5 |
| "Do you have sushi?" | Responds that it's not on the menu |
| "Show me the desserts" | Lists dessert items |

### Reservations (Booking Engine)
| Query | Expected Behavior |
|-------|------------------|
| "Book a table for 4 at 19:00" | Confirms reservation with details |
| "Check availability at 20:00" | Shows available tables for the slot |
| "Reserve for 2 guests at 18:00 on 2026-02-21" | Books on specified date |
| "Book a table at 21:00 for 3 guests" | Handles booking (limited tables) |
| "Are there tables at 20:00?" | Checks and reports availability |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 (local, free) |
| RAG | LangChain, FAISS |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data | JSON (menu, availability) |
| Config | python-dotenv |

---

## Project Structure

```
bella-roma-ai/
├── app.py                  # FastAPI application & routing
├── booking_engine.py       # Rule-based reservation system
├── rag_engine.py           # RAG pipeline (LangChain + FAISS)
├── data/
│   ├── menu.json           # Full menu data
│   ├── compressed_menu.json # Simplified text chunks for RAG
│   └── availability.json   # Table availability by date/time
├── templates/
│   └── index.html          # Chat interface
├── static/
│   ├── style.css           # Café-themed styles
│   └── pizza.svg           # Pizza slice icon
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Future Improvements

- **Persistent Storage** — Use SQLite or PostgreSQL for bookings that survive restarts
- **Multi-language Support** — Add Italian and Spanish language options
- **Order Placement** — Allow users to place food orders directly through chat
- **User Accounts** — Authentication and reservation history
- **Admin Dashboard** — Real-time table management and analytics
- **Voice Input** — Speech-to-text for hands-free ordering
- **Email Confirmations** — Send booking confirmations via email
- **Calendar Integration** — Sync reservations with Google Calendar
- **Menu Images** — Display dish photos in chat responses
- **Feedback System** — Collect ratings and reviews through the bot

---

## License

MIT License

---

*Built with ❤️ and 🍕 by Bella Roma AI*
