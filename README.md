# Agentic AI Chatbot

An agentic AI chatbot built with LangChain and Groq that intelligently selects external tools (Wikipedia, Arxiv, Tavily) based on the user's query, or falls back to the LLM's own knowledge when no external lookup is needed.

## How It Works

1. **Query Analysis** – The chatbot inspects each incoming query for keywords that indicate a need for external or up-to-date information (e.g., "latest", "news", "research paper", "current").
2. **Tool Selection** – If external information is required, the most appropriate tool is chosen:
   - **Arxiv** – academic / research-paper queries (keywords: "arxiv", "research", "paper", "transformer", etc.)
   - **Tavily** – real-time web search for fresh information (keywords: "latest", "news", "today", "update", etc.)
   - **Wikipedia** – general encyclopedic background knowledge (default fallback when a tool is needed)
3. **LLM Response** – The selected tool's output is fed to the Groq LLM (`llama-3.1-8b-instant`) together with the original query, and a concise answer is returned. Queries that don't need external data are answered directly from the model's knowledge.

## Project Structure

```
AGENTIC_AI_CHATBOT/
├── main.py                  # Entry point
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Pip-compatible dependency list
├── .env.development         # Template for environment variables
└── src/
    ├── __init__.py
    ├── settings.py          # Loads API keys from environment
    ├── llm_service.py       # Core logic: tool selection and LLM invocation
    └── tools/
        ├── __init__.py
        ├── arxiv_tool.py    # LangChain Arxiv tool wrapper
        ├── tavily_tool.py   # LangChain Tavily search tool wrapper
        └── wikipedia_tool.py # LangChain Wikipedia tool wrapper
```

## Prerequisites

- Python ≥ 3.11
- A [Groq](https://console.groq.com/) API key
- A [Tavily](https://app.tavily.com/) API key

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/chakrateja70/AGENTIC_AI_CHATBOT.git
cd AGENTIC_AI_CHATBOT

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy `.env.development` to `.env` and fill in your API keys:

```bash
cp .env.development .env
```

`.env` contents:

```env
GROQ_API_KEY="your_groq_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

## Usage

```python
from src.llm_service import llm_run

# General knowledge query – answered directly by the LLM
response = llm_run("What is machine learning?")
print(response)

# Research query – routed to the Arxiv tool
response = llm_run("Find recent papers on transformer architectures")
print(response)

# Real-time query – routed to the Tavily search tool
response = llm_run("What are the latest AI news today?")
print(response)

# Encyclopedic query – routed to the Wikipedia tool
response = llm_run("Tell me about the history of neural networks")
print(response)
```

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework |
| `langchain-core` | LangChain base abstractions |
| `langchain-community` | Community integrations (Wikipedia, Arxiv) |
| `langchain-groq` | Groq LLM integration |
| `langchain-tavily` | Tavily web search integration |
| `langgraph` | Graph-based agent orchestration |
| `arxiv` | Arxiv paper search |
| `wikipedia` | Wikipedia article retrieval |
| `pydantic` | Data validation |
| `python-dotenv` | Environment variable loading |
