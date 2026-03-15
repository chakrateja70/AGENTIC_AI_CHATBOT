import os
from dotenv import load_dotenv  


load_dotenv(override=True)  # Load environment variables from .env file, override if already set

class Settings:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_llm_model = "llama-3.1-8b-instant"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

settings = Settings()