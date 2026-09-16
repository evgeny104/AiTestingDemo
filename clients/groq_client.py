import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
"""подключение к сервису, который тестируем"""
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
