"""SUT-клиент: Groq играет роль тестируемого 'вендорского' LLM-эндпоинта.

Импортируется всеми тестами, которые делают реальный запрос к модели
(уровень 1 factual, уровень 3 hallucination).
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))