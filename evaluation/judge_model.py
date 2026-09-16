import os
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

load_dotenv()

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model
        base_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
        )

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model