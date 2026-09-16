import requests, os
import json
from dotenv import load_dotenv

load_dotenv()
r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
)


with open("../output.json", "w") as f:
    json.dump(r.json(), f, indent=2, ensure_ascii=False)
