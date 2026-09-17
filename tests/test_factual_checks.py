import re
import pytest
from clients.groq_client import client

QUESTION = "Сколько дней у меня есть на возврат товара?"
CONTEXT = (
    "Статья 15 закона о защите прав потребителей: "
    "потребитель имеет право вернуть товар в течение 14 дней "
    "с момента покупки без объяснения причин."
)
BLOCKLIST = ["гарантирую", "100%", "бесплатно"]


@pytest.fixture(scope="module")
def model_answer():
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {"role": "system", "content": f"Отвечай только на основе этого текста закона: {CONTEXT}. "
                                          f"В ответе обязательно указывай номер стать"},
            {"role": "user","content": QUESTION},
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content



def test_answer_cites_article(model_answer):
    assert re.search(r"стать\w*\s*15", model_answer,
                     re.IGNORECASE), f"В ответе нет упоминания 'Статья 15'. Ответ: {model_answer!r}"
