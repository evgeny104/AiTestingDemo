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
                                          f"В ответе обязательно указывай номер статьи в формате 'Статья 15'."},
            {"role": "user","content": QUESTION},
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@pytest.mark.parametrize("pattern, description", [
    (r"(?i)стать\w*\s*15", "цитата статьи"),
    (r"\b14\b", "число дней 14"),
    (r"(?i)возврат|верн\w*|возвращ\w*", "слово 'возврат'")
])
def test_answer_matches_pattern(model_answer, pattern, description):
    assert re.search(pattern, model_answer), f"В ответе не найдено: {description} . Ответ: {model_answer!r}"
