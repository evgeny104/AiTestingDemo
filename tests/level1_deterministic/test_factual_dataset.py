"""Уровень 1: детерминированные проверки на МИНИ-DATASET из нескольких кейсов.

Параметризованная фикстура `model_answer` пробегает по CASES,
для каждого делает вызов Groq и возвращает dict с ответом + эталонами.
Тесты строят regex ДИНАМИЧЕСКИ через f-string из полей кейса —
заготовка golden dataset (шаг 3 roadmap).
"""
import re
import pytest
from clients.groq_client import client

pytestmark = pytest.mark.deterministic

CASES = [
    {
        "id": "return-14-days",
        "question": "Сколько дней у меня есть на возврат товара?",
        "context": "Статья 15 закона о защите прав потребителей: "
                   "потребитель имеет право вернуть товар в течение 14 дней "
                   "с момента покупки без объяснения причин.",
        "expected_article": 15,
        "expected_days": 14,
    },
    {
        "id": "warranty-2-years",
        "question": "Какой срок гарантии на товар?",
        "context": "Статья 19 закона о защите прав потребителей: "
                   "гарантийный срок на товар составляет 24 месяца со дня продажи.",
        "expected_article": 19,
        "expected_days": 24,
    },
]


@pytest.fixture(scope="module", params=CASES, ids=[c["id"] for c in CASES])
def model_answer(request):
    case = request.param
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Отвечай только на основе этого текста закона: {case['context']}. "
                    f"В ответе обязательно указывай номер статьи в формате 'Статья {case['expected_article']}'."
                ),
            },
            {"role": "user", "content": case["question"]},

        ],
    )
    return {**case, "answer": response.choices[0].message.content}

def test_answer_cites_expected_article(model_answer):
    pattern = rf"(?i)стать\w*\s*{model_answer['expected_article']}"
    assert re.search(pattern, model_answer["answer"]), (
        f"Не найдена цитата 'Статья {model_answer['expected_article']}' "
        f"в ответе: {model_answer['answer']!r}"
    )

def test_answer_cites_expected_days(model_answer):
    pattern = rf"\b{model_answer['expected_days']}\b"
    assert re.search(pattern, model_answer["answer"]),  (
          f"Не найдено число {model_answer['expected_days']} "                                                                                                                                                                               
          f"в ответе: {model_answer['answer']!r}"
    )