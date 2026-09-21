"""Уровень 3: LLM-as-judge — проверка галлюцинаций.

HallucinationMetric просит судью оценить, выходил ли ответ
за пределы переданного контекста (`context`).
DeepEval 4.x: 1 = хорошо, 0 = плохо, threshold — минимум.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from evaluation.judge_model import GroqJudge
from clients.groq_client import client

pytestmark = pytest.mark.llm_judge

judge = GroqJudge()

def test_hallucination_legal_context():
    context = [
        "Статья 15 закона о защите прав потребителей: "
        "потребитель имеет право вернуть товар в течение 14 дней "
        "с момента покупки без объяснения причин."
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": f"Отвечай только на основе этого текста закона: {context[0]}"},
            {"role": "user", "content": "Сколько дней у меня есть на возврат товара?"}
        ]

    )
    actual_output = response.choices[0].message.content

    test_case = LLMTestCase(
        input="Сколько дней у меня есть на возврат товара?",
        actual_output=actual_output,
        context=context
    )
    metric = HallucinationMetric(threshold=0.3, model=judge)
    assert_test(test_case, [metric])