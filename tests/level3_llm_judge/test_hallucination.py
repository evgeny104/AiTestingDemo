"""Уровень 3: LLM-as-judge — проверка галлюцинаций.

HallucinationMetric просит судью оценить, выходил ли ответ
за пределы переданного контекста (`context`).
DeepEval 4.x: 1 = хорошо, 0 = плохо, threshold — минимум.
"""
import pytest
from deepeval import assert_test
# HallucinationMetric — сверяет actual_output с полем context.
# Всё в ответе, чего НЕТ в context, judge считает галлюцинацией.
# DeepEval 4.x: 1 = ответ полностью опирается на context, 0 = сплошные выдумки.
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from evaluation.judge_model import GroqJudge
# client — SUT-клиент (Groq в роли тестируемого "вендора"). ВАЖНО: тут два разных вызова модели:
#   1) через client → генерируем actual_output (роль SUT),
#   2) внутри HallucinationMetric → judge оценивает ответ (роль оценщика).
# Обе роли исполняет Groq, но концептуально это разные ответственности.
from clients.groq_client import client

pytestmark = pytest.mark.llm_judge

judge = GroqJudge()

def test_hallucination_legal_context():
    # context — СПИСОК строк, не строка. DeepEval поддерживает несколько кусков контекста
    # (аналог retrieved chunks в RAG). Тут один элемент — можно легко расширить до пяти.
    context = [
        "Статья 15 закона о защите прав потребителей: "
        "потребитель имеет право вернуть товар в течение 14 дней "
        "с момента покупки без объяснения причин."
    ]

    # === Роль 1: SUT. Просим "вендорскую" модель ответить на юзер-вопрос. ===
    # В system-промпт кладём context[0] — единственный кусок; в реальном RAG сюда
    # уйдёт объединение всех retrieved docs.
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": f"Отвечай только на основе этого текста закона: {context[0]}"},
            {"role": "user", "content": "Сколько дней у меня есть на возврат товара?"}
        ]

    )
    actual_output = response.choices[0].message.content

    # === Роль 2: сборка данных для оценки. ===
    # LLMTestCase несёт три поля, которые нужны HallucinationMetric:
    # input — вопрос, actual_output — что модель ответила, context — эталонная "истина".
    test_case = LLMTestCase(
        input="Сколько дней у меня есть на возврат товара?",
        actual_output=actual_output,
        context=context
    )
    # threshold=0.3 — МЯГКИЙ порог: "минимум 30% фактов должны опираться на context".
    # На legal/fintech обычно 0.7–0.8 (строже). Тут пока учебный кейс.
    metric = HallucinationMetric(threshold=0.3, model=judge)
    # assert_test → metric.measure(test_case) → judge получает промпт вида
    # "вот ответ [actual_output], вот эталон [context], оцени долю фактов от context"
    # → judge возвращает JSON {score, reason} → метрика сверяет score с threshold.
    assert_test(test_case, [metric])