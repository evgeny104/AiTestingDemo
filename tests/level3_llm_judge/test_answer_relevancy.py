"""Уровень 3: LLM-as-judge через DeepEval.

AnswerRelevancyMetric — судья-модель (GroqJudge) оценивает,
отвечает ли модель по существу заданного вопроса.
Дороже и медленнее уровня 1, ловит семантику, а не подстроки.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from evaluation.judge_model import GroqJudge

pytestmark = pytest.mark.llm_judge

judge = GroqJudge()

def test_answer_relevancy():
    test_case = LLMTestCase(
        input="Столица Франции?",
        actual_output="Столица Франции — Париж."
    )
    metric = AnswerRelevancyMetric(threshold=1, model=judge)
    assert_test(test_case, [metric])