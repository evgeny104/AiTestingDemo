from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from evaluation.judge_model import GroqJudge

judge = GroqJudge()

def test_answer_relevancy():
    test_case = LLMTestCase(
        input="Столица Франции?",
        actual_output="Столица Франции — Париж."
    )
    metric = AnswerRelevancyMetric(threshold=1, model=judge)
    assert_test(test_case, [metric])