from unittest.mock import patch, MagicMock
from mathpackage.llm_judge import llm_judge


def _make_mock_client(answer_text: str, judgment_text: str):
    client = MagicMock()
    answer_msg = MagicMock()
    answer_msg.content = [MagicMock(text=answer_text)]
    judge_msg = MagicMock()
    judge_msg.content = [MagicMock(text=judgment_text)]
    client.messages.create.side_effect = [answer_msg, judge_msg]
    return client


@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_llm_judge_pass(mock_anthropic):
    mock_anthropic.return_value = _make_mock_client(
        answer_text="Paris is the capital of France.",
        judgment_text="The answer is accurate and clear.\nPASS",
    )
    result = llm_judge("What is the capital of France?")
    assert result["verdict"] == "PASS"
    assert "Paris" in result["answer"]
    assert result["question"] == "What is the capital of France?"


@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_llm_judge_fail(mock_anthropic):
    mock_anthropic.return_value = _make_mock_client(
        answer_text="London is the capital of France.",
        judgment_text="The answer is factually incorrect.\nFAIL",
    )
    result = llm_judge("What is the capital of France?")
    assert result["verdict"] == "FAIL"


@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_llm_judge_keys(mock_anthropic):
    mock_anthropic.return_value = _make_mock_client("answer", "good\nPASS")
    result = llm_judge("test question")
    assert set(result.keys()) == {"question", "answer", "judgment", "verdict"}
