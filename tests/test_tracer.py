from unittest.mock import MagicMock, patch
from mathpackage.tracer import LangfuseTracer


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_langfuse():
    """Return a mock Langfuse client wired up with a trace."""
    client = MagicMock()
    trace  = MagicMock()
    client.trace.return_value = trace
    return client, trace


# ── disabled mode ─────────────────────────────────────────────────────────────

def test_tracer_disabled_has_no_client():
    tracer = LangfuseTracer(enabled=False)
    assert tracer.client is None


def test_tracer_disabled_flush_is_noop():
    tracer = LangfuseTracer(enabled=False)
    tracer.flush()   # must not raise


def test_tracer_disabled_start_trace_returns_none():
    tracer = LangfuseTracer(enabled=False)
    assert tracer.start_trace("test") is None


# ── trace_llm_judge ───────────────────────────────────────────────────────────

@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_trace_llm_judge_returns_correct_keys(mock_anthropic):
    """Disabled tracer still returns the correct dict from llm_judge."""
    answer_msg = MagicMock(content=[MagicMock(text="4")])
    judge_msg  = MagicMock(content=[MagicMock(text="Correct.\nPASS")])
    mock_anthropic.return_value.messages.create.side_effect = [answer_msg, judge_msg]

    tracer = LangfuseTracer(enabled=False)
    result = tracer.trace_llm_judge("What is 2+2?")

    assert set(result.keys()) == {"question", "answer", "judgment", "verdict"}


@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_trace_llm_judge_calls_langfuse(mock_anthropic):
    """When enabled, trace_llm_judge should call client.trace and two generations."""
    # Mock Anthropic
    answer_msg = MagicMock(content=[MagicMock(text="4")])
    judge_msg  = MagicMock(content=[MagicMock(text="Correct.\nPASS")])
    mock_anthropic.return_value.messages.create.side_effect = [answer_msg, judge_msg]

    tracer = LangfuseTracer(enabled=False)
    tracer.enabled = True
    client, trace = _mock_langfuse()
    tracer._client = client

    result = tracer.trace_llm_judge("What is 2+2?")

    # Langfuse trace was opened
    client.trace.assert_called_once()
    # Two generations were recorded
    assert trace.generation.call_count == 2
    # Result still comes back correctly
    assert result["verdict"] == "PASS"


# ── trace_similarity ──────────────────────────────────────────────────────────

@patch("mathpackage.similarity._openai")
def test_trace_similarity_calls_langfuse(mock_openai):
    """When enabled, trace_similarity should call client.trace and one span."""
    # Mock OpenAI embeddings
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[1.0, 0.0]),
            MagicMock(embedding=[0.8, 0.6]),
        ]
    )

    tracer = LangfuseTracer(enabled=False)
    tracer.enabled = True
    client, trace = _mock_langfuse()
    tracer._client = client

    result = tracer.trace_similarity("cat", "kitten")

    client.trace.assert_called_once()
    trace.span.assert_called_once()
    assert "similarity" in result


# ── start_trace ───────────────────────────────────────────────────────────────

def test_start_trace_returns_trace_object():
    tracer = LangfuseTracer(enabled=False)
    tracer.enabled = True
    client, trace = _mock_langfuse()
    tracer._client = client

    t = tracer.start_trace("my-pipeline", input={"q": "hello"})

    client.trace.assert_called_once_with(
        name="my-pipeline",
        input={"q": "hello"},
        metadata={},
        tags=[],
    )
    assert t is trace


# ── flush ─────────────────────────────────────────────────────────────────────

def test_flush_calls_client_flush():
    tracer = LangfuseTracer(enabled=False)
    tracer.enabled = True
    client, _ = _mock_langfuse()
    tracer._client = client

    tracer.flush()
    client.flush.assert_called_once()
