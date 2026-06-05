from unittest.mock import MagicMock, patch
from mathpackage.tracer import LangfuseTracer


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_langfuse_client():
    """
    Build a mock Langfuse v3 client.
    v3 is context-manager based:
        with client.start_as_current_span(...) as span: ...
        with client.start_as_current_generation(...) as gen: ...
    MagicMock supports __enter__/__exit__ by default, so this works naturally.
    """
    client = MagicMock()
    return client


def _make_tracer(enabled=True):
    """Create a LangfuseTracer with a mock client injected."""
    tracer = LangfuseTracer(enabled=False)   # skip real init
    tracer.enabled = enabled
    tracer._client = _mock_langfuse_client() if enabled else None
    return tracer


# ── disabled mode ─────────────────────────────────────────────────────────────

def test_disabled_has_no_client():
    tracer = LangfuseTracer(enabled=False)
    assert tracer.client is None


def test_disabled_flush_is_noop():
    tracer = LangfuseTracer(enabled=False)
    tracer.flush()   # must not raise


def test_disabled_start_trace_returns_noop_context():
    tracer = LangfuseTracer(enabled=False)
    with tracer.start_trace("test") as result:
        assert result is None   # noop context yields None


# ── trace_llm_judge ───────────────────────────────────────────────────────────

@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_trace_llm_judge_returns_correct_keys(mock_anthropic):
    answer_msg = MagicMock(content=[MagicMock(text="4")])
    judge_msg  = MagicMock(content=[MagicMock(text="Correct.\nPASS")])
    mock_anthropic.return_value.messages.create.side_effect = [answer_msg, judge_msg]

    tracer = LangfuseTracer(enabled=False)
    result = tracer.trace_llm_judge("What is 2+2?")

    assert set(result.keys()) == {"question", "answer", "judgment", "verdict"}


@patch("mathpackage.llm_judge.anthropic.Anthropic")
def test_trace_llm_judge_calls_langfuse(mock_anthropic):
    """Enabled tracer: start_as_current_span called once, generations called twice."""
    answer_msg = MagicMock(content=[MagicMock(text="4")])
    judge_msg  = MagicMock(content=[MagicMock(text="Correct.\nPASS")])
    mock_anthropic.return_value.messages.create.side_effect = [answer_msg, judge_msg]

    tracer = _make_tracer(enabled=True)
    result = tracer.trace_llm_judge("What is 2+2?")

    # Root span opened once
    tracer.client.start_as_current_span.assert_called_once()
    # Two generations recorded (answer + judge)
    assert tracer.client.start_as_current_generation.call_count == 2
    # Trace metadata updated
    tracer.client.update_current_trace.assert_called_once()
    # Flushed
    tracer.client.flush.assert_called_once()
    # Result intact
    assert result["verdict"] == "PASS"


# ── trace_similarity ──────────────────────────────────────────────────────────

@patch("mathpackage.similarity._openai")
def test_trace_similarity_calls_langfuse(mock_openai):
    """Enabled tracer: root span + one child span for the embedding call."""
    mock_openai.OpenAI.return_value.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]),
              MagicMock(embedding=[0.8, 0.6])]
    )

    tracer = _make_tracer(enabled=True)
    result = tracer.trace_similarity("cat", "kitten")

    # Root span + one child span = 2 calls to start_as_current_span
    assert tracer.client.start_as_current_span.call_count == 2
    tracer.client.update_current_trace.assert_called_once()
    tracer.client.flush.assert_called_once()
    assert "similarity" in result


@patch("mathpackage.similarity._openai")
def test_trace_similarity_disabled_no_langfuse_calls(mock_openai):
    """Disabled tracer: similarity still runs but no Langfuse calls."""
    mock_openai.OpenAI.return_value.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]),
              MagicMock(embedding=[0.8, 0.6])]
    )

    tracer = LangfuseTracer(enabled=False)
    result = tracer.trace_similarity("cat", "kitten")

    assert "similarity" in result   # result still returned
    assert tracer.client is None    # nothing was called on Langfuse


# ── start_trace ───────────────────────────────────────────────────────────────

def test_start_trace_returns_context_manager():
    tracer = _make_tracer(enabled=True)

    with tracer.start_trace("my-pipeline", input={"q": "hello"}) as span:
        pass   # just verify it runs without error

    tracer.client.start_as_current_span.assert_called_once_with(
        name="my-pipeline",
        input={"q": "hello"},
        metadata={},
    )


# ── flush ─────────────────────────────────────────────────────────────────────

def test_flush_calls_client_flush():
    tracer = _make_tracer(enabled=True)
    tracer.flush()
    tracer.client.flush.assert_called_once()
