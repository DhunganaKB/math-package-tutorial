"""
LangfuseTracer — a reusable Langfuse v3 observability abstraction.

Langfuse v3 is fully context-manager based. There is no .trace() method.
The hierarchy is:

    start_as_current_span()        ← root span (acts as the trace)
        update_current_trace()     ← set trace-level name / input / output
        start_as_current_span()    ← child span  (e.g. retrieval step)
        start_as_current_generation() ← LLM call (model, tokens, etc.)

Usage
-----
from mathpackage import LangfuseTracer

tracer = LangfuseTracer()

# Pre-built traced helpers
result = tracer.trace_llm_judge("What is 2+2?")
result = tracer.trace_similarity("cat", "kitten")
tracer.flush()

# Raw trace — instrument your own code
with tracer.start_trace("my-pipeline", input={"query": "..."}) as span:
    span.update(output="done")
tracer.flush()

Environment variables
---------------------
LANGFUSE_PUBLIC_KEY  — from https://us.cloud.langfuse.com → Settings → API Keys
LANGFUSE_SECRET_KEY  — from the same page
LANGFUSE_HOST        — defaults to https://us.cloud.langfuse.com
"""

import os
from contextlib import contextmanager
from typing import Any, Optional


class LangfuseTracer:
    """
    Thin, importable wrapper around the Langfuse v3 Python SDK.

    Parameters
    ----------
    public_key : str, optional
        Falls back to LANGFUSE_PUBLIC_KEY env var.
    secret_key : str, optional
        Falls back to LANGFUSE_SECRET_KEY env var.
    host : str, optional
        Falls back to LANGFUSE_HOST env var, then https://us.cloud.langfuse.com.
    enabled : bool
        Set False to disable all tracing (useful in tests / CI).
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._client = None

        if not enabled:
            return

        try:
            from langfuse import Langfuse
        except ImportError:
            raise ImportError("langfuse package is required: pip install langfuse")

        self._client = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host or os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
        )

    # ------------------------------------------------------------------ #
    #  Pre-built traced helpers                                            #
    # ------------------------------------------------------------------ #

    def trace_llm_judge(
        self,
        question: str,
        model: str = "claude-sonnet-4-6",
    ) -> dict:
        """
        Run llm_judge() and record both LLM calls (answer + judge)
        as Langfuse generations inside one root span.

        Returns the same dict as llm_judge():
            {question, answer, judgment, verdict}
        """
        from .llm_judge import llm_judge

        result = llm_judge(question, model=model)

        if self.enabled and self._client:
            with self._client.start_as_current_span(
                name="llm-judge",
                input=question,
            ) as span:
                self._client.update_current_trace(
                    name="llm-judge",
                    input=question,
                    output=result["verdict"],
                    metadata={"model": model},
                )

                # Generation 1 — the answering call
                with self._client.start_as_current_generation(
                    name="answer-generation",
                    model=model,
                    input=[{"role": "user", "content": question}],
                    output=result["answer"],
                ):
                    pass  # work already done above; we're just recording

                # Generation 2 — the judging call
                with self._client.start_as_current_generation(
                    name="judge-generation",
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n"
                                f"Answer: {result['answer']}\n"
                                "Evaluate the answer."
                            ),
                        }
                    ],
                    output=result["judgment"],
                    metadata={"verdict": result["verdict"]},
                ):
                    pass

                span.update(output=result["verdict"])

            self._client.flush()

        return result

    def trace_similarity(
        self,
        text1: str,
        text2: str,
        model: str = "text-embedding-3-small",
    ) -> dict:
        """
        Run text_similarity() and record the embedding call as a
        Langfuse span inside one root span.

        Returns the same dict as text_similarity():
            {text1, text2, similarity}
        """
        from .similarity import text_similarity

        result = text_similarity(text1, text2, model=model)

        if self.enabled and self._client:
            with self._client.start_as_current_span(
                name="text-similarity",
                input={"text1": text1, "text2": text2},
            ) as span:
                self._client.update_current_trace(
                    name="text-similarity",
                    input={"text1": text1, "text2": text2},
                    output={"similarity": result["similarity"]},
                )

                with self._client.start_as_current_span(
                    name="embedding-cosine-similarity",
                    input={"text1": text1, "text2": text2},
                    output={"similarity": result["similarity"]},
                    metadata={"model": model},
                ):
                    pass

                span.update(output={"similarity": result["similarity"]})

            self._client.flush()

        return result

    # ------------------------------------------------------------------ #
    #  Low-level — for custom instrumentation in other applications        #
    # ------------------------------------------------------------------ #

    def start_trace(
        self,
        name: str,
        input: Any = None,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
    ):
        """
        Return a context manager that opens a root Langfuse span.
        Use this in your own app to instrument any code.

        Example
        -------
        tracer = LangfuseTracer()

        with tracer.start_trace("my-pipeline", input={"query": "..."}) as span:
            # update trace-level fields
            tracer.client.update_current_trace(
                name="my-pipeline",
                input={"query": "..."},
                output="final answer",
            )
            # add a child span
            with tracer.client.start_as_current_span(name="retrieval") as s:
                s.update(output=docs)
            # add an LLM generation
            with tracer.client.start_as_current_generation(
                name="llm-call", model="gpt-4o",
                input=[...], output="response"
            ):
                pass
            span.update(output="done")

        tracer.flush()

        Returns
        -------
        Context manager yielding a LangfuseSpan (or a no-op if disabled).
        """
        if not self.enabled or self._client is None:
            return _noop_context()

        return self._client.start_as_current_span(
            name=name,
            input=input,
            metadata=metadata or {},
        )

    def flush(self) -> None:
        """Push all buffered events to Langfuse. Call at end of script/request."""
        if self.enabled and self._client:
            self._client.flush()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def client(self):
        """
        Direct access to the underlying Langfuse client for advanced use
        (scores, datasets, prompt management, experiments, etc.).
        """
        return self._client

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"LangfuseTracer(status={status})"


# ── helpers ────────────────────────────────────────────────────────────────────

@contextmanager
def _noop_context():
    """A no-op context manager returned when tracing is disabled."""
    yield None
