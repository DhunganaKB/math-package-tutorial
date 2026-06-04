"""
LangfuseTracer — a reusable Langfuse observability abstraction.

Usage
-----
from mathpackage import LangfuseTracer

tracer = LangfuseTracer()                        # reads keys from env
result = tracer.trace_llm_judge("What is 2+2?")  # runs + traces
result = tracer.trace_similarity("cat", "kitten") # runs + traces
tracer.flush()                                    # send all events

Environment variables required
-------------------------------
LANGFUSE_PUBLIC_KEY  — from https://us.cloud.langfuse.com → Settings → API Keys
LANGFUSE_SECRET_KEY  — from the same page
LANGFUSE_HOST        — defaults to https://us.cloud.langfuse.com
"""

import os
from typing import Any, Optional


class LangfuseTracer:
    """
    Thin, importable wrapper around the Langfuse Python SDK.

    Provides:
      - trace_llm_judge()   : run llm_judge() and record both LLM calls
      - trace_similarity()  : run text_similarity() and record the embedding call
      - start_trace()       : open a raw trace for custom instrumentation
      - flush()             : push all buffered events to Langfuse
      - .client             : direct access to the underlying Langfuse object

    Parameters
    ----------
    public_key : str, optional
        Langfuse public key. Falls back to LANGFUSE_PUBLIC_KEY env var.
    secret_key : str, optional
        Langfuse secret key. Falls back to LANGFUSE_SECRET_KEY env var.
    host : str, optional
        Langfuse host URL. Defaults to https://us.cloud.langfuse.com
        (the US cloud instance).
    enabled : bool
        Set to False to disable all tracing (useful in testing / CI).
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
            raise ImportError(
                "langfuse package is required: pip install langfuse"
            )

        self._client = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host
            or os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
        )

    # ------------------------------------------------------------------ #
    #  High-level traced helpers                                           #
    # ------------------------------------------------------------------ #

    def trace_llm_judge(
        self,
        question: str,
        model: str = "claude-sonnet-4-6",
    ) -> dict:
        """
        Run llm_judge() and record both API calls (answer + judge) as
        separate Langfuse generations inside one trace.

        Returns
        -------
        dict  — same as llm_judge(): {question, answer, judgment, verdict}
        """
        from .llm_judge import llm_judge

        result = llm_judge(question, model=model)

        if self.enabled and self._client:
            trace = self._client.trace(
                name="llm-judge",
                input=question,
                output=result["verdict"],
                metadata={"model": model},
            )

            # Generation 1 — the answering call
            trace.generation(
                name="answer-generation",
                model=model,
                input=[{"role": "user", "content": question}],
                output=result["answer"],
            )

            # Generation 2 — the judging call
            trace.generation(
                name="judge-generation",
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Answer: {result['answer']}\n"
                            f"Evaluate the answer."
                        ),
                    }
                ],
                output=result["judgment"],
                metadata={"verdict": result["verdict"]},
            )

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
        Langfuse span inside one trace.

        Returns
        -------
        dict  — same as text_similarity(): {text1, text2, similarity}
        """
        from .similarity import text_similarity

        result = text_similarity(text1, text2, model=model)

        if self.enabled and self._client:
            trace = self._client.trace(
                name="text-similarity",
                input={"text1": text1, "text2": text2},
                output={"similarity": result["similarity"]},
            )

            trace.span(
                name="embedding-cosine-similarity",
                input={"text1": text1, "text2": text2},
                output={"similarity": result["similarity"]},
                metadata={"model": model},
            )

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
        Open a raw Langfuse trace. Use this in your own app to instrument
        any code, not just the functions in this package.

        Returns the Langfuse trace object (or None if tracing is disabled).

        Example
        -------
        tracer = LangfuseTracer()
        trace = tracer.start_trace("my-pipeline", input={"query": "..."})
        span  = trace.span(name="step-1", input=..., output=...)
        gen   = trace.generation(name="llm-call", model="...", ...)
        tracer.flush()
        """
        if not self.enabled or self._client is None:
            return None

        return self._client.trace(
            name=name,
            input=input,
            metadata=metadata or {},
            tags=tags or [],
        )

    def flush(self) -> None:
        """
        Flush all buffered Langfuse events to the server.
        Call this at the end of a script or request lifecycle.
        """
        if self.enabled and self._client:
            self._client.flush()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def client(self):
        """
        Direct access to the underlying Langfuse client for advanced use
        (scores, datasets, prompt management, etc.).
        """
        return self._client

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        host = (
            self._client.base_url
            if self._client and hasattr(self._client, "base_url")
            else "https://us.cloud.langfuse.com"
        )
        return f"LangfuseTracer(status={status}, host={host})"
