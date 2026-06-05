"""
Local runner — try every feature of mathpackage.
Run with:  conda run -n mcp python run_local.py
"""

from dotenv import load_dotenv
load_dotenv()   # reads .env file and sets env vars

from mathpackage import add, subtract, multiply, divide
from mathpackage import text_similarity
from mathpackage import llm_judge
from mathpackage import LangfuseTracer

# ── 1. Math operations (no API key needed) ────────────────────────────────────
print("\n── Math Operations ──────────────────────────────")
print("add(10, 5)      =", add(10, 5))
print("subtract(10, 5) =", subtract(10, 5))
print("multiply(10, 5) =", multiply(10, 5))
print("divide(10, 5)   =", divide(10, 5))

# ── 2. Text similarity (needs OPENAI_API_KEY) ─────────────────────────────────
print("\n── Text Similarity ──────────────────────────────")
result = text_similarity(
    "machine learning is a subset of artificial intelligence",
    "deep learning is part of AI and ML"
)
print(f"Text 1   : {result['text1']}")
print(f"Text 2   : {result['text2']}")
print(f"Similarity: {result['similarity']}")   # 0.0 to 1.0

# ── 3. LLM as a judge (needs ANTHROPIC_API_KEY) ───────────────────────────────
print("\n── LLM as a Judge ───────────────────────────────")
result = llm_judge("What is the boiling point of water?")
print(f"Question : {result['question']}")
print(f"Answer   : {result['answer'][:120]}...")
print(f"Verdict  : {result['verdict']}")       # PASS or FAIL

# ── 4. LangfuseTracer (needs all 3 keys) ─────────────────────────────────────
print("\n── Langfuse Tracer ──────────────────────────────")
tracer = LangfuseTracer()

result = tracer.trace_llm_judge("What is the speed of light?")
print(f"Verdict  : {result['verdict']}")

result = tracer.trace_similarity("cat", "kitten")
print(f"Similarity: {result['similarity']}")

tracer.flush()
print("Traces sent to Langfuse ✓")
print("\nDone.")
