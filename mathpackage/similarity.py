from typing import List
import math

try:
    import openai as _openai
except ImportError:
    _openai = None


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = _norm(a) * _norm(b)
    if denom == 0:
        return 0.0
    return _dot(a, b) / denom


def text_similarity(text1: str, text2: str, model: str = "text-embedding-3-small") -> dict:
    """
    Compute cosine similarity between two texts using OpenAI embeddings.

    Args:
        text1: First input text.
        text2: Second input text.
        model: OpenAI embedding model name.

    Returns:
        dict with keys 'text1', 'text2', 'similarity' (float between -1 and 1).

    Requires:
        OPENAI_API_KEY environment variable or openai package configured.
    """
    if _openai is None:
        raise ImportError("openai package is required: pip install openai")

    client = _openai.OpenAI()

    response = client.embeddings.create(input=[text1, text2], model=model)
    emb1 = response.data[0].embedding
    emb2 = response.data[1].embedding

    score = _cosine_similarity(emb1, emb2)

    return {
        "text1": text1,
        "text2": text2,
        "similarity": round(score, 6),
    }
