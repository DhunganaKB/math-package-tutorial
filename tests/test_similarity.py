import pytest
from unittest.mock import patch, MagicMock
from mathpackage.similarity import text_similarity, _cosine_similarity


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


@patch("mathpackage.similarity._openai")
def test_text_similarity_returns_dict(mock_openai):
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    fake_emb = MagicMock()
    fake_emb.data = [
        MagicMock(embedding=[1.0, 0.0, 0.0]),
        MagicMock(embedding=[0.8, 0.6, 0.0]),
    ]
    mock_client.embeddings.create.return_value = fake_emb

    result = text_similarity("hello", "hi")

    assert "text1" in result
    assert "text2" in result
    assert "similarity" in result
    assert isinstance(result["similarity"], float)
    assert -1.0 <= result["similarity"] <= 1.0
