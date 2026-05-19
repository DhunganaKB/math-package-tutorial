import anthropic


def llm_judge(question: str, model: str = "claude-sonnet-4-6") -> dict:
    """
    Ask a question to Claude, then have a second Claude call judge the answer.

    Args:
        question: The question to answer.
        model: Anthropic model used for both answering and judging.

    Returns:
        dict with 'question', 'answer', 'judgment', and 'verdict' keys.

    Requires:
        ANTHROPIC_API_KEY environment variable.
    """
    client = anthropic.Anthropic()

    # Step 1: get the answer
    answer_msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    answer = answer_msg.content[0].text

    # Step 2: judge the answer
    judge_prompt = (
        f"You are a strict but fair evaluator.\n\n"
        f"Question: {question}\n\n"
        f"Answer provided: {answer}\n\n"
        f"Evaluate the answer on: accuracy, completeness, and clarity. "
        f"End your evaluation with a one-word verdict on its own line: PASS or FAIL."
    )

    judge_msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    judgment = judge_msg.content[0].text

    # Extract final verdict from last non-empty line
    lines = [l.strip() for l in judgment.splitlines() if l.strip()]
    verdict = lines[-1].upper() if lines else "UNKNOWN"
    if verdict not in ("PASS", "FAIL"):
        verdict = "UNKNOWN"

    return {
        "question": question,
        "answer": answer,
        "judgment": judgment,
        "verdict": verdict,
    }
