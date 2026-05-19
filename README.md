# mathpackage

A Python package with three utilities:

1. **Math operations** — add, subtract, multiply, divide
2. **Text similarity** — cosine similarity via OpenAI embeddings
3. **LLM as a judge** — ask a question, get an answer, have Claude evaluate it

---

## Install

```bash
cd math-package-tutorial
pip install -e .                 # installing locally if no PyPI available
pip install mathpackage          # from PyPI (after publishing)
pip install -e ".[dev]"          # local editable install (for development)
```

---

## API Key Setup

Two of the three features call external AI APIs. You need to set up API keys
before using them. **Never hardcode keys in your code** — always use environment
variables or a `.env` file.

### Option 1 — `.env` file (recommended for local development)

**Step 1:** Install `python-dotenv`

```bash
pip install python-dotenv
```

**Step 2:** Create a `.env` file in your project root

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

> ⚠️ Never commit this file to git. It is already excluded by `.gitignore`.

**Step 3:** Load it at the top of your script before importing the package

```python
from dotenv import load_dotenv
load_dotenv()   # reads .env and sets the environment variables

from mathpackage import text_similarity, llm_judge
```

---

### Option 2 — Export in your terminal (quick, session-only)

The key is only set for the current terminal session. It disappears when you close the terminal.

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export OPENAI_API_KEY="sk-your-key-here"

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-your-key-here
set OPENAI_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
$env:OPENAI_API_KEY    = "sk-your-key-here"
```

Then run your Python script normally — no extra code needed.

---

### Option 3 — Pass the key directly in code (useful for testing)

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-your-key-here"
os.environ["OPENAI_API_KEY"]    = "sk-your-key-here"

from mathpackage import text_similarity, llm_judge
```

> ⚠️ Only do this in private scripts. Never commit a file with a real key in it.

---

### Where to get the keys

| Key | Where to get it |
|-----|----------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) → API Keys |

---

## Usage

### Math operations (no API key needed)

```python
from mathpackage import add, subtract, multiply, divide

print(add(3, 4))        # 7
print(subtract(10, 3))  # 7
print(multiply(2, 5))   # 10
print(divide(9, 3))     # 3.0
```

### Text similarity (requires `OPENAI_API_KEY`)

```python
from dotenv import load_dotenv
load_dotenv()

from mathpackage import text_similarity

result = text_similarity("The cat sat on the mat", "A feline rested on the rug")
print(result["similarity"])   # e.g. 0.872341  (1.0 = identical, 0.0 = unrelated)
```

### LLM as a judge (requires `ANTHROPIC_API_KEY`)

```python
from dotenv import load_dotenv
load_dotenv()

from mathpackage import llm_judge

result = llm_judge("What is the speed of light?")
print(result["answer"])    # Claude's answer
print(result["judgment"])  # Claude's evaluation of the answer
print(result["verdict"])   # PASS or FAIL
```

---

## How it works

```
math_ops.py   — pure Python, no API calls
similarity.py — sends both texts to OpenAI embeddings API → returns cosine similarity score
llm_judge.py  — call 1: Claude answers the question
                call 2: Claude judges its own answer → returns PASS or FAIL
```

API keys are **never stored in the package**. The `anthropic` and `openai` libraries
automatically read them from the environment variables at runtime.

---

## License

MIT
