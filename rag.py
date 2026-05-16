"""
rag.py
RAG pipeline: retrieve relevant transcript chunks, then generate an answer.

Usage:
    uv run python rag.py
"""

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from search import search

load_dotenv()

_openai = OpenAI()
_enc = tiktoken.get_encoding("cl100k_base")

CHAT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are an expert on Star Trek: The Next Generation. "
    "Answer the user's question using ONLY the transcript excerpts provided. "
    "If the answer isn't in the excerpts, say so honestly. "
    "Reference specific episode titles and seasons in your answer. "
    "Be conversational but accurate."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        m = chunk["metadata"]
        header = f'[S{m["season"]:02d}E{m["episode"]:02d} "{m["title"]}" — {m["scene_location"]}]'
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def _count_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += 4  # per-message role/formatting overhead
        total += len(_enc.encode(m["content"]))
    return total


def trim_history(history: list[dict], budget: int) -> list[dict]:
    """Drop oldest user/assistant pairs until history fits within the token budget."""
    while len(history) >= 2 and _count_tokens(history) > budget:
        history = history[2:]
    return history


def ask(
    question: str,
    history: list[dict] | None = None,
    top_k: int = 5,
    history_budget: int = 3000,
) -> dict:
    """
    Retrieve relevant chunks and generate an answer, threading conversation history.

    history is a flat list of {"role": "user"/"assistant", "content": ...} pairs
    representing prior bare questions and answers (no retrieved context). Only the
    current turn gets fresh RAG context injected into its user message.

    Returns a dict with question, answer, chunks, and the updated history.
    """
    chunks = search(question, top_k=top_k)
    context = build_context(chunks)

    history = trim_history(list(history or []), history_budget)

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"Here are the most relevant TNG transcript excerpts:\n\n{context}\n\nQuestion: {question}"}]
    )

    response = _openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "history": updated_history,
    }


def display(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print("\nSources:")
    seen: set[str] = set()
    for chunk in result["chunks"]:
        m = chunk["metadata"]
        if m["episode_id"] not in seen:
            print(f"  - S{m['season']:02d}E{m['episode']:02d} {m['title']}")
            seen.add(m["episode_id"])


if __name__ == "__main__":
    test_questions = [
        "What episodes feature the Borg as the main threat?",
        "Which episode deals with Picard being tortured by Cardassians?",
        "Are there any episodes where Data explores what it means to be human?",
    ]

    for question in test_questions:
        display(ask(question))
