"""
rag.py
RAG pipeline: retrieve relevant transcript chunks, then generate an answer.

Usage:
    uv run python rag.py
"""

from dotenv import load_dotenv
from openai import OpenAI

from search import search

load_dotenv()

_openai = OpenAI()
CHAT_MODEL = "gpt-4o-mini"


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        m = chunk["metadata"]
        header = f'[S{m["season"]:02d}E{m["episode"]:02d} "{m["title"]}" — {m["scene_location"]}]'
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def ask(question: str, top_k: int = 5) -> dict:
    chunks = search(question, top_k=top_k)
    context = build_context(chunks)

    response = _openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert on Star Trek: The Next Generation. "
                    "Answer the user's question using ONLY the transcript excerpts provided. "
                    "If the answer isn't in the excerpts, say so honestly. "
                    "Reference specific episode titles and seasons in your answer. "
                    "Be conversational but accurate."
                ),
            },
            {
                "role": "user",
                "content": f"Here are the most relevant TNG transcript excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.3,
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "chunks": chunks,
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
