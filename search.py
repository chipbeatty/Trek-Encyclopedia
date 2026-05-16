"""
search.py
Semantic search over TNG transcript chunks stored in ChromaDB.

Usage:
    uv run python search.py
    uv run python search.py --top 5 --db data/chroma
"""

import argparse

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "tng_transcripts"

_openai = OpenAI()
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_collection(COLLECTION_NAME)


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed query and return top_k most similar chunks from ChromaDB.
    Each result: {"text": str, "metadata": dict, "distance": float}
    """
    embedding = _openai.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def display_results(results: list[dict]) -> None:
    print("\n--- Top Matches ---")
    for rank, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"\n#{rank} (distance: {r['distance']:.4f})")
        print(f"  S{m['season']:02d}E{m['episode']:02d} - {m['title']} [{m['scene_location']}]")
        print(f"  {r['text'][:200]}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    test_queries = [
        "Borg attack the Enterprise",
        "Data tries to understand human emotions",
        "Picard is captured and tortured",
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: '{query}'")
        display_results(search(query, top_k=args.top))
