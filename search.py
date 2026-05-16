"""
search.py
Semantic search over TNG transcript chunks stored in ChromaDB.

Uses two-stage retrieval when the tng_synopses collection is available:
  1. Query tng_transcripts directly (dialogue-level)
  2. Query tng_synopses for concept-level episode matches, then fetch their
     best chunks from tng_transcripts
  3. Merge both result sets, deduplicate by chunk_id, return top_k

Usage:
    uv run python search.py
    uv run python search.py --top 5
"""

import argparse

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "tng_transcripts"
SYNOPSIS_COLLECTION_NAME = "tng_synopses"

_openai = OpenAI()
_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma.get_collection(COLLECTION_NAME)

try:
    _synopsis_collection = _chroma.get_collection(SYNOPSIS_COLLECTION_NAME)
except Exception:
    _synopsis_collection = None


def search(query: str, top_k: int = 5, synopsis_candidates: int = 3) -> list[dict]:
    """
    Embed query and return top_k most similar chunks from ChromaDB.

    If the synopsis collection exists, also queries it for concept-level
    episode matches and merges those chunks into the results.

    Each result: {"text": str, "metadata": dict, "distance": float}
    """
    embedding = _openai.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

    def _query_chunks(extra_kwargs: dict = {}) -> list[dict]:
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            **extra_kwargs,
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # Stage 1: direct chunk search
    chunk_results = _query_chunks()

    if _synopsis_collection is None:
        return chunk_results

    # Stage 2: synopsis search → guaranteed slot for best chunk of each matched episode.
    # Synopsis matches are guaranteed inclusions regardless of chunk distance — the synopsis
    # correctly identifies the episode even when dialogue vocabulary doesn't match the query.
    synopsis_results = _synopsis_collection.query(
        query_embeddings=[embedding],
        n_results=synopsis_candidates,
        include=["metadatas"],
    )
    guaranteed: list[dict] = []
    for m in synopsis_results["metadatas"][0]:
        eid = m["episode_id"]
        r = _collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"episode_id": {"$in": [eid]}},
            include=["documents", "metadatas", "distances"],
        )
        if r["ids"][0]:
            guaranteed.append({
                "text": r["documents"][0][0],
                "metadata": r["metadatas"][0][0],
                "distance": r["distances"][0][0],
            })

    # Fill remaining slots with direct chunk results, skipping already-guaranteed episodes
    guaranteed_eids = {r["metadata"]["episode_id"] for r in guaranteed}
    fill = [r for r in chunk_results if r["metadata"]["episode_id"] not in guaranteed_eids]

    return (guaranteed + fill)[:top_k]


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
