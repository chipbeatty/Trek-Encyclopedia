"""
embed.py
Embeds TNG transcript chunks and loads them into a persistent ChromaDB collection.

Reads  : data/chunks.json   (output of chunk.py)
Writes : data/chroma/       (ChromaDB persistent storage)

Usage:
    uv run python embed.py
    uv run python embed.py --in data/chunks.json --db data/chroma
    uv run python embed.py --batch 200
    uv run python embed.py --collection tng_transcripts
"""

import argparse
import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "tng_transcripts"
DEFAULT_BATCH = 200


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # API returns results in the same order as input, but sort by index to be safe
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def main():
    parser = argparse.ArgumentParser(description="Embed TNG chunks into ChromaDB")
    parser.add_argument("--in", dest="input", default="data/chunks.json")
    parser.add_argument("--db", default="data/chroma", help="ChromaDB storage directory")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Chunks per OpenAI embedding call (default: 200)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {in_path}")

    print(f"Loading chunks from {in_path}…")
    with in_path.open() as f:
        chunks: list[dict] = json.load(f)
    print(f"Loaded {len(chunks):,} chunks.")

    # Set up ChromaDB persistent collection
    db_path = Path(args.db)
    db_path.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(db_path))
    collection = chroma.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    # Resume: skip chunks that are already in the collection
    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    todo = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if existing_ids:
        print(f"Resuming — {len(existing_ids):,} already embedded, {len(todo):,} remaining.")
    if not todo:
        print("Nothing to embed. Collection is up to date.")
        return

    openai_client = OpenAI()
    total = len(todo)
    done = 0

    for i in range(0, total, args.batch):
        batch = todo[i : i + args.batch]
        embeddings = _embed_batch(openai_client, [c["text"] for c in batch])

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "episode_id": c["episode_id"],
                    "title": c["title"],
                    "season": c["season"],
                    "episode": c["episode"],
                    "stardate": c["stardate"],
                    "airdate": c["airdate"],
                    "scene_location": c["scene_location"],
                    "chunk_index": c["chunk_index"],
                    "chunk_total": c["chunk_total"],
                    "token_count": c["token_count"],
                }
                for c in batch
            ],
        )

        done += len(batch)
        print(f"  {done:,}/{total:,} chunks embedded")

    print(f"\nDone. '{args.collection}' has {collection.count():,} chunks in {db_path.resolve()}")


if __name__ == "__main__":
    main()
