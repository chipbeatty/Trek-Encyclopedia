"""
embed.py
Embeds TNG transcript chunks and loads them into a persistent ChromaDB collection.
Optionally also embeds per-episode synopses into a second collection used for
concept-level retrieval.

Reads  : data/chunks.json        (output of chunk.py)
         data/tng_episodes.csv   (for --synopses)
Writes : data/chroma/            (ChromaDB persistent storage)

Usage:
    uv run python embed.py                  # embed chunks only
    uv run python embed.py --synopses       # also embed episode synopses
    uv run python embed.py --in data/chunks.json --db data/chroma
    uv run python embed.py --batch 200
"""

import argparse
import csv
import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "tng_transcripts"
SYNOPSIS_COLLECTION_NAME = "tng_synopses"
DEFAULT_BATCH = 200


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # API returns results in the same order as input, but sort by index to be safe
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed_synopses(
    csv_path: Path,
    chroma: chromadb.ClientAPI,
    openai_client: OpenAI,
    batch_size: int,
) -> None:
    """Embed one document per episode from tng_episodes.csv into tng_synopses."""
    episodes: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = f"s{int(row['season']):02d}e{int(row['episode']):02d}"
            if eid not in episodes:  # keep first occurrence for two-part episodes
                episodes[eid] = row

    collection = chroma.get_or_create_collection(
        name=SYNOPSIS_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(collection.get(include=[])["ids"])
    todo = [(eid, row) for eid, row in episodes.items() if eid not in existing_ids]

    if not todo:
        print("Synopsis collection is up to date.")
        return

    print(f"Embedding {len(todo)} episode synopses…")
    episode_ids = [eid for eid, _ in todo]
    texts = [
        f"Title: {row['title']}. Season {row['season']}, Episode {row['episode']}. "
        f"Airdate: {row['airdate']}. Synopsis: {row['synopsis']}"
        for _, row in todo
    ]
    metadatas = [
        {
            "episode_id": eid,
            "title": row["title"],
            "season": int(row["season"]),
            "episode": int(row["episode"]),
            "airdate": row["airdate"],
        }
        for eid, row in todo
    ]

    for i in range(0, len(todo), batch_size):
        batch_ids = episode_ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        embeddings = _embed_batch(openai_client, batch_texts)
        collection.upsert(ids=batch_ids, embeddings=embeddings,
                          documents=batch_texts, metadatas=batch_metas)
        print(f"  {min(i + batch_size, len(todo))}/{len(todo)} synopses embedded")

    print(f"Done. '{SYNOPSIS_COLLECTION_NAME}' has {collection.count()} episodes.")


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
    parser.add_argument(
        "--synopses",
        action="store_true",
        help="Also embed episode synopses into tng_synopses collection",
    )
    parser.add_argument("--episodes", default="data/tng_episodes.csv",
                        help="CSV for --synopses (default: data/tng_episodes.csv)")
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
    openai_client = OpenAI()

    if not todo:
        print("Nothing to embed. Collection is up to date.")
    else:
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

    if args.synopses:
        print()
        embed_synopses(Path(args.episodes), chroma, openai_client, args.batch)


if __name__ == "__main__":
    main()
