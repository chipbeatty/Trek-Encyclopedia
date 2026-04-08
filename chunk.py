"""
chunk.py
Splits TNG transcripts into overlapping token-window chunks, ready for
embedding and loading into ChromaDB.

Each chunk carries full metadata so retrieval results are self-contained:
  - episode title, season, episode number, stardate, airdate
  - scene location (if scene-splitting succeeded)
  - chunk index within the episode
  - the raw text

Output: data/chunks.json   (list of chunk dicts)

Usage:
    uv run python chunk.py
    uv run python chunk.py --size 500 --overlap 50
    uv run python chunk.py --in data/transcripts.json --out data/chunks.json
    uv run python chunk.py --stats          # print stats and exit, no file written
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokeniser — tiktoken if available, otherwise whitespace fallback
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _enc = tiktoken.get_encoding(
        "cl100k_base"
    )  # same encoding as text-embedding-3-small

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    def split_tokens(text: str, max_tokens: int) -> list[str]:
        """Split text into token-exact slices of max_tokens."""
        ids = _enc.encode(text)
        return [
            _enc.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)
        ]

    TOKENISER = "tiktoken (cl100k_base)"

except ImportError:
    # Rough approximation: 1 token ≈ 4 chars
    def count_tokens(text: str) -> int:
        return len(text) // 4

    def split_tokens(text: str, max_tokens: int) -> list[str]:
        words = text.split()
        chars_per_chunk = max_tokens * 4
        chunks, buf = [], []
        buf_len = 0
        for word in words:
            buf.append(word)
            buf_len += len(word) + 1
            if buf_len >= chars_per_chunk:
                chunks.append(" ".join(buf))
                buf, buf_len = [], 0
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    TOKENISER = "whitespace approximation (install tiktoken for accuracy)"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    # Identity
    chunk_id: str  # e.g. "s01e01_chunk_003"
    episode_id: str  # e.g. "s01e01"

    # Episode metadata
    title: str
    season: int
    episode: int
    stardate: str
    airdate: str

    # Location within transcript
    scene_location: str  # e.g. "BRIDGE" or "UNKNOWN"
    chunk_index: int  # 0-based index within this episode
    chunk_total: int  # total chunks for this episode (filled in post-hoc)

    # Content
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------


def chunk_episode(ep: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    """
    Produce overlapping chunks for one episode.

    Strategy (in priority order):
    1. If the episode has scene data, chunk within each scene first.
       This keeps dialogue in one location together.
    2. If a scene is larger than chunk_size, slide a window over it.
    3. If there are no scenes (or scenes are empty), fall back to chunking
       the raw transcript field.
    """
    title = ep.get("name") or ep.get("title", "Unknown")
    season = ep.get("season", 0)
    number = ep.get("number", 0)
    stardate = ep.get("stardate", "")
    airdate = ep.get("airdate") or ep.get("airdate", "")
    episode_id = f"s{season:02d}e{number:02d}"

    scenes: list[dict] = ep.get("scenes", [])
    transcript: str = ep.get("transcript", "")

    # Build a list of (location, text) segments to chunk
    segments: list[tuple[str, str]] = []
    if scenes:
        for scene in scenes:
            loc = scene.get("location", "UNKNOWN")
            text = scene.get("text", "").strip()
            if text:
                segments.append((loc, text))
    if not segments and transcript:
        segments.append(("UNKNOWN", transcript.strip()))

    chunks: list[Chunk] = []

    for location, text in segments:
        windows = sliding_window(text, chunk_size, overlap)
        for window_text in windows:
            token_count = count_tokens(window_text)
            idx = len(chunks)
            chunk = Chunk(
                chunk_id=f"{episode_id}_chunk_{idx:03d}",
                episode_id=episode_id,
                title=title,
                season=season,
                episode=number,
                stardate=stardate,
                airdate=str(airdate),
                scene_location=location,
                chunk_index=idx,
                chunk_total=0,  # filled below
                text=window_text,
                token_count=token_count,
            )
            chunks.append(chunk)

    # Back-fill total
    total = len(chunks)
    for c in chunks:
        c.chunk_total = total
        # Re-stamp chunk_index relative to episode (already correct)

    return chunks


def sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """
    Slide a token window of `size` tokens over `text` with `overlap` tokens
    of context carried forward from the previous window.

    Returns a list of text strings, each at most `size` tokens.
    """
    if not text.strip():
        return []

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
    except ImportError:
        # Approximate: work on words
        words = text.split()
        token_ids = words  # treat words as "tokens"

        def decode(ids):
            return " ".join(ids)
    else:

        def decode(ids):
            return enc.decode(ids)

    step = size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({size})")

    windows = []
    start = 0
    while start < len(token_ids):
        end = min(start + size, len(token_ids))
        window = token_ids[start:end]
        windows.append(decode(window))
        if end == len(token_ids):
            break
        start += step

    return windows


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------


def print_stats(chunks: list[Chunk], episodes: list[dict]) -> None:
    total_chunks = len(chunks)
    total_tokens = sum(c.token_count for c in chunks)
    avg_tokens = total_tokens / total_chunks if total_chunks else 0

    episodes_with_transcript = sum(
        1 for ep in episodes if ep.get("transcript") or ep.get("scenes")
    )
    episodes_with_scenes = sum(1 for ep in episodes if ep.get("scenes"))

    # Cost estimate: text-embedding-3-small at $0.02 / 1M tokens
    embed_cost = (total_tokens / 1_000_000) * 0.02

    print("\n── Chunking Stats ─────────────────────────────────")
    print(f"  Tokeniser        : {TOKENISER}")
    print(f"  Episodes loaded  : {len(episodes)}")
    print(f"  With transcripts : {episodes_with_transcript}")
    print(f"  With scenes      : {episodes_with_scenes}")
    print(f"  Total chunks     : {total_chunks:,}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Avg tokens/chunk : {avg_tokens:.0f}")
    print(f"  Embed cost est.  : ${embed_cost:.4f}  (text-embedding-3-small)")

    # Per-season breakdown
    by_season: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_season.setdefault(c.season, []).append(c)
    print("\n  Chunks by season:")
    for season in sorted(by_season):
        sc = by_season[season]
        stok = sum(c.token_count for c in sc)
        print(f"    S{season:02d}  {len(sc):4d} chunks   {stok:7,} tokens")

    # Largest and smallest episodes
    by_ep: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_ep.setdefault(c.episode_id, []).append(c)
    ep_tokens = {eid: sum(c.token_count for c in cs) for eid, cs in by_ep.items()}
    sorted_eps = sorted(ep_tokens.items(), key=lambda x: x[1])
    print(f"\n  Smallest episode : {sorted_eps[0][0]}  ({sorted_eps[0][1]:,} tokens)")
    print(f"  Largest episode  : {sorted_eps[-1][0]}  ({sorted_eps[-1][1]:,} tokens)")
    print("────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Chunk TNG transcripts for RAG")
    parser.add_argument("--in", dest="input", default="data/transcripts.json")
    parser.add_argument("--out", default="data/chunks.json")
    parser.add_argument(
        "--size", type=int, default=500, help="Chunk size in tokens (default: 500)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in tokens (default: 50)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print stats and exit without writing output",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Loading transcripts from {in_path}…")
    with in_path.open() as f:
        episodes: list[dict] = json.load(f)
    print(f"Loaded {len(episodes)} episodes.")

    print(f"Chunking (size={args.size}, overlap={args.overlap})…")
    all_chunks: list[Chunk] = []
    for ep in episodes:
        title = ep.get("name") or ep.get("title", "?")
        ep_chunks = chunk_episode(ep, args.size, args.overlap)
        all_chunks.extend(ep_chunks)
        print(f"  {title:45s}  {len(ep_chunks):3d} chunks")

    print_stats(all_chunks, episodes)

    if args.stats:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_chunks):,} chunks to {out_path}")


if __name__ == "__main__":
    main()
