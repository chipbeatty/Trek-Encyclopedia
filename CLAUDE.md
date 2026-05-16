# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) encyclopedia for Star Trek: The Next Generation. The pipeline ingests episode data from two sources, chunks transcripts, embeds them, and serves answers via a CLI chatbot.

## Commands

```bash
# Run any script
uv run python <script>.py

# Ingestion
uv run python scrape.py                          # fetch episode list from TVMaze → data/tng_episodes.csv
uv run python scrape_transcripts.py              # fetch full transcripts from Chakoteya → data/transcripts.json
uv run python scrape_transcripts.py --limit 5   # test run with 5 episodes
uv run python scrape_transcripts.py --delay 2.0 # be polite to the server

# Chunking
uv run python chunk.py                           # chunk transcripts → data/chunks.json
uv run python chunk.py --stats                  # print stats without writing
uv run python chunk.py --size 500 --overlap 50  # tune chunk parameters

# Embedding (early prototype — embeds episode synopses, not chunks)
uv run python embed.py                           # embed episodes → data/embeddings.json

# Search / RAG
uv run python search.py                          # run test semantic searches
uv run python rag.py                             # run test RAG queries
uv run python main.py                            # interactive CLI chatbot

# Add / update dependencies
uv add <package>
```

## Pipeline Architecture

The pipeline runs in discrete stages, each producing a file consumed by the next:

```
scrape.py          → data/tng_episodes.csv      (episode metadata from TVMaze API)
scrape_transcripts.py → data/transcripts.json   (full transcripts from Chakoteya, merged with TVMaze metadata)
chunk.py           → data/chunks.json           (overlapping token-window chunks with scene metadata)
embed.py           → data/embeddings.json       (embeddings — currently embeds synopses, not chunks)
search.py / rag.py / main.py                    (retrieval and generation layer)
```

### Data Sources

- **TVMaze API** (`scrape.py`): episode titles, season/episode numbers, airdates, plot synopses
- **Chakoteya** (`scrape_transcripts.py`): full episode transcripts with scene boundaries. Merges with TVMaze data by episode title.

### Chunking Strategy (`chunk.py`)

Scene-aware sliding window: chunks within scene boundaries first (keeping `[BRIDGE]` dialogue together), then slides a token window (default 500 tokens, 50 overlap) within each scene. Falls back to raw transcript if no scenes parsed. Uses `tiktoken cl100k_base` (same tokenizer as `text-embedding-3-small`) with a whitespace approximation fallback.

Each `Chunk` carries full episode metadata (`title`, `season`, `episode`, `stardate`, `airdate`, `scene_location`) so retrieval results are self-contained.

### Current State

- **Done**: TVMaze scrape, Chakoteya transcript scrape, scene-splitting, token chunking
- **Prototype (not yet connected to chunks)**: `embed.py` embeds per-episode synopses from the old CSV; `search.py` and `rag.py` do cosine similarity over those synopsis embeddings
- **Next**: embed `data/chunks.json` with `text-embedding-3-small`, load into ChromaDB, update `search.py` and `rag.py` to query ChromaDB instead of the flat JSON

### Remaining Pipeline Steps

1. **Embedding chunks** — call `text-embedding-3-small` on each chunk's `text` field from `chunks.json`; estimated cost in `chunk.py --stats`
2. **Vector DB** — load chunk embeddings into ChromaDB (`chroma` already in dependencies); replace flat-file cosine search in `search.py`
3. **Conversation history** — `rag.py`'s `ask()` is stateless; add a message list to pass prior turns to `gpt-4o-mini`
4. **Evaluation** — build a golden Q&A set and measure retrieval recall and answer quality

## Coding Conventions

- All scripts are standalone `if __name__ == "__main__"` entry points run via `uv run python <script>.py`
- CLI flags follow the argparse pattern established in `chunk.py` and `scrape_transcripts.py` (use `--in`/`--out` for I/O paths, `--limit` for test runs)
- Data files live in `data/`; scripts create the directory with `mkdir(parents=True, exist_ok=True)`
- Long-running scrapers checkpoint every 10 items and support resume by skipping already-fetched IDs
- OpenAI client initialised from `OPENAI_API_KEY` in `.env` via `python-dotenv`; the key is gitignored
- `dataclasses` + `asdict()` for structured intermediate data (see `Chunk` in `chunk.py`)
- Tiktoken import is wrapped in try/except for a whitespace-approximation fallback

## Environment

Requires `OPENAI_API_KEY` in `.env`. Python 3.12+, managed with `uv`.
