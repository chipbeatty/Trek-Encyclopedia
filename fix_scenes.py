"""
fix_scenes.py
Re-runs scene splitting on existing transcripts.json without re-scraping.

Run this once after the scene regex fix, then re-run chunk.py and embed.py.

Usage:
    uv run python fix_scenes.py
    uv run python fix_scenes.py --in data/transcripts.json
"""

import argparse
import json
from pathlib import Path

from scrape_transcripts import _split_into_scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", default="data/transcripts.json")
    args = parser.parse_args()

    path = Path(args.input)
    with path.open() as f:
        episodes: list[dict] = json.load(f)

    unknown_before = sum(
        1 for ep in episodes
        if ep.get("scenes") and ep["scenes"][0]["location"] == "UNKNOWN"
    )
    print(f"Loaded {len(episodes)} episodes, {unknown_before} with UNKNOWN scenes.")
    print("Re-splitting scenes…")

    for ep in episodes:
        transcript = ep.get("transcript", "")
        if not transcript:
            continue
        ep["scenes"] = _split_into_scenes(transcript)

    unknown_after = sum(
        1 for ep in episodes
        if ep.get("scenes") and ep["scenes"][0]["location"] == "UNKNOWN"
    )
    total_scenes = sum(len(ep.get("scenes", [])) for ep in episodes)
    print(f"Done. {total_scenes:,} scenes extracted, {unknown_after} episodes still UNKNOWN.")

    with path.open("w") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print(f"Saved updated transcripts to {path}")
    print("\nNext steps: uv run python chunk.py && uv run python embed.py")


if __name__ == "__main__":
    main()
