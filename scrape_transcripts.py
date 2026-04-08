"""
scrape_transcripts.py
Fetches all TNG episode transcripts from chakoteya.net and saves them to
data/transcripts.json, merging with existing data/episodes.json if present.

Usage:
    uv run python scrape_transcripts.py
    uv run python scrape_transcripts.py --delay 2.0   # be polite to the server
    uv run python scrape_transcripts.py --out data/transcripts.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_URL = "http://www.chakoteya.net/NextGen"
INDEX_URL = f"{BASE_URL}/episodes.htm"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TNG-RAG-bot/1.0; personal research project)"
    )
}


# ---------------------------------------------------------------------------
# Step 1 – Discover episode IDs from the index page
# ---------------------------------------------------------------------------


def fetch_episode_ids(client: httpx.Client) -> list[dict]:
    """
    Parse the Chakoteya episode index and return a list of dicts:
      {"chakoteya_id": "116", "title": "11001001"}
    The index page links look like:  <a href="116.htm">11001001</a>
    """
    resp = client.get(INDEX_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    episodes = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Match bare numeric filenames: 101.htm, 245.htm, etc.
        m = re.fullmatch(r"(\d+)\.htm", href)
        if m:
            episodes.append(
                {
                    "chakoteya_id": m.group(1),
                    "title": a.get_text(strip=True),
                }
            )

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for ep in episodes:
        key = ep["chakoteya_id"]
        if key not in seen:
            seen.add(key)
            unique.append(ep)

    return unique


# ---------------------------------------------------------------------------
# Step 2 – Parse a single transcript page
# ---------------------------------------------------------------------------


def parse_transcript_page(html: str, chakoteya_id: str) -> dict:
    """
    Returns a dict with keys:
      chakoteya_id, title, stardate, airdate, transcript, scenes
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- metadata -----------------------------------------------------------
    # Title is typically the first <h3> or the <title> tag
    title = ""
    h3 = soup.find("h3")
    if h3:
        title = h3.get_text(strip=True)
    else:
        tag = soup.find("title")
        if tag:
            # "The Next Generation Transcripts - Encounter at Farpoint"
            raw = tag.get_text(strip=True)
            title = raw.split(" - ")[-1].strip() if " - " in raw else raw

    # Stardate and airdate live in the first paragraph-ish block
    full_text = soup.get_text(separator="\n")
    stardate_match = re.search(r"Stardate[:\s]+([\d.]+)", full_text)
    airdate_match = re.search(
        r"Original Airdate[:\s]+(\d{1,2}\s+\w+,?\s+\d{4})", full_text
    )
    stardate = stardate_match.group(1).strip() if stardate_match else ""
    airdate = airdate_match.group(1).strip() if airdate_match else ""

    # --- transcript ---------------------------------------------------------
    # Chakoteya pages put the transcript in the main <body> as plain text /
    # simple tags — no wrapper <div class="transcript"> etc.
    # Strategy: grab all text, strip nav boilerplate at top/bottom.
    lines = full_text.splitlines()
    transcript_lines = []
    in_transcript = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Heuristic: transcript starts after the airdate line
        if not in_transcript:
            if airdate and airdate in stripped:
                in_transcript = True
            elif stardate and stardate in stripped:
                in_transcript = True
            continue

        # Stop at common footer boilerplate
        if re.search(
            r"(Star Trek.+transcript|chakoteya\.net|^<|^\[end\])",
            stripped,
            re.IGNORECASE,
        ):
            break

        transcript_lines.append(stripped)

    transcript = "\n".join(transcript_lines)

    # --- scenes -------------------------------------------------------------
    # Scene directions appear as lines like:  [Bridge] or (Picard enters)
    # Split into scenes on capitalised bracketed headings: [BRIDGE]
    scenes = _split_into_scenes(transcript)

    return {
        "chakoteya_id": chakoteya_id,
        "title": title,
        "stardate": stardate,
        "airdate": airdate,
        "transcript": transcript,
        "scenes": scenes,
        "char_count": len(transcript),
        "word_count": len(transcript.split()),
    }


def _split_into_scenes(transcript: str) -> list[dict]:
    """
    Split transcript into scenes using bracketed uppercase headings like
    [BRIDGE] or [CAPTAIN'S READY ROOM]. Falls back to a single scene.
    Returns list of {"location": str, "text": str}.
    """
    # Pattern: line that is solely a bracketed label in all-caps / title case
    scene_re = re.compile(r"^\[([A-Z][A-Z '\-]+)\]$", re.MULTILINE)
    boundaries = [(m.start(), m.group(1)) for m in scene_re.finditer(transcript)]

    if not boundaries:
        return [{"location": "UNKNOWN", "text": transcript}]

    scenes = []
    for i, (start, location) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(transcript)
        # Skip the heading line itself
        scene_text = transcript[start:end]
        scene_text = scene_re.sub("", scene_text, count=1).strip()
        scenes.append({"location": location, "text": scene_text})

    return scenes


# ---------------------------------------------------------------------------
# Step 3 – Merge with existing episodes.json
# ---------------------------------------------------------------------------


def load_existing_episodes(episodes_path: Path) -> dict[str, dict]:
    """
    Load data/episodes.json (from scrape.py / TVMaze) as a dict keyed by
    episode title (lower-stripped) for fuzzy matching.
    """
    if not episodes_path.exists():
        return {}
    with episodes_path.open() as f:
        episodes = json.load(f)
    return {ep.get("name", "").lower().strip(): ep for ep in episodes}


def merge_transcript(tvmaze_ep: dict, transcript_data: dict) -> dict:
    """Add transcript fields onto a TVMaze episode dict."""
    merged = dict(tvmaze_ep)
    merged["chakoteya_id"] = transcript_data["chakoteya_id"]
    merged["stardate"] = transcript_data["stardate"]
    merged["transcript"] = transcript_data["transcript"]
    merged["scenes"] = transcript_data["scenes"]
    merged["transcript_word_count"] = transcript_data["word_count"]
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Scrape TNG transcripts from chakoteya.net"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between requests (default: 1.5)",
    )
    parser.add_argument(
        "--out",
        default="data/transcripts.json",
        help="Output file path (default: data/transcripts.json)",
    )
    parser.add_argument(
        "--episodes",
        default="data/episodes.json",
        help="Existing TVMaze episodes file to merge with",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N episodes (0 = all, useful for testing)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-fetched IDs
    existing_transcripts: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open() as f:
            for ep in json.load(f):
                if "chakoteya_id" in ep:
                    existing_transcripts[ep["chakoteya_id"]] = ep
        print(f"Resuming — {len(existing_transcripts)} episodes already saved.")

    tvmaze_by_title = load_existing_episodes(Path(args.episodes))

    results: list[dict] = list(existing_transcripts.values())

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        # Discover episode IDs from the index
        print("Fetching episode index…")
        episode_stubs = fetch_episode_ids(client)
        print(f"Found {len(episode_stubs)} episodes on Chakoteya.")

        if args.limit:
            episode_stubs = episode_stubs[: args.limit]

        for i, stub in enumerate(episode_stubs, 1):
            cid = stub["chakoteya_id"]

            if cid in existing_transcripts:
                print(
                    f"  [{i}/{len(episode_stubs)}] {stub['title']} — already saved, skipping."
                )
                continue

            url = f"{BASE_URL}/{cid}.htm"
            try:
                resp = client.get(url, headers=HEADERS)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(
                    f"  [{i}/{len(episode_stubs)}] {stub['title']} — HTTP {e.response.status_code}, skipping."
                )
                continue
            except httpx.RequestError as e:
                print(
                    f"  [{i}/{len(episode_stubs)}] {stub['title']} — request error: {e}, skipping."
                )
                continue

            data = parse_transcript_page(resp.text, cid)
            # Use stub title as fallback if parser got nothing
            if not data["title"]:
                data["title"] = stub["title"]

            # Try to merge with TVMaze data
            title_key = data["title"].lower().strip()
            if title_key in tvmaze_by_title:
                data = merge_transcript(tvmaze_by_title[title_key], data)

            results.append(data)
            word_count = data.get("word_count") or data.get("transcript_word_count", 0)
            print(
                f"  [{i}/{len(episode_stubs)}] {data['title']} ({word_count:,} words)"
            )

            # Checkpoint save every 10 episodes
            if i % 10 == 0:
                _save(out_path, results)
                print(f"  → Checkpoint saved ({len(results)} episodes).")

            time.sleep(args.delay)

    _save(out_path, results)
    total_words = sum(
        ep.get("word_count") or ep.get("transcript_word_count", 0) for ep in results
    )
    print(f"\nDone. {len(results)} transcripts saved to {out_path}")
    print(f"Total transcript corpus: {total_words:,} words")


def _save(path: Path, data: list[dict]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
