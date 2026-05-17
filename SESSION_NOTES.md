# Session Notes

## 1. Season/Episode metadata always S00E00 (chunk.py)
`transcripts.json` never received season/episode numbers because `scrape_transcripts.py` looked for `episodes.json` (TVMaze format) but only `tng_episodes.csv` existed. Fixed by loading the CSV directly in `chunk.py` and joining on title. Added `_normalize_title()` to reconcile ~20 formatting inconsistencies between Chakoteya and TVMaze (embedded `\r\n`, British spelling, roman vs arabic part numbers). Two explicit aliases handled the remaining cases where Chakoteya labels a part number the CSV omits.

## 2. All scene locations showing as UNKNOWN (scrape_transcripts.py)
The scene-splitting regex required ALL CAPS (`[BRIDGE]`) but Chakoteya uses Title Case (`[Bridge]`). One-character fix: changed `[A-Z][A-Z '\-]+` to `[A-Z][^\]]*`. Added `fix_scenes.py` to re-extract scenes from existing `transcripts.json` without re-scraping. Also documented that `embed.py`'s resume logic skips chunks by ID — so after re-chunking, the old UNKNOWN chunks persisted alongside new ones until the collection was manually deleted and re-embedded.

## 3. Synopsis index not surfacing concept-level matches (search.py)
The synopsis collection correctly identified the right episodes (The Inner Light, Darmok) but their best chunks scored worse on cosine distance than unrelated episodes, so they were filtered out after merging. Fixed by guaranteeing inclusion of the best chunk from each synopsis-matched episode regardless of distance, then filling remaining slots with direct chunk results.
