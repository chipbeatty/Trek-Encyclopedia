import urllib.request
import json
import csv
import os


def fetch_json(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "star-trek-rag-project/1.0"}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    # TVMaze ID for Star Trek: TNG is 1372
    # This single endpoint returns ALL episodes with summaries
    print("Fetching all TNG episodes from TVMaze API...")

    episodes_data = fetch_json("https://api.tvmaze.com/shows/491/episodes")

    print(f"Found {len(episodes_data)} episodes")

    # Clean HTML tags from summaries — TVMaze wraps them in <p> tags
    import re

    def strip_html(text):
        if not text:
            return ""
        return re.sub(r"<[^>]+>", "", text).strip()

    episodes = []
    for ep in episodes_data:
        episodes.append(
            {
                "season": ep.get("season", ""),
                "episode": ep.get("number", ""),
                "title": ep.get("name", ""),
                "airdate": ep.get("airdate", ""),
                "synopsis": strip_html(ep.get("summary", "")),
            }
        )

    # Preview first 3
    print("\nFirst 3 episodes:")
    for ep in episodes[:3]:
        print(f"  S{ep['season']}E{ep['episode']} - {ep['title']}")
        print(f"  Synopsis: {ep['synopsis'][:100]}...")
        print()

    # Check synopsis coverage
    with_synopsis = sum(1 for ep in episodes if ep["synopsis"])
    print(f"Episodes with synopsis: {with_synopsis}/{len(episodes)}")

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    fieldnames = ["season", "episode", "title", "airdate", "synopsis"]

    with open("data/tng_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episodes)

    print("\n✅ Saved to data/tng_episodes.csv")


if __name__ == "__main__":
    main()
