import os
import json
import csv
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_episodes(filepath="data/tng_episodes.csv"):
    episodes = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(row)
    print(f"Loaded {len(episodes)} episodes from CSV")
    return episodes


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def build_embeddings(episodes):
    results = []
    total = len(episodes)

    for i, ep in enumerate(episodes):
        # Build a rich text chunk combining key fields
        # This is what gets embedded — the more context the better
        text = (
            f"Title: {ep['title']}. "
            f"Season {ep['season']}, Episode {ep['episode']}. "
            f"Air date: {ep['airdate']}. "
            f"Synopsis: {ep['synopsis']}"
        )

        embedding = get_embedding(text)

        results.append(
            {
                "season": ep["season"],
                "episode": ep["episode"],
                "title": ep["title"],
                "airdate": ep["airdate"],
                "synopsis": ep["synopsis"],
                "embedding": embedding,
            }
        )

        # Progress indicator every 10 episodes
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Embedded {i + 1}/{total}: {ep['title']}")

        # Small delay to be polite to the API
        time.sleep(0.05)

    return results


def main():
    # Safety check — don't re-embed if file already exists
    output_path = "data/embeddings.json"
    if os.path.exists(output_path):
        print("⚠️  embeddings.json already exists!")
        print("Delete it manually if you want to re-embed.")
        return

    episodes = load_episodes()

    print(f"\nEmbedding {len(episodes)} episodes with text-embedding-3-small...")
    print("This will cost roughly $0.001 — basically free.\n")

    results = build_embeddings(episodes)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"\n✅ Saved {len(results)} embeddings to {output_path}")
    print(f"Each embedding has {len(results[0]['embedding'])} dimensions")


if __name__ == "__main__":
    main()
