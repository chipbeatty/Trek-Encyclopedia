import json
import math


def load_embeddings(filepath="data/embeddings.json"):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} episode embeddings")
    return data


def cosine_similarity(vec_a, vec_b):
    # Cosine similarity measures the angle between two vectors
    # 1.0 = identical direction (very similar meaning)
    # 0.0 = perpendicular (unrelated)
    # -1.0 = opposite directions
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def search(query_embedding, episodes, top_k=5):
    # Score every episode against the query
    scored = []
    for ep in episodes:
        score = cosine_similarity(query_embedding, ep["embedding"])
        scored.append((score, ep))

    # Sort by score descending and return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def display_results(results):
    print("\n--- Top Matches ---")
    for rank, (score, ep) in enumerate(results, 1):
        print(f"\n#{rank} (score: {score:.4f})")
        print(f"  S{ep['season']}E{ep['episode']} - {ep['title']}")
        print(f"  {ep['synopsis'][:120]}...")


# --- Test it directly ---
if __name__ == "__main__":
    import os
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    episodes = load_embeddings()

    # Test queries — try changing these!
    test_queries = [
        "Borg attack the Enterprise",
        "Data tries to understand human emotions",
        "Picard is captured and tortured",
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Query: '{query}'")

        # Embed the query using the same model
        response = client.embeddings.create(model="text-embedding-3-small", input=query)
        query_embedding = response.data[0].embedding

        results = search(query_embedding, episodes, top_k=3)
        display_results(results)
