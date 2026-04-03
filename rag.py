import os
import json
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Load embeddings once at startup ---
def load_embeddings(filepath="data/embeddings.json"):
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# --- Cosine similarity (same as search.py) ---
def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# --- Search for relevant episodes ---
def find_relevant_episodes(query, episodes, top_k=5):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embedding = response.data[0].embedding

    scored = []
    for ep in episodes:
        score = cosine_similarity(query_embedding, ep["embedding"])
        scored.append((score, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for score, ep in scored[:top_k]]


# --- Build the context block from retrieved episodes ---
def build_context(episodes):
    lines = []
    for ep in episodes:
        lines.append(
            f"- S{ep['season']}E{ep['episode']} '{ep['title']}' "
            f"(aired {ep['airdate']}): {ep['synopsis']}"
        )
    return "\n".join(lines)


# --- The RAG query — search + generate ---
def ask(question, episodes, top_k=5):
    # Step 1: Retrieve relevant episodes
    relevant = find_relevant_episodes(question, episodes, top_k=top_k)
    context = build_context(relevant)

    # Step 2: Build the prompt with retrieved context
    system_prompt = """You are an expert on Star Trek: The Next Generation.
Answer the user's question using ONLY the episode information provided.
If the answer isn't in the provided episodes, say so honestly.
Always reference specific episode titles and seasons in your answer.
Be conversational but accurate."""

    user_prompt = f"""Here are the most relevant TNG episodes for this question:

{context}

Question: {question}"""

    # Step 3: Generate the answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": relevant,
    }


def display(result):
    print(f"\n{'=' * 60}")
    print(f"Q: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources used:")
    for ep in result["sources"]:
        print(f"  - S{ep['season']}E{ep['episode']} {ep['title']}")


# --- Test it ---
if __name__ == "__main__":
    print("Loading embeddings...")
    episodes = load_embeddings()
    print(f"Loaded {len(episodes)} episodes\n")

    test_questions = [
        "What episodes feature the Borg as the main threat?",
        "Which episode deals with Picard being tortured by Cardassians?",
        "Are there any episodes where Data explores what it means to be human?",
    ]

    for question in test_questions:
        result = ask(question, episodes)
        display(result)
