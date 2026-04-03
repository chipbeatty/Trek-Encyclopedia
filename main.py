import os
import sys
from rag import load_embeddings, ask, display


def main():
    print("=" * 60)
    print("  🖖 Star Trek TNG Encyclopedia")
    print("  Powered by RAG + OpenAI")
    print("=" * 60)
    print("\nLoading episode knowledge base...")

    episodes = load_embeddings()
    print(f"✅ {len(episodes)} episodes loaded and ready.\n")
    print("Ask me anything about TNG! (type 'quit' to exit)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nLive long and prosper. 🖖")
            sys.exit(0)

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("\nLive long and prosper. 🖖")
            break

        print("\nSearching the knowledge base...\n")

        result = ask(question, episodes)

        print(f"TNG Bot: {result['answer']}")

        print("\n📺 Episodes referenced:")
        for ep in result["sources"][:3]:
            print(f"   S{ep['season']}E{ep['episode']} - {ep['title']}")

        print()


if __name__ == "__main__":
    main()
