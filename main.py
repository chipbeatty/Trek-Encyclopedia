import sys

from rag import ask


def main():
    print("=" * 60)
    print("  Star Trek TNG Encyclopedia")
    print("  Powered by RAG + ChromaDB + OpenAI")
    print("=" * 60)
    print("\nAsk me anything about TNG! (type 'quit' to exit)\n")

    history: list[dict] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nLive long and prosper.")
            sys.exit(0)

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nLive long and prosper.")
            break

        result = ask(question, history=history)
        history = result["history"]

        print(f"\nTNG Bot: {result['answer']}")
        print("\nSources:")
        seen: set[str] = set()
        for chunk in result["chunks"][:3]:
            m = chunk["metadata"]
            if m["episode_id"] not in seen:
                print(f"  S{m['season']:02d}E{m['episode']:02d} - {m['title']} [{m['scene_location']}]")
                seen.add(m["episode_id"])
        print()


if __name__ == "__main__":
    main()
