import urllib.request
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_episode(title):
    # Memory Alpha URL pattern: title with spaces replaced by underscores + _(episode)
    slug = title.strip().replace(" ", "_")
    url = f"https://memory-alpha.fandom.com/wiki/{slug}_(episode)"
    print(f"Fetching: {url}")

    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        html = r.read().decode("utf-8")

    soup = BeautifulSoup(html, "lxml")

    # Memory Alpha puts the main article content in div.mw-parser-output
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return None

    # Grab all paragraphs — skip empty ones and nav boilerplate
    paragraphs = []
    for p in content.find_all("p"):
        text = p.get_text(separator=" ").strip()
        if len(text) > 50:  # skip tiny/empty paragraphs
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


if __name__ == "__main__":
    text = fetch_episode("Encounter at Farpoint")
    if text:
        print(f"\nFetched {len(text)} characters")
        print(f"\nFirst 500 chars:\n{text[:500]}")
    else:
        print("Failed to fetch page")
