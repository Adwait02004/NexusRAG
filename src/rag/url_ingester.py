import re
import urllib.parse
from pathlib import Path
from typing import Tuple
import requests
from bs4 import BeautifulSoup

class URLIngester:
    """
    Universal Web & Article Ingestion Engine.
    Fetches, cleans, and converts arbitrary web pages, documentation, Reddit threads,
    or articles into standardized text assets for FAISS & BM25 indexing.
    """
    def __init__(self, user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"):
        self.headers = {"User-Agent": user_agent}

    def _slugify(self, text: str) -> str:
        """Create a safe filesystem filename from an article title."""
        text = re.sub(r"[^\w\s-]", "", text).strip().lower()
        slug = re.sub(r"[-\s]+", "_", text)
        return slug[:45] if slug else "web_document"

    def fetch_and_parse(self, url: str) -> Tuple[str, str]:
        """
        Download and extract clean textual body and title from a web URL.
        Returns: (title, clean_content)
        """
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: '{url}'. Please include http:// or https://")

        resp = requests.get(url, headers=self.headers, timeout=12)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        else:
            title = parsed.netloc + parsed.path.rstrip("/").split("/")[-1]

        # Strip clutter
        for tag in soup(["script", "style", "nav", "aside", "header", "footer", "form", "noscript"]):
            tag.decompose()

        # Prioritize main article container if available
        main_container = soup.find("article") or soup.find("main") or soup.find("div", {"role": "main"}) or soup

        paragraphs = []
        for elem in main_container.find_all(["p", "h2", "h3", "li"]):
            text = elem.get_text().strip()
            if len(text) > 25:
                paragraphs.append(text)

        if not paragraphs:
            # Fallback to general text extraction
            raw_text = soup.get_text(separator="\n", strip=True)
            paragraphs = [line for line in raw_text.splitlines() if len(line.strip()) > 35]

        clean_text = "\n\n".join(paragraphs)
        if len(clean_text) < 80:
            raise ValueError("The target URL did not yield sufficient readable text content.")

        return title, clean_text

    def ingest_url_to_file(self, url: str, output_dir: Path) -> Tuple[Path, str, int]:
        """
        Fetch URL and save as an indexed text file.
        Returns: (file_path, title, char_count)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        title, content = self.fetch_and_parse(url)
        slug = self._slugify(title)
        file_path = output_dir / f"{slug}.txt"

        # If file exists, append timestamp to make unique
        if file_path.exists():
            import time
            file_path = output_dir / f"{slug}_{int(time.time())}.txt"

        formatted_document = f"# {title}\n\n**Source URL:** {url}\n\n---\n\n{content}\n"
        file_path.write_text(formatted_document, encoding="utf-8")

        return file_path, title, len(content)
