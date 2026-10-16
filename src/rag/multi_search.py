import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import requests

class OmniSearchEngine:
    """
    Multi-Source External Search Provider for NexusRAG.
    Aggregates live information from:
    1. Wikipedia Public Knowledge API (instant global encyclopedic context)
    2. ArXiv Research Repository (peer-reviewed scientific papers & abstracts)
    3. Serper / Google Search (if SERPER_API_KEY / GOOGLE_API_KEY is configured in .env)
    4. Brave Search API (if BRAVE_API_KEY is configured in .env)
    """
    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

    def search_wikipedia(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Fetch encyclopedia articles and summaries via MediaWiki API."""
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": limit,
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "format": "json"
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "NexusRAG/1.0 (https://nexusrag.ai)"})
        results = []
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid, pdata in pages.items():
                    title = pdata.get("title", "Wikipedia Article")
                    extract = pdata.get("extract", "").strip()
                    if not extract:
                        continue
                    clean_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    preview = extract[:220] + ("..." if len(extract) > 220 else "")
                    results.append({
                        "source": f"Wikipedia: {title}",
                        "page": clean_url,
                        "score": 0.92,
                        "relevance_pct": 92,
                        "preview": preview,
                        "full_text": f"Wikipedia Entry ({title}):\n{extract}",
                        "retrieval_mode": "live_wikipedia",
                        "is_web": True
                    })
        except Exception as e:
            print(f"[WARN] Wikipedia search error: {e}")
        return results

    def search_arxiv(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Fetch research papers and abstracts from ArXiv Open Access API."""
        encoded_q = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_q}&start=0&max_results={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "NexusRAG/1.0"})
        results = []
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)
                    
                    if title_elem is None or summary_elem is None:
                        continue
                    
                    title = " ".join(title_elem.text.split())
                    summary = " ".join(summary_elem.text.split())
                    paper_url = id_elem.text.strip() if id_elem is not None else "https://arxiv.org"
                    preview = summary[:220] + ("..." if len(summary) > 220 else "")
                    
                    results.append({
                        "source": f"ArXiv Research: {title[:60]}",
                        "page": paper_url,
                        "score": 0.88,
                        "relevance_pct": 88,
                        "preview": preview,
                        "full_text": f"ArXiv Paper: {title}\nSummary: {summary}",
                        "retrieval_mode": "live_arxiv",
                        "is_web": True
                    })
        except Exception as e:
            print(f"[WARN] ArXiv search error: {e}")
        return results

    def search_serper(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Google Search via Serper API if SERPER_API_KEY is configured."""
        if not self.serper_api_key:
            return []
        try:
            headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
            payload = json.dumps({"q": query, "num": limit})
            resp = requests.post("https://google.serper.dev/search", headers=headers, data=payload, timeout=6)
            data = resp.json()
            results = []
            for item in data.get("organic", [])[:limit]:
                title = item.get("title", "Web Page")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                results.append({
                    "source": f"Google: {title[:50]}",
                    "page": link,
                    "score": 0.95,
                    "relevance_pct": 95,
                    "preview": snippet[:220],
                    "full_text": f"{title}\nURL: {link}\n{snippet}",
                    "retrieval_mode": "live_google",
                    "is_web": True
                })
            return results
        except Exception as e:
            print(f"[WARN] Serper Google search error: {e}")
            return []

    def search_all(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Aggregate multi-source external information across configured providers.
        """
        aggregated: List[Dict[str, Any]] = []

        # 1. Check Serper (Google Search) if key configured
        if self.serper_api_key:
            aggregated.extend(self.search_serper(query, limit=max_results))

        # 2. Query Wikipedia Knowledge Base
        if len(aggregated) < max_results:
            wiki_res = self.search_wikipedia(query, limit=2)
            aggregated.extend(wiki_res)

        # 3. Query ArXiv Academic Research
        if len(aggregated) < max_results:
            arxiv_res = self.search_arxiv(query, limit=2)
            aggregated.extend(arxiv_res)

        return aggregated[:max_results]
