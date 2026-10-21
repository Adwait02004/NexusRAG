import os
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

class OmniSearchEngine:
    """
    Multi-Source External Search Provider for NexusRAG.
    Aggregates live information from:
    1. Wikipedia Public Knowledge API (instant global encyclopedic context & visuals)
    2. ArXiv Research Repository (peer-reviewed scientific papers & abstracts)
    3. Serper / Google Search & Images (if SERPER_API_KEY is configured in .env)
    4. Brave Search API (if BRAVE_API_KEY is configured in .env)
    """
    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

    @staticmethod
    def is_directive_or_meta_query(query: str) -> bool:
        """Detect conversational instructions, formatting demands, or system directives."""
        q = (query or "").strip().lower()
        if not q:
            return True
        if q in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "cool", "great", "sure", "bye", "yes", "no"}:
            return True
        patterns = [
            r"\b(?:do\s+not|don't|stop|hide|ignore|remove|delete|omit)\s+(?:show|include|use|display|cite|mention|add)\b",
            r"\b(?:henceforth|from\s+now\s+on)\b",
            r"^(?:please\s+)?(?:do\s+not|don't|stop|hide|ignore|remove)\b",
            r"\b(?:in\s+your\s+response|in\s+the\s+answer|in\s+future\s+responses)\b",
            r"^(?:format\s+as|rewrite|rephrase|summarize|translate|shorten|bullet\s+points|table\s+only)\b"
        ]
        return any(bool(re.search(p, q, re.IGNORECASE)) for p in patterns)

    @staticmethod
    def is_entity_relevant(query: str, title: str, extract: str = "") -> bool:
        """Strict semantic & token overlap guardrail to reject random un-related Wikipedia results."""
        if not title or not query:
            return False
        q_norm = query.lower()
        t_norm = title.lower()
        if q_norm in t_norm or t_norm in q_norm:
            return True

        stopwords = {
            "the", "and", "for", "with", "from", "about", "what", "which",
            "who", "whom", "this", "that", "song", "album", "music", "video",
            "days", "tell", "show", "response", "please", "also", "glorious",
            "latest", "release", "year", "info", "details", "real", "name",
            "source", "web", "doc", "page"
        }
        q_words = set(re.findall(r"\b[a-z0-9]{3,}\b", q_norm)) - stopwords
        t_words = set(re.findall(r"\b[a-z0-9]{3,}\b", t_norm)) - stopwords
        if not q_words:
            return False

        if len(q_words.intersection(t_words)) >= 1:
            return True

        if extract:
            e_norm = extract[:350].lower()
            if q_norm in e_norm:
                return True
            if len(q_words) >= 2 and all(w in e_norm for w in q_words):
                return True

        return False

    @staticmethod
    def is_genai_image_query(query: str) -> bool:
        """Detect whether a query specifically asks to generate/create/draw an AI image."""
        if not query:
            return False
        q = query.strip()
        gen_pattern = (
            r"\b(generate|create|draw|paint|render|make)\s+(?:an?\s+|one\s+|a\s+|some\s+)?(?:ai\s+)?(?:images?|pictures?|photos?|artwork|illustrations?|portraits?|art|visuals?)\b|"
            r"^(?:please\s+)?(?:draw|paint|render)\s+[a-zA-Z0-9\s]+|"
            r"\b(?:ai\s+generated|flux|sdxl|dall-e|midjourney|generative\s+ai|genai)\b"
        )
        return bool(re.search(gen_pattern, q, re.IGNORECASE))

    @staticmethod
    def is_scene_or_compositional_prompt(q: str) -> bool:
        """Detect whether a query describes a composite scene, action, or creative concept requiring GenAI generation."""
        if not q:
            return False
        pattern = (
            r"\b(playing\s+with|on\s+a|in\s+a|with\s+|wearing|breathing|flying|running|eating|sitting|standing|"
            r"holding|looking\s+at|surrounded\s+by|riding|floating|swimming|jumping|walking|driving|"
            r"futuristic|cyberpunk|anime|digital\s+art|concept\s+art|neon|fantasy|mythical|space|extraterrestrial|"
            r"photorealistic|hyperrealistic|cinematic|oil\s+painting|watercolor|sketch|drawing|illustration)\b|"
            r"^(?:my|our)\s+(?:dog|cat|house|room|car|family|pet)\b"
        )
        return bool(re.search(pattern, q, re.IGNORECASE))

    @staticmethod
    def is_pure_image_query(query: str) -> bool:
        """Detect whether a query is asking purely for visuals/images/photos/portraits of any entity or thing."""
        if not query:
            return False
        q = query.strip()
        # Direct GenAI verb check
        gen_kw = bool(re.search(
            r"\b(generate|create|draw|paint|render)\s+(?:an?\s+|one\s+|some\s+)?(?:ai\s+)?(?:images?|pictures?|photos?|artwork|art|visuals?)\b|"
            r"^(?:please\s+)?(?:draw|paint|render)\s+",
            q,
            re.IGNORECASE
        ))
        if gen_kw:
            return True

        concept_pattern = r"\b(explain|how\s+does|how\s+do|how\s+to|what\s+is|what\s+are|why\s+is|why\s+do|difference\s+between|compare|describe\s+how|write\s+a|algorithm|mechanism)\b"
        if re.search(concept_pattern, q, re.IGNORECASE):
            return False
            
        has_img_kw = bool(re.search(r"\b(images?|pictures?|photos?|pics?|diagrams?|illustrations?|figures?|visuals?|portraits?|artwork)\b", q, re.IGNORECASE))
        if not has_img_kw:
            return False

        # Any query with an image keyword that is not an explicit concept explanation is a pure image request!
        return True

    @staticmethod
    def generate_ai_image(prompt: str, width: int = 1024, height: int = 1024, model: str = "flux") -> Dict[str, str]:
        """
        Synthesize high-fidelity GenAI visual using Pollinations FLUX / SDXL inference.
        Returns a dictionary with title, image_url, source_url, and caption.
        """
        clean_prompt = prompt.strip()
        encoded = urllib.parse.quote(clean_prompt)
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&model={model}&nologo=true"
        return {
            "title": f"GenAI: {clean_prompt[:40]}",
            "image_url": img_url,
            "source_url": img_url,
            "caption": "GenAI Synthesis • FLUX Neural Engine",
            "is_genai": True
        }

    @staticmethod
    def clean_entity_query(query: str) -> str:
        """Strip conversational filler words, desire commands, and image tokens to extract the exact entity or thing."""
        if not query:
            return ""
        cleaned = query.strip()
        cleaned = re.sub(r"[\?\.\!]+$", "", cleaned).strip()

        # Multi-pass strip of conversational starters and desire prefixes
        prefix_pattern = (
            r"^(?:also\s+)?(?:please\s+)?"
            r"(?:can\s+(?:i|we|you)\s+(?:see|have|get|show\s+me)\s+|could\s+you\s+(?:show|give|send)\s+|"
            r"i\s+(?:want|need)\s+(?:to\s+see\s+)?|(?:i\s+)?need\s+|let\s+me\s+see\s+|tell\s+me\s+about|"
            r"what\s+does\s+[a-zA-Z0-9\s]+look\s+like|search\s+(?:for)?\s+|"
            r"(?:show|display|give|get|find|fetch|provide|send|generate|create|draw|produce|paint|render|see|view)\s+(?:me\s+)?|"
            r"(?:only|just|simply)\s+)+"
        )
        for _ in range(3):
            cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Multi-pass strip of image prefixes and quantifiers (handles "only image of", "1 photo of", etc.)
        img_prefix = (
            r"^(?:the\s+|some\s+|a\s+|an\s+|one\s+|1\s+|single\s+|only\s+|just\s+|all\s+|any\s+|few\s+|several\s+|latest\s+|best\s+|hd\s+|hq\s+|4k\s+|high\s+resolution\s+|high\s+quality\s+)?"
            r"(?:ai\s+)?"
            r"(?:images?|pictures?|photos?|pics?|diagrams?|illustrations?|figures?|visuals?|portraits?|artwork)\s+"
            r"(?:of|for|about|showing|depicting)?\s*"
        )
        for _ in range(3):
            cleaned = re.sub(img_prefix, "", cleaned, flags=re.IGNORECASE).strip()

        # Strip trailing image keywords (e.g. 'Virat Kohli image' -> 'Virat Kohli')
        cleaned = re.sub(
            r"\s+(?:images?|pictures?|photos?|pics?|diagrams?|illustrations?|figures?|visuals?|portraits?|artwork)$",
            "", cleaned, flags=re.IGNORECASE
        ).strip()

        # Strip leading prepositions, articles, or remnant qualifiers
        cleaned = re.sub(r"^(?:of|for|about|an?|the|only|just|single)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned or query

    def search_wikipedia(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Fetch encyclopedia articles, summaries, and high-res entity visuals via MediaWiki API."""
        if self.is_directive_or_meta_query(query):
            return []

        entity_q = self.clean_entity_query(query)
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": entity_q,
            "gsrlimit": str(limit + 4),
            "prop": "extracts|pageimages",
            "exintro": "1",
            "explaintext": "1",
            "piprop": "thumbnail|original",
            "pithumbsize": "600",
            "format": "json"
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "NexusRAG/1.0 (https://nexusrag.ai)"})
        results = []
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                sorted_pages = sorted(pages.values(), key=lambda p: p.get("index", 999))
                for pdata in sorted_pages:
                    title = pdata.get("title", "Wikipedia Article")
                    extract = pdata.get("extract", "").strip()
                    if not extract:
                        continue
                    if not self.is_entity_relevant(entity_q, title, extract):
                        continue

                    clean_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    preview = extract[:220] + ("..." if len(extract) > 220 else "")
                    thumb_url = pdata.get("thumbnail", {}).get("source") or pdata.get("original", {}).get("source")
                    
                    results.append({
                        "source": f"Wikipedia: {title}",
                        "page": clean_url,
                        "score": 0.92,
                        "relevance_pct": 92,
                        "preview": preview,
                        "full_text": f"Wikipedia Entry ({title}):\n{extract}",
                        "retrieval_mode": "live_wikipedia",
                        "is_web": True,
                        "image_url": thumb_url
                    })
                    if len(results) >= limit:
                        break
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

    def fetch_relevant_images(
        self,
        query: str,
        limit: int = 3,
        data_dir: Optional[Path] = None
    ) -> List[Dict[str, str]]:
        """
        Fetch relevant high-resolution entity visuals (portraits, diagrams, figures)
        from local data directories, Google Serper Images, and Wikipedia.
        Returns a list of dicts: [{"title": ..., "image_url": ..., "source_url": ..., "caption": ...}]
        """
        if self.is_directive_or_meta_query(query):
            return []

        images: List[Dict[str, str]] = []
        cleaned_entity = self.clean_entity_query(query)

        # 0. GenAI Synthesis for creative prompts, descriptive scenes, or explicit generation requests
        is_genai = self.is_genai_image_query(query)
        is_scene = self.is_scene_or_compositional_prompt(cleaned_entity)
        if is_genai or is_scene:
            genai_img = self.generate_ai_image(cleaned_entity)
            images.append(genai_img)
            if len(images) >= limit:
                return images

        # 1. Local Image Matching (if images exist in data/images)
        if data_dir:
            images_dir = Path(data_dir) / "images"
            if images_dir.exists() and images_dir.is_dir():
                clean_tokens = [t.lower() for t in re.split(r"\W+", cleaned_entity) if len(t) > 2]
                for img_file in images_dir.iterdir():
                    if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                        fname_clean = img_file.stem.lower()
                        if any(tok in fname_clean for tok in clean_tokens):
                            images.append({
                                "title": img_file.stem.replace("_", " ").title(),
                                "image_url": str(img_file),
                                "source_url": "#",
                                "caption": "Local Knowledge Asset"
                            })
                            if len(images) >= limit:
                                return images

        # 2. Check Serper Images API if key configured
        if self.serper_api_key:
            try:
                headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
                payload = json.dumps({"q": cleaned_entity, "num": limit + 3})
                resp = requests.post("https://google.serper.dev/images", headers=headers, data=payload, timeout=5)
                data = resp.json()
                for item in data.get("images", [])[:limit + 3]:
                    img_url = item.get("imageUrl")
                    img_title = item.get("title", cleaned_entity)
                    if img_url and self.is_entity_relevant(cleaned_entity, img_title):
                        images.append({
                            "title": img_title,
                            "image_url": img_url,
                            "source_url": item.get("link", item.get("domain", "")),
                            "caption": "Google Web Search"
                        })
                        if len(images) >= limit:
                            return images
            except Exception as e:
                print(f"[WARN] Serper image search error: {e}")

        # 3. Wikipedia MediaWiki pageimages API (high-resolution encyclopedic portraits & diagrams)
        try:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": cleaned_entity,
                "gsrlimit": str(limit + 5),
                "prop": "extracts|pageimages",
                "exintro": "1",
                "explaintext": "1",
                "piprop": "thumbnail|original",
                "pithumbsize": "1200",
                "format": "json"
            }
            url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "NexusRAG/1.0 (https://nexusrag.ai)"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                sorted_pages = sorted(pages.values(), key=lambda p: p.get("index", 999))
                for pdata in sorted_pages:
                    thumb_url = pdata.get("thumbnail", {}).get("source") or pdata.get("original", {}).get("source")
                    title = pdata.get("title", "")
                    extract = pdata.get("extract", "")
                    if thumb_url and title and self.is_entity_relevant(cleaned_entity, title, extract):
                        clean_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                        images.append({
                            "title": title,
                            "image_url": thumb_url,
                            "source_url": clean_url,
                            "caption": "Wikipedia Reference"
                        })
                        if len(images) >= limit:
                            break
        except Exception as e:
            print(f"[WARN] MediaWiki image search error: {e}")

        # 4. Wikimedia Commons API (high-resolution entity photographs, portraits, and diagrams)
        if len(images) < limit:
            search_terms = [cleaned_entity]
            if any(w in query.lower() for w in ["diagram", "chart", "schema", "architecture", "flowchart"]):
                search_terms.insert(0, f"{cleaned_entity} diagram")
            elif any(w in query.lower() for w in ["photo", "portrait", "picture", "visual", "image"]):
                search_terms.append(f"{cleaned_entity} photo")

            for sterm in search_terms:
                if len(images) >= limit:
                    break
                try:
                    commons_params = {
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": sterm,
                        "gsrnamespace": "6",
                        "gsrlimit": str(limit + 5),
                        "prop": "imageinfo",
                        "iiprop": "url|size|mime",
                        "iiurlwidth": "1280",
                        "format": "json"
                    }
                    c_url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(commons_params)
                    c_req = urllib.request.Request(c_url, headers={"User-Agent": "NexusRAG/1.0 (https://nexusrag.ai)"})
                    with urllib.request.urlopen(c_req, timeout=5) as resp:
                        c_data = json.loads(resp.read().decode("utf-8"))
                        c_pages = c_data.get("query", {}).get("pages", {})
                        for cp in sorted(c_pages.values(), key=lambda p: p.get("index", 999)):
                            info_list = cp.get("imageinfo", [])
                            if not info_list:
                                continue
                            info = info_list[0]
                            img_url = info.get("thumburl") or info.get("url")
                            page_url = info.get("descriptionurl", "https://commons.wikimedia.org")
                            raw_c_title = cp.get("title", "")
                            clean_c_title = re.sub(r"^File:", "", raw_c_title).replace("_", " ")
                            clean_c_title = re.sub(r"\.(svg|png|jpg|jpeg|gif|webp)$", "", clean_c_title, flags=re.IGNORECASE).strip()

                            # Filter out signatures, logos, stamps, coat of arms
                            if re.search(r"\b(signature|autograph|sign|icon|logo|coat of arms|stamp)\b", clean_c_title, re.IGNORECASE):
                                continue

                            # Skip very low resolution / thumbnail icons
                            w = info.get("thumbwidth") or info.get("width", 800)
                            h = info.get("thumbheight") or info.get("height", 800)
                            if w < 160 or h < 160:
                                continue

                            # Shorten verbose archive titles cleanly
                            if len(clean_c_title) > 45:
                                shortened = re.sub(r"^(?:The\s+President[^\n,]+,\s*)", "", clean_c_title).strip()
                                if len(shortened) < len(clean_c_title):
                                    clean_c_title = shortened
                                if len(clean_c_title) > 45:
                                    clean_c_title = clean_c_title[:42] + "..."

                            if img_url and not any(existing.get("image_url") == img_url for existing in images):
                                caption_label = "Wikimedia Commons" if "photo" not in clean_c_title.lower() else "Verified Photo"
                                images.append({
                                    "title": clean_c_title,
                                    "image_url": img_url,
                                    "source_url": page_url,
                                    "caption": caption_label
                                })
                                if len(images) >= limit:
                                    break
                except Exception as e_c:
                    print(f"[WARN] Wikimedia Commons image search error: {e_c}")

        # 5. GenAI Integration: If specifically requested, synthesize with FLUX neural engine!
        if self.is_genai_image_query(query):
            genai_img = self.generate_ai_image(cleaned_entity)
            images.insert(0, genai_img)
        elif not images and cleaned_entity:
            # Fallback: Synthesize with GenAI so the user always receives a visual!
            genai_img = self.generate_ai_image(f"{cleaned_entity}, photorealistic, high quality, 8k")
            images.append(genai_img)

        return images

    def search_live_web(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """Live real-time Web Search via DuckDuckGo HTML endpoint without requiring paid API keys."""
        if self.is_directive_or_meta_query(query):
            return []

        cleaned_q = self.clean_entity_query(query)
        # Normalize common transliteration variants (e.g. Barssat -> Barsaat)
        cleaned_q = re.sub(r"\bBarssat\b", "Barsaat", cleaned_q, flags=re.I)

        sub_queries = [cleaned_q]
        if not any(k in cleaned_q.lower() for k in ["release", "date", "year", "details"]):
            sub_queries.append(f"{cleaned_q} release date details")

        results = []
        seen_urls = set()

        try:
            from bs4 import BeautifulSoup
            for sq in sub_queries:
                if len(results) >= limit:
                    break
                data = urllib.parse.urlencode({"q": sq}).encode("utf-8")
                req = urllib.request.Request(
                    "https://html.duckduckgo.com/html/",
                    data=data,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                )
                try:
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        soup = BeautifulSoup(resp.read().decode("utf-8"), "html.parser")
                        for r in soup.find_all("div", class_="result__body"):
                            title_a = r.find("a", class_="result__a")
                            snippet_a = r.find("a", class_="result__snippet") or r.find("div", class_="result__snippet")
                            title = title_a.text.strip() if title_a else "Live Web Reference"
                            snippet = snippet_a.text.strip() if snippet_a else ""
                            href = title_a.get("href", "") if title_a else ""
                            if "uddg=" in href:
                                m = re.search(r"uddg=([^&]+)", href)
                                if m:
                                    href = urllib.parse.unquote(m.group(1))

                            if href and href not in seen_urls and snippet and len(snippet) > 20:
                                seen_urls.add(href)
                                results.append({
                                    "source": f"Web: {title[:50]}",
                                    "page": href,
                                    "score": 0.96,
                                    "relevance_pct": 96,
                                    "preview": snippet[:220] + ("..." if len(snippet) > 220 else ""),
                                    "full_text": f"Title: {title}\nURL: {href}\n{snippet}",
                                    "retrieval_mode": "live_web",
                                    "is_web": True
                                    })
                            if len(results) >= limit:
                                break
                except Exception as e_sq:
                    print(f"[WARN] Error in subquery '{sq}': {e_sq}")
        except Exception as e:
            print(f"[WARN] Live Web search error: {e}")
        return results

    def search_all(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Aggregate multi-source external information across configured providers:
        1. Google Serper (if API key present)
        2. Real-time Live Web Search (free, fast, live up-to-date web results)
        3. Wikipedia Knowledge Base (encyclopedic summaries & entities)
        4. ArXiv Research (restricted to scientific / academic research queries only)
        """
        if self.is_directive_or_meta_query(query):
            return []

        aggregated: List[Dict[str, Any]] = []

        # 1. Check Serper (Google Search) if key configured
        if self.serper_api_key:
            aggregated.extend(self.search_serper(query, limit=max_results))

        # 2. Real-time Live Web Search (free DuckDuckGo live engine)
        if len(aggregated) < max_results:
            needed = max_results - len(aggregated)
            live_res = self.search_live_web(query, limit=needed)
            aggregated.extend(live_res)

        # 3. Query Wikipedia Knowledge Base
        if len(aggregated) < max_results:
            wiki_res = self.search_wikipedia(query, limit=2)
            aggregated.extend(wiki_res)

        # 4. Query ArXiv Academic Research (ONLY for academic/scientific/technical queries)
        is_academic = bool(re.search(
            r"\b(paper|arxiv|research|algorithm|theorem|proof|quantum|physics|equation|mathematics|dataset|benchmark|neural|nlp|transformer|architecture|deep learning|machine learning)\b",
            query,
            re.IGNORECASE
        ))
        if len(aggregated) < max_results and is_academic:
            arxiv_res = self.search_arxiv(query, limit=2)
            aggregated.extend(arxiv_res)

        return aggregated[:max_results]
