from typing import List, Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import os

# Suppress noisy symlink warnings and parallel tokenizer deadlocks on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        # Fast local load: avoid remote HF Hub roundtrips if model is already cached
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} document sections into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)