import pickle
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2


class PDFVectorDB:
    """Simple FAISS-based vector database for PDF documents."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.texts: List[str] = []

    def build_from_pdf(self, pdf_path: str) -> None:
        """Read a PDF file, create embeddings and build a FAISS index."""
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            pages = [page.extract_text() or "" for page in reader.pages]

        segments: List[str] = []
        for page_text in pages:
            segments.extend([seg.strip() for seg in page_text.split("\n") if seg.strip()])

        if not segments:
            raise ValueError("PDF contains no extractable text")

        embeddings = self.model.encode(segments)
        dims = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dims)
        self.index.add(np.array(embeddings))
        self.texts = segments

    def save(self, path: str) -> None:
        """Save the index and associated texts to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_texts.pkl", "wb") as fh:
            pickle.dump(self.texts, fh)

    def load(self, path: str) -> None:
        """Load the index and texts from disk."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_texts.pkl", "rb") as fh:
            self.texts = pickle.load(fh)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the closest segments to the query text."""
        if self.index is None:
            raise ValueError("Index not built or loaded")
        vector = self.model.encode([text])
        distances, indices = self.index.search(np.array(vector), top_k)
        return [
            (self.texts[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
