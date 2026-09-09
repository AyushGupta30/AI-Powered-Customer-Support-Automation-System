

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")


def _load_chunks():
    """Load every .txt file in knowledge_base/ and split it into paragraph
    chunks. Very short paragraphs (standalone titles, mostly) get merged
    into the next one, since alone they're low-information and tend to
    crowd out better chunks during ranking."""
    chunks = []
    sources = []
    MIN_CHUNK_LEN = 80

    for filename in sorted(os.listdir(KB_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        merged = []
        buffer = ""
        for p in paragraphs:
            buffer = (buffer + "\n\n" + p).strip() if buffer else p
            if len(buffer) >= MIN_CHUNK_LEN:
                merged.append(buffer)
                buffer = ""
        if buffer:
            # short trailing leftover - just tack it onto the last chunk
            if merged:
                merged[-1] = merged[-1] + "\n\n" + buffer
            else:
                merged.append(buffer)

        for chunk in merged:
            chunks.append(chunk)
            sources.append(filename)

    return chunks, sources


class KnowledgeBaseRetriever:
    """TF-IDF retriever over the knowledge base documents."""

    DOC_TITLES = {
        "company_policy.txt": "company policy refund cancellation account closure compensation escalation",
        "pricing_guide.txt": "pricing guide plans subscription cost",
        "technical_manual.txt": "technical manual troubleshooting errors installation login crashes",
        "faq_document.txt": "frequently asked questions faq",
    }

    def __init__(self):
        self.chunks, self.sources = _load_chunks()

        # prepend each chunk's document-topic keywords when building the
        # vectorizer vocabulary - only for scoring purposes, the original
        # chunk text is what actually gets returned
        vectorize_texts = [
            f"{self.DOC_TITLES.get(src, '')} {chunk}"
            for chunk, src in zip(self.chunks, self.sources)
        ]

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(vectorize_texts)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return the top_k most relevant chunks for the query, formatted as text."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue  # not actually relevant, skip it
            results.append(f"[Source: {self.sources[i]}]\n{self.chunks[i]}")

        if not results:
            return "No relevant information found in the knowledge base."

        return "\n\n".join(results)


# built once at import time and shared across every agent node
retriever = KnowledgeBaseRetriever()


def rag_search(query: str, top_k: int = 3) -> str:
    """Convenience wrapper the agent nodes call directly."""
    return retriever.retrieve(query, top_k=top_k)
