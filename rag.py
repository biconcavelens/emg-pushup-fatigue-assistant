"""Tiny dependency-free RAG retriever (BM25) over knowledge/*.md paragraphs."""
import math
import re
from collections import Counter
from pathlib import Path

KNOWLEDGE = Path(__file__).parent / "knowledge"
STOP = set("a an the of to and or in on for with is are be as at by it this that from "
           "can may not no if than then so".split())


def _tok(s):
    return [w for w in re.findall(r"[a-z0-9%]+", s.lower()) if w not in STOP]


class Retriever:
    def __init__(self, folder=KNOWLEDGE):
        self.chunks = []   # (source title, text)
        for p in sorted(Path(folder).glob("*.md")):
            title, body = None, []
            for line in p.read_text().splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.strip():
                    body.append(line.strip())
            for para in body:
                self.chunks.append((title or p.stem, para))
        self.docs = [_tok(t + " " + c) for t, c in self.chunks]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.avg = sum(len(d) for d in self.docs) / max(len(self.docs), 1)

    def search(self, query, k=3, k1=1.5, b=0.75):
        q = _tok(query)
        n = len(self.docs)
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / self.avg))
            scores.append(s)
        order = sorted(range(n), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i][0], self.chunks[i][1], scores[i]) for i in order if scores[i] > 0]


if __name__ == "__main__":
    import sys
    r = Retriever()
    for title, text, s in r.search(" ".join(sys.argv[1:]) or "stop set tempo loss", 3):
        print(f"[{s:.2f}] {title}: {text[:110]}...")
