"""RAG 知识库 — jieba 分词 + sklearn TF-IDF + 余弦相似度"""
import os
import hashlib
import json
import pickle
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = "kb_data"
DOCS_FILE = os.path.join(KB_DIR, "docs.json")
VEC_FILE = os.path.join(KB_DIR, "vectorizer.pkl")
EMB_FILE = os.path.join(KB_DIR, "embeddings.npy")

_vectorizer: TfidfVectorizer | None = None
_embeddings: np.ndarray | None = None  # 每个块的向量
_documents: list[dict] = []  # [{id, title, chunk_idx, content}]


def _tokenize(text: str) -> str:
    """jieba 分词后用空格连接（TF-IDF 需要空格分隔的 token）"""
    return " ".join(jieba.cut(text))


def _ensure_dir():
    os.makedirs(KB_DIR, exist_ok=True)


def _load():
    global _vectorizer, _embeddings, _documents
    _ensure_dir()
    if os.path.exists(VEC_FILE):
        with open(VEC_FILE, "rb") as f:
            _vectorizer = pickle.load(f)
    if os.path.exists(EMB_FILE):
        _embeddings = np.load(EMB_FILE)
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            _documents = json.load(f)


def _save():
    _ensure_dir()
    if _vectorizer:
        with open(VEC_FILE, "wb") as f:
            pickle.dump(_vectorizer, f)
    if _embeddings is not None:
        np.save(EMB_FILE, _embeddings)
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(_documents, f, ensure_ascii=False)


def _doc_id(title: str, chunk_idx: int) -> str:
    h = hashlib.md5((title + str(chunk_idx)).encode()).hexdigest()[:12]
    return f"doc_{h}"


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def add_document(title: str, content: str) -> int:
    """添加文档到知识库，返回入库块数"""
    global _vectorizer, _embeddings

    if not content or len(content) < 50:
        return 0

    _load()
    chunks = split_text(content)
    new_count = 0

    for i, chunk in enumerate(chunks):
        doc_id = _doc_id(title, i)
        if any(d["id"] == doc_id for d in _documents):
            continue
        _documents.append({"id": doc_id, "title": title, "chunk_idx": i, "content": chunk})
        new_count += 1

    if not new_count:
        return 0

    # 重建 TF-IDF 向量（全量重算，简单可靠）
    all_texts = [_tokenize(d["content"]) for d in _documents]
    _vectorizer = TfidfVectorizer(max_features=512)
    _embeddings = _vectorizer.fit_transform(all_texts).toarray()

    _save()
    return new_count


def search(query: str, top_k: int = 5) -> list[dict]:
    """语义搜索知识库"""
    _load()

    if _vectorizer is None or _embeddings is None or len(_documents) == 0:
        return []

    query_vec = _vectorizer.transform([_tokenize(query)])
    scores = cosine_similarity(query_vec, _embeddings)[0]

    # Top-K indices
    top_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]

    results = []
    for idx in top_indices:
        if scores[idx] < 0.05:  # 阈值过滤
            continue
        doc = _documents[idx]
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": round(float(scores[idx]), 3),
        })

    return results


def get_stats() -> dict:
    _load()
    titles = set(d["title"] for d in _documents)
    return {"total_chunks": len(_documents), "total_documents": len(titles)}


def delete_document(title: str) -> bool:
    global _vectorizer, _embeddings
    _load()
    old_len = len(_documents)
    _documents[:] = [d for d in _documents if d["title"] != title]
    if len(_documents) == old_len:
        return False

    if _documents:
        all_texts = [_tokenize(d["content"]) for d in _documents]
        _vectorizer = TfidfVectorizer(max_features=512)
        _embeddings = _vectorizer.fit_transform(all_texts).toarray()
    else:
        _vectorizer = None
        _embeddings = None

    _save()
    return True
