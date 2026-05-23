"""技能库 — 自进化行程缓存（TF-IDF 检索 + JSON 存储）"""
import os
import json
import jieba
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_DIR = "skill_data"
SKILL_FILE = os.path.join(SKILL_DIR, "skills.json")


def _ensure_dir():
    os.makedirs(SKILL_DIR, exist_ok=True)


def _load_skills() -> list[dict]:
    _ensure_dir()
    if os.path.exists(SKILL_FILE):
        with open(SKILL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_skills(skills: list[dict]):
    _ensure_dir()
    with open(SKILL_FILE, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)


def _tokenize(text: str) -> str:
    return " ".join(jieba.cut(text))


def add_skill(query: str, itinerary: str) -> int:
    """添加技能，返回技能总数"""
    skills = _load_skills()

    # 去重——相同 query 覆盖旧记录
    skills = [s for s in skills if s["query"] != query]

    skills.append({
        "query": query,
        "itinerary": itinerary[:10000],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "visit_count": 1,
    })

    _save_skills(skills)
    return len(skills)


def search_skills(query: str, top_k: int = 3) -> list[dict]:
    """检索匹配的技能"""
    skills = _load_skills()
    if not skills:
        return []

    corpus = [_tokenize(s["query"]) for s in skills]
    vectorizer = TfidfVectorizer()
    try:
        embeddings = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([_tokenize(query)])
        scores = cosine_similarity(query_vec, embeddings)[0]
    except ValueError:
        return []

    # Top-K
    top_indices = scores.argsort()[::-1][:min(top_k, len(scores))]
    results = []
    for idx in top_indices:
        if scores[idx] < 0.1:
            continue
        skill = skills[idx]
        # 增加访问计数
        skill["visit_count"] = skill.get("visit_count", 1) + 1
        results.append({
            "query": skill["query"],
            "itinerary": skill["itinerary"],
            "score": round(float(scores[idx]), 3),
            "created_at": skill.get("created_at", ""),
        })

    _save_skills(skills)
    return results


def get_stats() -> dict:
    skills = _load_skills()
    return {"total_skills": len(skills)}
