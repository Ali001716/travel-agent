"""技能库 — 结构化 SOP 存储（策略 + 已验证数据，非纯文本行程）"""
import os
import json
import jieba
import re
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


def _extract_strategy(itinerary: str, query: str) -> dict:
    """从行程文本中提取结构化策略"""
    city_match = re.search(r"(广州|深圳|北京|上海|成都|重庆|杭州|南京|武汉|西安|长沙|厦门|青岛|大连|三亚|[^\s]{2}市)", itinerary[:200])
    city = city_match.group(0) if city_match else ""

    # 提取景点列表
    places = set()
    for m in re.finditer(r"\*\*(.+?)\*\*", itinerary):
        name = m.group(1).strip()
        if len(name) >= 3 and not re.match(r"^\d|分钟|小时|站$|号线$", name):
            places.add(name)

    # 提取路线经验
    routes = []
    for m in re.finditer(r"(\S{2,8})\s*→\s*(\S{2,8})", itinerary):
        routes.append({"from": m.group(1), "to": m.group(2)})

    # 提取天数
    days = 1
    day_match = re.search(r"(\d+)\s*日|Day\s*(\d+)", itinerary)
    if day_match:
        days = int(day_match.group(1) or day_match.group(2) or 1)

    return {
        "city": city,
        "days": days,
        "places": list(places)[:15],
        "proven_routes": routes[:10],
        "day_count": len([p for p in places if "Day" in p or "日" in p]) or days,
    }


def add_skill(query: str, itinerary: str) -> int:
    """添加结构化的技能"""
    skills = _load_skills()
    skills = [s for s in skills if s["query"] != query]

    strategy = _extract_strategy(itinerary, query)

    skills.append({
        "query": query,
        "query_pattern": re.sub(r"\d+", "N", query),  # "成都3日游" → "成都N日游"
        "city": strategy["city"],
        "days": strategy["days"],
        "strategy": strategy,
        "itinerary": itinerary[:10000],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "visit_count": 1,
    })

    _save_skills(skills)
    return len(skills)


def search_skills(query: str, top_k: int = 3) -> list[dict]:
    """检索匹配的技能，返回结构化策略指导"""
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

    top_indices = scores.argsort()[::-1][:min(top_k, len(scores))]
    results = []
    for idx in top_indices:
        if scores[idx] < 0.1:
            continue
        skill = skills[idx]
        skill["visit_count"] = skill.get("visit_count", 1) + 1
        strategy = skill.get("strategy", {})
        results.append({
            "query": skill["query"],
            "city": skill.get("city", ""),
            "days": skill.get("days", 1),
            "strategy": strategy,
            "itinerary": skill["itinerary"],
            "score": round(float(scores[idx]), 3),
            "created_at": skill.get("created_at", ""),
            "visit_count": skill.get("visit_count", 1),
        })

    _save_skills(skills)
    return results


def get_stats() -> dict:
    skills = _load_skills()
    cities = set(s.get("city", "") for s in skills if s.get("city"))
    return {"total_skills": len(skills), "cities_covered": len(cities)}
