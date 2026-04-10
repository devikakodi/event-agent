import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Any, List, Optional
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

DB_PATH = "events.db"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Schema
# -----------------------------
class SearchFilters(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    verified_only: bool = True
    exclude_cancelled: bool = True
    limit: int = 25
    sort: str = "date_asc"
    intent_type: Optional[str] = "explore"


# -----------------------------
# DB connection
# -----------------------------
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_int(n: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(n)
        return max(lo, min(hi, v))
    except Exception:
        return default


# -----------------------------
# Location resolution
# -----------------------------
_db_locations_cache: Optional[List[str]] = None

def _get_db_locations() -> List[str]:
    global _db_locations_cache
    if _db_locations_cache is not None:
        return _db_locations_cache
    conn = _conn()
    rows = conn.execute("SELECT DISTINCT location FROM events WHERE location IS NOT NULL").fetchall()
    conn.close()
    _db_locations_cache = [r["location"] for r in rows if r["location"]]
    return _db_locations_cache


def _city_patterns_from_locations(resolved_locs: List[str]) -> List[str]:
    patterns = set()
    for loc in resolved_locs:
        parts = [p.strip() for p in loc.split(',')]
        for part in parts:
            if (len(part) > 3
                    and not re.match(r'^[A-Z]{2}$', part)
                    and not any(c.isdigit() for c in part)
                    and len(part) < 35):
                patterns.add(part)
    return list(patterns)


# -----------------------------
# LLM: parse query + resolve locations in ONE call
# -----------------------------
def parse_user_to_filters(user_text: str):
    today = datetime.now().date().isoformat()
    db_locations = _get_db_locations()
    locations_list = "\n".join("- " + l for l in db_locations)

    stopwords = {
        "events", "event", "show", "shows", "find", "get", "list",
        "search", "near", "any", "all", "some", "good", "best",
        "the", "and", "for", "with", "about", "upcoming", "latest"
    }

    system_prompt = (
        "Extract structured search intent from the user's query.\n"
        "Return JSON with these fields:\n"
        "- keywords: topic/subject keywords only, never location words\n"
        "- location: location as user said it or null\n"
        "- resolved_locations: array of EXACT strings from the DB list below matching the user's location. [] if none.\n"
        "- start_date: YYYY-MM-DD or null\n"
        "- end_date: YYYY-MM-DD or null\n"
        "- intent_type: one of [specific, explore, vague]\n\n"
        "Location matching rules for resolved_locations:\n"
        "- Include venue-prefixed variants e.g. 'IBM, 425 Market, San Francisco, CA'\n"
        "- east coast = New York, Boston, Chicago, Orlando + venue variants\n"
        "- west coast = San Francisco, San Jose, Mountain View, Oakland, Stanford, San Diego, Tacoma + variants\n"
        "- bay area = San Francisco, San Jose, Mountain View, Oakland, Stanford + variants\n"
        "- europe = London, Berlin, Paris, Amsterdam, Madrid, Brussels, Gent, Lausanne + variants\n"
        "- asia = Tokyo, Bangalore, Singapore\n"
        "- online/virtual = Online\n"
        "- When in doubt INCLUDE rather than exclude\n\n"
        "Database locations:\n" + locations_list
    )

    try:
        resp = client.responses.create(
            model="gpt-4.1-nano",
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Today: " + today + "\nQuery: " + user_text}
            ],
        )
        data = json.loads(resp.output_text.strip())
    except Exception:
        return SearchFilters(keywords=[user_text]), []

    data["limit"] = _safe_int(data.get("limit"), 25, 1, 50)
    if "keywords" in data and isinstance(data["keywords"], list):
        data["keywords"] = [k for k in data["keywords"]
                            if k.lower() not in stopwords and len(k) > 2]

    resolved = data.pop("resolved_locations", [])
    db_set = set(db_locations)
    resolved = [l for l in resolved if isinstance(l, str) and l in db_set]

    try:
        filters = SearchFilters(**data)
    except ValidationError:
        filters = SearchFilters(keywords=[user_text])

    return filters, resolved


# -----------------------------
# Query events — direct SQL
# -----------------------------
def query_events(filters: SearchFilters, resolved_locs: List[str]) -> list:
    where = ["verified = 1", "cancelled = 0"]
    params: List[Any] = []

    if resolved_locs:
        city_patterns = _city_patterns_from_locations(resolved_locs)
        if city_patterns:
            loc_clauses = " OR ".join(["location LIKE ?" for _ in city_patterns])
            where.append("(" + loc_clauses + ")")
            params.extend(["%" + p + "%" for p in city_patterns])
    elif filters.location:
        # Direct substring fallback
        where.append("location LIKE ?")
        params.append("%" + filters.location + "%")

    if filters.keywords:
        kw_clauses = " OR ".join(["title LIKE ?" for _ in filters.keywords])
        where.append("(" + kw_clauses + ")")
        params.extend(["%" + kw + "%" for kw in filters.keywords])

    if filters.start_date:
        where.append("date_iso >= ?")
        params.append(filters.start_date)

    if filters.end_date:
        where.append("date_iso <= ?")
        params.append(filters.end_date)

    order_sql = "ASC" if filters.sort == "date_asc" else "DESC"
    where_sql = " AND ".join(where)

    sql = (
        "SELECT id, title, date_iso, location, link, website_source, cancelled, verified "
        "FROM events WHERE " + where_sql + " "
        "ORDER BY CASE WHEN date_iso IS NULL THEN 1 ELSE 0 END, "
        "date_iso " + order_sql + ", title ASC "
        "LIMIT " + str(filters.limit)
    )

    conn = _conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            conn.execute("UPDATE events SET search_count = search_count + 1 WHERE id = ?", (r["id"],))
        conn.commit()
    except Exception as e:
        print("[DEBUG] SQL error:", e)
        rows = []
    conn.close()
    return rows


# -----------------------------
# Semantic search
# -----------------------------
def embed_query(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(user_text, limit=25, resolved_locs=None):
    query_vec = embed_query(user_text)

    conn = _conn()
    query = "SELECT * FROM events WHERE verified = 1 AND cancelled = 0"
    params = []

    if resolved_locs:
        city_patterns = _city_patterns_from_locations(resolved_locs)
        if city_patterns:
            loc_clauses = " OR ".join(["location LIKE ?" for _ in city_patterns])
            params.extend(["%" + p + "%" for p in city_patterns])
            query += " AND (" + loc_clauses + ")"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    scored = []
    for r in rows:
        if not r["embedding"]:
            continue
        event_vec = json.loads(r["embedding"])
        semantic_score = cosine_similarity(query_vec, event_vec)
        popularity = r["search_count"] or 0
        popularity_norm = min(popularity / 10, 1)
        score = 0.7 * semantic_score + 0.3 * popularity_norm
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


# -----------------------------
# Logging
# -----------------------------
def log_search(user_text: str):
    conn = _conn()
    conn.execute("INSERT INTO searches (query, timestamp) VALUES (?, ?)",
                 (user_text, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# -----------------------------
# Personalization
# -----------------------------
def get_recent_queries(limit: int = 20) -> List[str]:
    conn = _conn()
    rows = conn.execute("SELECT query FROM searches ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [r["query"] for r in rows]


def extract_keywords_from_queries(queries: List[str]) -> List[str]:
    keywords = []
    for q in queries:
        for w in q.lower().split():
            if len(w) > 3:
                keywords.append(w)
    return list(set(keywords))


def extract_location_from_queries(queries: List[str]) -> Optional[str]:
    location_words = {"in", "at", "near", "around"}
    for q in queries:
        words = q.lower().split()
        for i, w in enumerate(words):
            if w in location_words and i + 1 < len(words):
                return " ".join(words[i+1:])
    return None


def get_personalized_recommendations(limit: int = 10):
    queries = get_recent_queries()

    stopwords = {
        "events", "event", "in", "at", "near", "around", "from", "show",
        "find", "get", "the", "and", "for", "with", "about", "some", "any"
    }
    topic_keywords = set()
    for q in queries:
        for w in q.lower().split():
            if len(w) > 3 and w not in stopwords:
                topic_keywords.add(w)

    recent_location = extract_location_from_queries(queries)

    conn = _conn()
    clauses = []
    params = []

    kw_list = list(topic_keywords)[:15]
    if kw_list:
        kw_parts = " OR ".join(["title LIKE ?" for _ in kw_list])
        clauses.append("(" + kw_parts + ")")
        params.extend(["%" + kw + "%" for kw in kw_list])

    where_sql = ("verified = 1 AND cancelled = 0 AND (" + " OR ".join(clauses) + ")"
                 if clauses else "verified = 1 AND cancelled = 0")

    rows = list(conn.execute(
        "SELECT title, date_iso, location, link, search_count FROM events "
        "WHERE " + where_sql + " ORDER BY search_count DESC, date_iso ASC LIMIT ?",
        params + [limit]
    ).fetchall())

    if len(rows) < limit:
        existing = {r["title"] for r in rows}
        extra = conn.execute(
            "SELECT title, date_iso, location, link, search_count FROM events "
            "WHERE verified = 1 AND cancelled = 0 "
            "ORDER BY search_count DESC, date_iso ASC LIMIT ?", (limit,)
        ).fetchall()
        for r in extra:
            if r["title"] not in existing and len(rows) < limit:
                rows.append(r)

    conn.close()
    return rows, recent_location, list(topic_keywords)


# -----------------------------
# CLI Agent
# -----------------------------
def ask_agent(user_text: str) -> str:
    log_search(user_text)
    filters, resolved_locs = parse_user_to_filters(user_text)
    has_filters = bool(filters.location or filters.start_date or filters.end_date)

    if has_filters or filters.intent_type == "specific":
        rows = query_events(filters, resolved_locs)
        if not rows and resolved_locs:
            rows = semantic_search(user_text, filters.limit, resolved_locs)
    else:
        rows = semantic_search(user_text, filters.limit, resolved_locs)

    return "Found " + str(len(rows)) + " events."


# -----------------------------
# API Agent
# -----------------------------
def ask_agent_structured(user_text: str):
    log_search(user_text)
    filters, resolved_locs = parse_user_to_filters(user_text)
    has_filters = bool(filters.location or filters.start_date or filters.end_date)

    if has_filters or filters.intent_type == "specific":
        rows = query_events(filters, resolved_locs)
        if not rows and resolved_locs:
            rows = semantic_search(user_text, filters.limit, resolved_locs)
    else:
        rows = semantic_search(user_text, filters.limit, resolved_locs)

    if not rows and not filters.location:
        conn = _conn()
        rows = conn.execute(
            "SELECT * FROM events WHERE verified = 1 AND cancelled = 0 "
            "ORDER BY search_count DESC, date_iso ASC LIMIT ?",
            (filters.limit,)
        ).fetchall()
        conn.close()

    events = []
    for r in rows:
        events.append({
            "title": r["title"],
            "date": r["date_iso"],
            "location": r["location"],
            "link": r["link"],
            "verified": bool(r["verified"]),
            "cancelled": bool(r["cancelled"]),
        })

    return {
        "query": user_text,
        "count": len(events),
        "events": events
    }