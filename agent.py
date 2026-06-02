import os
import re
import json
from datetime import datetime
from typing import Any, List, Optional
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")
P = "%s" if DATABASE_URL else "?"

# -----------------------------
# DB connection
# -----------------------------
def _conn():
    if DATABASE_URL:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        return conn
    else:
        import sqlite3
        conn = sqlite3.connect("events.db")
        conn.row_factory = sqlite3.Row
        return conn

def _fetchall(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    if DATABASE_URL:
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    return cur.fetchall()

def _execute(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur

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
    if _db_locations_cache is not None and not DATABASE_URL:
        return _db_locations_cache
    conn = _conn()
    rows = _fetchall(conn, "SELECT DISTINCT location FROM events WHERE location IS NOT NULL")
    conn.close()
    _db_locations_cache = [r["location"] if isinstance(r, dict) else r[0] for r in rows if r]
    return _db_locations_cache


def _city_patterns_from_locations(resolved_locs: List[str]) -> List[str]:
    skip_words = {
        "germany", "france", "spain", "netherlands", "switzerland",
        "belgium", "italy", "japan", "india", "canada", "singapore",
        "united kingdom", "uk", "usa", "united states"
    }
    patterns = set()
    for loc in resolved_locs:
        parts = [p.strip() for p in loc.split(',')]
        for part in parts:
            part_lower = part.lower()
            if re.match(r'^[A-Z]{2}$', part):
                continue
            if any(c.isdigit() for c in part):
                continue
            if part_lower in skip_words:
                continue
            if len(part) <= 3:
                continue
            if len(part) > 35:
                continue
            if '(' in loc and loc not in patterns:
                patterns.add(loc)
                break
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
        "- start_date: YYYY-MM-DD or null. Only set if user explicitly says 'upcoming', 'next week', 'this month', 'in June' etc. Otherwise leave null.\n"
        "- end_date: YYYY-MM-DD or null. Set only if user specifies a time window like next week or next month.\n"
        "- intent_type: one of [specific, explore, vague]\n\n"
        "Location matching rules for resolved_locations:\n"
        "- Use your geographic knowledge to match any region, city, country or continent\n"
        "- Be comprehensive and inclusive — if user says europe, return ALL European cities in the list\n"
        "- Include venue-prefixed entries if the city matches e.g. 'IBM, 425 Market, San Francisco, CA'\n"
        "- Match both formats: 'Paris, France' AND 'Paris (France)' for the same city\n"
        "- When in doubt INCLUDE rather than exclude\n\n"
        "Database locations:\n" + locations_list
    )

    try:
        resp = client.responses.create(
            model="gpt-4.1",
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
            loc_clauses = " OR ".join([f"location LIKE {P}" for _ in city_patterns])
            where.append("(" + loc_clauses + ")")
            params.extend(["%" + p + "%" for p in city_patterns])
    elif filters.location:
        where.append(f"location LIKE {P}")
        params.append("%" + filters.location + "%")

    if filters.keywords:
        kw_clauses = " OR ".join([f"title LIKE {P}" for _ in filters.keywords])
        where.append("(" + kw_clauses + ")")
        params.extend(["%" + kw + "%" for kw in filters.keywords])

    if filters.start_date:
        where.append(f"date_iso >= {P}")
        params.append(filters.start_date)

    if filters.end_date:
        where.append(f"date_iso <= {P}")
        params.append(filters.end_date)

    order_sql = "ASC" if filters.sort == "date_asc" else "DESC"
    where_sql = " AND ".join(where)

    sql = (
        "SELECT id, title, date_iso, location, link, website_source, cancelled, verified, organizer "
        "FROM events WHERE " + where_sql + " "
        "ORDER BY CASE WHEN date_iso IS NULL THEN 1 ELSE 0 END, "
        "date_iso " + order_sql + ", title ASC "
        "LIMIT " + str(filters.limit)
    )

    conn = _conn()
    try:
        rows = _fetchall(conn, sql, params)
        for r in rows:
            event_id = r["id"] if isinstance(r, dict) else r[0]
            _execute(conn, f"UPDATE events SET search_count = search_count + 1 WHERE id = {P}", (event_id,))
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
            loc_clauses = " OR ".join([f"location LIKE {P}" for _ in city_patterns])
            params.extend(["%" + p + "%" for p in city_patterns])
            query += " AND (" + loc_clauses + ")"

    rows = _fetchall(conn, query, params)
    conn.close()

    scored = []
    for r in rows:
        emb = r["embedding"] if isinstance(r, dict) else r[list(r.keys()).index("embedding")]
        if not emb:
            continue
        event_vec = json.loads(emb)
        semantic_score = cosine_similarity(query_vec, event_vec)
        popularity = (r["search_count"] if isinstance(r, dict) else r["search_count"]) or 0
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
    _execute(conn, f"INSERT INTO searches (query, timestamp) VALUES ({P}, {P})",
             (user_text, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# -----------------------------
# Personalization
# -----------------------------
def get_recent_queries(limit: int = 20) -> List[str]:
    conn = _conn()
    rows = _fetchall(conn, f"SELECT query FROM searches ORDER BY timestamp DESC LIMIT {P}", (limit,))
    conn.close()
    return [r["query"] if isinstance(r, dict) else r[0] for r in rows]


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


def get_personalized_recommendations(limit: int = 10, user_location: str = "New York"):
    queries = get_recent_queries(20)

    time_words = {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "today", "tomorrow", "week", "month", "next", "this"
    }
    action_words = {
        "events", "event", "find", "show", "get", "list", "search"
    }
    location_preps = {"in", "at", "near", "around"}

    locations = set()
    topics = set()

    for q in queries:
        if not any(c.isalpha() for c in q):
            continue
        words = q.strip().lower().split()
        while words and words[-1] in time_words:
            words.pop()
        if words and words[-1] in ("in", "at", "near"):
            words.pop()

        loc_phrase = []
        topic_words_found = []
        i = 0
        while i < len(words):
            w = words[i]
            if w in location_preps and i + 1 < len(words):
                loc_phrase = []
                i += 1
                while i < len(words) and words[i] not in location_preps and words[i] not in action_words:
                    if words[i] not in time_words:
                        loc_phrase.append(words[i])
                    i += 1
                if loc_phrase:
                    locations.add(" ".join(loc_phrase).title())
            elif w not in action_words and w not in location_preps and w not in time_words and len(w) > 2:
                topic_words_found.append(w)
                i += 1
            else:
                i += 1

        if topic_words_found:
            topics.add(" ".join(topic_words_found).title())

    topic_keywords = locations | topics

    conn = _conn()
    kw_list = list(topic_keywords)

    rows = []
    seen_titles = set()
    per_topic = max(1, limit // len(kw_list)) if kw_list else limit

    for kw in kw_list:
        topic_rows = _fetchall(conn,
            f"SELECT title, date_iso, location, link, search_count, organizer FROM events "
            f"WHERE verified = 1 AND cancelled = 0 "
            f"AND (title LIKE {P} OR location LIKE {P}) "
            f"ORDER BY search_count DESC, date_iso ASC LIMIT {P}",
            ("%" + kw + "%", "%" + kw + "%", per_topic)
        )
        for r in topic_rows:
            title = r["title"] if isinstance(r, dict) else r[0]
            if title not in seen_titles:
                seen_titles.add(title)
                rows.append(r)

    if len(rows) < limit:
        extra = _fetchall(conn,
            f"SELECT title, date_iso, location, link, search_count, organizer FROM events "
            f"WHERE verified = 1 AND cancelled = 0 "
            f"ORDER BY search_count DESC, date_iso ASC LIMIT {P}", (limit,)
        )
        for r in extra:
            title = r["title"] if isinstance(r, dict) else r[0]
            if title not in seen_titles and len(rows) < limit:
                seen_titles.add(title)
                rows.append(r)

    popular = _fetchall(conn,
        f"SELECT id, title, date_iso, location, link, search_count, organizer, COALESCE(click_count, 0) as click_count FROM events "
        f"WHERE verified = 1 AND cancelled = 0 AND location LIKE {P} AND COALESCE(click_count, 0) > 0 "
        f"ORDER BY click_count DESC, date_iso ASC LIMIT 10",
        ("%" + user_location + "%",)
    )

    conn.close()
    return rows, user_location, list(topic_keywords), popular


# -----------------------------
# CLI Agent
# -----------------------------
def ask_agent(user_text: str) -> str:
    log_search(user_text)
    filters, resolved_locs = parse_user_to_filters(user_text)

    if resolved_locs or filters.location or filters.start_date or filters.end_date:
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

    past_events = []

    if resolved_locs or filters.location or filters.start_date or filters.end_date:
        rows = query_events(filters, resolved_locs)
        if not rows and filters.start_date and resolved_locs:
            past_filters = filters.copy()
            past_filters.start_date = None
            past_filters.end_date = filters.start_date
            past_filters.sort = "date_desc"
            past_filters.limit = 3
            past_rows = query_events(past_filters, resolved_locs)
            past_events = past_rows
        elif not rows and resolved_locs:
            rows = semantic_search(user_text, filters.limit, resolved_locs)
    else:
        rows = semantic_search(user_text, filters.limit, resolved_locs)

    if not rows and not filters.location and not past_events:
        conn = _conn()
        rows = _fetchall(conn,
            f"SELECT * FROM events WHERE verified = 1 AND cancelled = 0 "
            f"ORDER BY search_count DESC, date_iso ASC LIMIT {P}",
            (filters.limit,)
        )
        conn.close()

    def _get(r, key):
        if isinstance(r, dict):
            return r.get(key)
        try:
            return r[key]
        except Exception:
            return None

    events = []
    for r in rows:
        events.append({
            "id": _get(r, "id"),
            "title": _get(r, "title"),
            "date": _get(r, "date_iso"),
            "location": _get(r, "location"),
            "organizer": _get(r, "organizer"),
            "link": _get(r, "link"),
            "verified": bool(_get(r, "verified")),
            "cancelled": bool(_get(r, "cancelled")),
        })

    past = []
    for r in past_events:
        past.append({
            "id": _get(r, "id"),
            "title": _get(r, "title"),
            "date": _get(r, "date_iso"),
            "location": _get(r, "location"),
            "organizer": _get(r, "organizer"),
            "link": _get(r, "link"),
        })

    return {
        "query": user_text,
        "count": len(events),
        "events": events,
        "past_events": past
    }