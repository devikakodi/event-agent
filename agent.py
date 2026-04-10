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
def _get_db_locations() -> List[str]:
    conn = _conn()
    rows = conn.execute("SELECT DISTINCT location FROM events WHERE location IS NOT NULL").fetchall()
    conn.close()
    return [r["location"] for r in rows if r["location"]]


def _resolve_locations_via_llm(user_location_text: str, db_locations: List[str]) -> List[str]:
    if not db_locations:
        return []

    locations_list = "\n".join(f"- {l}" for l in db_locations)

    prompt = f"""You are a location matcher. The user is looking for events in: "{user_location_text}"

Here are ALL the locations in the database (exact strings, some have venue prefixes):
{locations_list}

Return a JSON array of ALL locations from the list that geographically match the user's request.
- Match exact cities: "new york" matches "New York, NY" and "LightningAI, New York, NY"
- Match regions: "east coast" = New York, Boston, Chicago, Orlando + venue variants
- Match regions: "west coast" = San Francisco, San Jose, Mountain View, Oakland, Stanford, San Diego, Tacoma + venue variants
- Match regions: "bay area" = San Francisco, San Jose, Mountain View, Oakland, Stanford + venue variants
- Match regions: "europe" = London, Berlin, Paris, Amsterdam, Madrid, Brussels, Gent, Lausanne + venue variants
- Match regions: "asia" = Tokyo, Bangalore, Singapore
- "online" or "virtual" = Online
- Always include venue-prefixed variants if the city matches e.g. "IBM, 425 Market, San Francisco, CA"
- When in doubt, INCLUDE rather than exclude
- Return ONLY a JSON array of exact strings from the list above, no explanation, no markdown."""

    try:
        resp = client.responses.create(model="gpt-4.1-nano", temperature=0, input=prompt)
        raw = resp.output_text.strip().replace("```json", "").replace("```", "").strip()
        matched = json.loads(raw)
        db_set = set(db_locations)
        valid = [l for l in matched if l in db_set]

        # Safety net: also pull in any DB location containing the same city names
        city_hints = set()
        for l in valid:
            for part in l.split(','):
                p = part.strip()
                if len(p) > 3 and not re.match(r'^[A-Z]{2}$', p) and not any(c.isdigit() for c in p):
                    city_hints.add(p.lower())

        for db_loc in db_locations:
            if db_loc not in valid:
                db_low = db_loc.lower()
                for city in city_hints:
                    if re.search(r'\b' + re.escape(city) + r'\b', db_low):
                        valid.append(db_loc)
                        break

        return valid

    except Exception as ex:
        print(f"[DEBUG] LLM location resolve failed: {ex}")
        low = user_location_text.lower()
        return [l for l in db_locations if low in l.lower() or l.lower() in low]


def _location_variants(loc: Optional[str], db_locations: Optional[List[str]] = None) -> List[str]:
    if not loc or not loc.strip():
        return []
    locations = db_locations or _get_db_locations()
    return _resolve_locations_via_llm(loc, locations)


def _city_patterns_from_locations(resolved_locs: List[str]) -> List[str]:
    """
    Extract city name patterns from resolved location strings for SQL LIKE matching.
    Handles both plain "San Francisco, CA" and venue-prefixed "IBM, 425 Market, San Francisco, CA".
    Also handles short inputs like "New York" directly.
    """
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
# Query events
# -----------------------------
def query_events(filters: SearchFilters):
    db_locations = _get_db_locations()
    resolved_locs = _location_variants(filters.location, db_locations) if filters.location else []

    where = ["verified = 1", "cancelled = 0"]
    params: List[Any] = []

    if resolved_locs:
        city_patterns = _city_patterns_from_locations(resolved_locs)
        if city_patterns:
            loc_clauses = " OR ".join(["location LIKE ?" for _ in city_patterns])
            where.append(f"({loc_clauses})")
            params.extend([f"%{p}%" for p in city_patterns])
        else:
            # Fallback: search raw location string directly
            where.append("location LIKE ?")
            params.append(f"%{filters.location}%")
    elif filters.location:
        # LLM returned nothing — do a direct substring search as fallback
        where.append("location LIKE ?")
        params.append(f"%{filters.location}%")

    if filters.keywords:
        kw_clauses = " OR ".join(["title LIKE ?" for _ in filters.keywords])
        where.append(f"({kw_clauses})")
        params.extend([f"%{kw}%" for kw in filters.keywords])

    if filters.start_date:
        where.append("date_iso >= ?")
        params.append(filters.start_date)

    if filters.end_date:
        where.append("date_iso <= ?")
        params.append(filters.end_date)

    order_sql = "ASC" if filters.sort == "date_asc" else "DESC"
    where_sql = " AND ".join(where)

    sql = f"""
    SELECT id, title, date_iso, location, link, website_source, cancelled, verified
    FROM events
    WHERE {where_sql}
    ORDER BY
      CASE WHEN date_iso IS NULL THEN 1 ELSE 0 END,
      date_iso {order_sql},
      title ASC
    LIMIT {filters.limit}
    """

    conn = _conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            conn.execute("UPDATE events SET search_count = search_count + 1 WHERE id = ?", (r["id"],))
        conn.commit()
    except Exception as e:
        print(f"[DEBUG] SQL error: {e}")
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


def semantic_search(user_text, limit=25, location=None):
    query_vec = embed_query(user_text)

    conn = _conn()
    query = "SELECT * FROM events WHERE verified = 1 AND cancelled = 0"
    params = []

    if location:
        db_locations = _get_db_locations()
        resolved_locs = _location_variants(location, db_locations)
        if resolved_locs:
            city_patterns = _city_patterns_from_locations(resolved_locs)
            if city_patterns:
                loc_clauses = " OR ".join(["location LIKE ?" for _ in city_patterns])
                params.extend([f"%{p}%" for p in city_patterns])
                query += f" AND ({loc_clauses})"
        else:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")

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
# LLM → Filters
# -----------------------------
def parse_user_to_filters(user_text: str) -> SearchFilters:
    today = datetime.now().date().isoformat()

    resp = client.responses.create(
        model="gpt-4.1-nano",
        temperature=0,
        input=[
            {"role": "system",
             "content": """Extract structured search intent from the user's query.
Return JSON with these fields:
- keywords: list of topic/subject keywords (e.g. ["AI", "machine learning"]) — never include location words
- location: location exactly as user said it (e.g. "east coast", "bay area", "New York", "NYC", "Europe") or null
- start_date: YYYY-MM-DD or null
- end_date: YYYY-MM-DD or null
- intent_type: one of ["specific", "explore", "vague"]

Rules:
- "specific" = has a clear location or date
- "explore" = broad topic interest with no specific location/date
- "vague" = totally unclear
- ALWAYS extract location if the user mentioned one. Never put location words in keywords.
- keywords = only topic words like "AI", "machine learning", "security", "open source", "agents"
"""},
            {"role": "user", "content": f"Today: {today}\nQuery: {user_text}"}
        ],
    )

    data = json.loads(resp.output_text.strip())
    data["limit"] = _safe_int(data.get("limit"), 25, 1, 50)

    stopwords = {
        "events", "event", "show", "shows", "find", "get", "list",
        "search", "near", "any", "all", "some", "good", "best",
        "the", "and", "for", "with", "about", "upcoming", "latest"
    }
    if "keywords" in data and isinstance(data["keywords"], list):
        data["keywords"] = [k for k in data["keywords"]
                            if k.lower() not in stopwords and len(k) > 2]

    try:
        return SearchFilters(**data)
    except ValidationError:
        return SearchFilters(keywords=[user_text])


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


def get_personalized_recommendations(limit: int = 10):
    """
    Returns personalized recommendations based on:
    1. Topics from recent searches (keyword matching)
    2. Locations from recent searches
    3. Overall popularity (search_count)
    Blends all three into a ranked list.
    """
    queries = get_recent_queries()

    # Extract topic keywords from past searches
    stopwords = {
        "events", "event", "in", "at", "near", "around", "from", "show",
        "find", "get", "the", "and", "for", "with", "about", "some", "any"
    }
    topic_keywords = set()
    for q in queries:
        for w in q.lower().split():
            if len(w) > 3 and w not in stopwords:
                topic_keywords.add(w)

    # Extract location from most recent location-bearing query
    recent_location = None
    location_prepositions = {"in", "at", "near", "around"}
    for q in queries:
        words = q.lower().split()
        for i, w in enumerate(words):
            if w in location_prepositions and i + 1 < len(words):
                recent_location = " ".join(words[i+1:])
                break
        if recent_location:
            break

    conn = _conn()

    # Build a scored recommendation query
    # Score = search_count (popularity) + keyword match bonus
    clauses = []
    params = []

    # Keyword match on title
    kw_list = list(topic_keywords)[:15]
    if kw_list:
        kw_parts = " OR ".join(["title LIKE ?" for _ in kw_list])
        clauses.append(f"({kw_parts})")
        params.extend([f"%{kw}%" for kw in kw_list])

    where_sql = f"verified = 1 AND cancelled = 0 AND ({' OR '.join(clauses)})" if clauses else "verified = 1 AND cancelled = 0"

    sql = f"""
    SELECT title, date_iso, location, link, search_count, website_source
    FROM events
    WHERE {where_sql}
    ORDER BY search_count DESC, date_iso ASC
    LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    # If not enough results, pad with most popular events overall
    if len(rows) < limit:
        existing_titles = {r["title"] for r in rows}
        extra = conn.execute("""
            SELECT title, date_iso, location, link, search_count, website_source
            FROM events
            WHERE verified = 1 AND cancelled = 0
            ORDER BY search_count DESC, date_iso ASC
            LIMIT ?
        """, (limit,)).fetchall()
        for r in extra:
            if r["title"] not in existing_titles and len(rows) < limit:
                rows = list(rows) + [r]

    conn.close()

    return rows, recent_location, list(topic_keywords)


# -----------------------------
# CLI Agent
# -----------------------------
def ask_agent(user_text: str) -> str:
    log_search(user_text)
    filters = parse_user_to_filters(user_text)
    has_filters = bool(filters.location or filters.start_date or filters.end_date)

    if has_filters or filters.intent_type == "specific":
        rows = query_events(filters)
        if not rows and filters.location:
            rows = semantic_search(user_text, filters.limit, filters.location)
    else:
        rows = semantic_search(user_text, filters.limit, filters.location)

    return f"Found {len(rows)} events."


# -----------------------------
# API Agent
# -----------------------------
def ask_agent_structured(user_text: str):
    log_search(user_text)
    filters = parse_user_to_filters(user_text)
    has_filters = bool(filters.location or filters.start_date or filters.end_date)

    if has_filters or filters.intent_type == "specific":
        rows = query_events(filters)
        if not rows and filters.location:
            rows = semantic_search(user_text, filters.limit, filters.location)
    else:
        rows = semantic_search(user_text, filters.limit, filters.location)

    # Fallback only when no location was specified
    if not rows and not filters.location:
        conn = _conn()
        rows = conn.execute("""
            SELECT * FROM events WHERE verified = 1 AND cancelled = 0
            ORDER BY search_count DESC, date_iso ASC LIMIT ?
        """, (filters.limit,)).fetchall()
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

    explanation = generate_explanation(user_text, events)
    return {
        "query": user_text,
        "count": len(events),
        "explanation": explanation,
        "events": events
    }


def generate_explanation(user_text, events):
    if not events:
        return "No exact matches found, showing popular events."

    titles = [e["title"] for e in events[:3]]
    prompt = f"User query: {user_text}\nTop results: {titles}\nExplain in 1 sentence why these events are relevant."

    try:
        resp = client.responses.create(model="gpt-4.1-nano", temperature=0.3, input=prompt)
        return resp.output_text.strip()
    except:
        return "Showing relevant events based on your query."