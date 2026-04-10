import os
import json
import sqlite3
from datetime import datetime
from typing import Any, List, Optional
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

DEBUG = False

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
    limit: int = 10
    sort: str = "date_asc"
    intent_type: Optional[str] = "explore"


# -----------------------------
# Helpers
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


def _get_db_locations() -> List[str]:
    """Fetch all distinct locations stored in the DB."""
    conn = _conn()
    rows = conn.execute(
        "SELECT DISTINCT location FROM events WHERE location IS NOT NULL"
    ).fetchall()
    conn.close()
    return [r["location"] for r in rows if r["location"]]


def _resolve_locations_via_llm(user_location_text: str, db_locations: List[str]) -> List[str]:
    """
    Ask the LLM to pick which DB locations match the user's location expression.
    Returns only locations that actually exist in the DB — no hallucination possible.
    """
    if not db_locations:
        return []

    locations_list = "\n".join(f"- {l}" for l in db_locations)

    prompt = f"""You are a location matcher. The user is looking for events in: "{user_location_text}"

Here are ALL the locations that exist in the database:
{locations_list}

Return a JSON array of ONLY the locations from the list above that match the user's request.
- Use geographic knowledge (e.g. "bay area" includes San Francisco, Mountain View, San Jose, Oakland, Stanford)
- Use regional knowledge (e.g. "east coast" includes New York, Boston; "europe" includes London, Berlin, Amsterdam, Paris)
- "online" or "virtual" should match "Online"
- If nothing matches, return an empty array []
- Return ONLY a JSON array of strings, no explanation, no markdown.

Example output: ["New York, NY", "Boston, MA"]"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-nano",
            temperature=0,
            input=prompt
        )
        raw = resp.output_text.strip().replace("```json", "").replace("```", "").strip()
        matched = json.loads(raw)
        db_set = set(db_locations)
        return [l for l in matched if l in db_set]
    except Exception:
        low = user_location_text.lower()
        return [l for l in db_locations if low in l.lower() or l.lower() in low]


def _location_variants(loc: Optional[str], db_locations: Optional[List[str]] = None) -> List[str]:
    """Resolve a user location string to actual DB location values via LLM."""
    if not loc or not loc.strip():
        return []
    locations = db_locations or _get_db_locations()
    return _resolve_locations_via_llm(loc, locations)


# -----------------------------
# Query builder
# -----------------------------
def _build_query(filters: SearchFilters, db_locations: Optional[List[str]] = None):
    where = []
    params: List[Any] = []

    if filters.verified_only:
        where.append("verified = 1")

    if filters.exclude_cancelled:
        where.append("cancelled = 0")

    loc_vars = _location_variants(filters.location, db_locations)
    if loc_vars:
        loc_clauses = []
        for v in loc_vars:
            loc_clauses.append("location LIKE ?")
            params.append(f"%{v}%")
        where.append("(" + " OR ".join(loc_clauses) + ")")

    if filters.keywords:
        kw_clauses = []
        for kw in filters.keywords:
            kw_clauses.append("title LIKE ?")
            params.append(f"%{kw}%")
        where.append("(" + " OR ".join(kw_clauses) + ")")

    if filters.start_date:
        where.append("date_iso >= ?")
        params.append(filters.start_date)

    if filters.end_date:
        where.append("date_iso <= ?")
        params.append(filters.end_date)

    where_sql = " AND ".join(where) if where else "1=1"
    order_sql = "ASC" if filters.sort == "date_asc" else "DESC"

    sql = f"""
    SELECT id, title, date_iso, location, link, website_source, cancelled, verified
    FROM events
    WHERE {where_sql}
    ORDER BY
      CASE WHEN date_iso IS NULL THEN 1 ELSE 0 END,
      date_iso {order_sql},
      title ASC
    LIMIT ?
    """

    params.append(filters.limit)
    return sql, params


def query_events(filters: SearchFilters):
    db_locations = _get_db_locations()
    sql, params = _build_query(filters, db_locations)

    conn = _conn()
    rows = conn.execute(sql, params).fetchall()

    for r in rows:
        conn.execute(
            "UPDATE events SET search_count = search_count + 1 WHERE id = ?",
            (r["id"],)
        )

    conn.commit()
    conn.close()

    return rows



def embed_query(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(user_text, limit=10, location=None):
    query_vec = embed_query(user_text)

    conn = _conn()
    query = "SELECT * FROM events WHERE verified = 1 AND cancelled = 0"
    params = []

    if location:
        db_locations = _get_db_locations()
        loc_vars = _location_variants(location, db_locations)
        if loc_vars:
            loc_clauses = []
            for v in loc_vars:
                loc_clauses.append("location LIKE ?")
                params.append(f"%{v}%")
            query += " AND (" + " OR ".join(loc_clauses) + ")"

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
- keywords: list of topic/subject keywords (e.g. ["AI", "machine learning"]) — exclude location words
- location: the location the user mentioned, exactly as they said it (e.g. "bay area", "east coast", "Europe", "NYC") or null
- start_date: YYYY-MM-DD or null
- end_date: YYYY-MM-DD or null
- intent_type: one of ["specific", "explore", "vague"]

Rules:
- "specific" = clear topic + location or date (e.g. "AI events in NYC next month")
- "explore" = broad interest with some signal (e.g. "AI events", "tech conferences")
- "vague" = unclear query (e.g. "something fun", "events")
- Keep location as the user typed it — do NOT normalize or expand it.
- keywords should only be topic words, never location words.
"""
            },
            {"role": "user", "content": f"Today: {today}\nQuery: {user_text}"}
        ],
    )

    data = json.loads(resp.output_text.strip())
    data["limit"] = _safe_int(data.get("limit"), 10, 1, 25)

    try:
        return SearchFilters(**data)
    except ValidationError:
        return SearchFilters(keywords=[user_text])


# -----------------------------
# Logging
# -----------------------------
def log_search(user_text: str):
    conn = _conn()
    conn.execute(
        "INSERT INTO searches (query, timestamp) VALUES (?, ?)",
        (user_text, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# -----------------------------
# Personalization Helpers
# -----------------------------
def get_recent_queries(limit: int = 10) -> List[str]:
    conn = _conn()
    rows = conn.execute("""
        SELECT query FROM searches
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [r["query"] for r in rows]


def extract_keywords_from_queries(queries: List[str]) -> List[str]:
    keywords = []
    for q in queries:
        words = q.lower().split()
        for w in words:
            if len(w) > 3:
                keywords.append(w)
    return list(set(keywords))


def extract_location_from_queries(queries: List[str]) -> Optional[str]:
    """Extract the most recent location mentioned across past queries."""
    location_words = {"in", "at", "near", "around", "from"}
    for q in queries:
        words = q.lower().split()
        for i, w in enumerate(words):
            if w in location_words and i + 1 < len(words):
                return " ".join(words[i+1:])  # return raw tail as location hint
    return None


# -----------------------------
# Personalized Recommendation
# -----------------------------
def get_personalized_recommendations(limit: int = 5):
    queries = get_recent_queries()
    keywords = extract_keywords_from_queries(queries)
    location = extract_location_from_queries(queries)

    conn = _conn()

    clauses = []
    params = []

    if keywords:
        for kw in keywords[:10]:
            clauses.append("title LIKE ?")
            params.append(f"%{kw}%")

    where_sql = " OR ".join(clauses) if clauses else "1=1"

    sql = f"""
    SELECT title, date_iso, location, link, search_count
    FROM events
    WHERE verified = 1 AND cancelled = 0
    AND ({where_sql})
    """

    if location:
        sql += " AND location LIKE ?"
        params.append(f"%{location}%")

    sql += """
    ORDER BY search_count DESC, date_iso ASC
    LIMIT ?
    """

    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    return rows


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

    print(f"[DEBUG] filters: location={filters.location!r}, keywords={filters.keywords}, intent={filters.intent_type}")

    # Resolve location → actual DB values immediately so we can log it
    db_locations = _get_db_locations()
    resolved_locs = _location_variants(filters.location, db_locations) if filters.location else []
    print(f"[DEBUG] resolved locations: {resolved_locs}")

    # -----------------------------
    # Agent decision: ALWAYS use SQL when location is present
    # -----------------------------
    has_filters = bool(filters.location or filters.start_date or filters.end_date)

    if has_filters or filters.intent_type == "specific":
        rows = query_events(filters)
        print(f"[DEBUG] SQL search returned {len(rows)} rows")
        # If SQL found nothing but location was set, try semantic with location
        if not rows and filters.location:
            rows = semantic_search(user_text, filters.limit, filters.location)
            print(f"[DEBUG] semantic fallback returned {len(rows)} rows")
    else:
        rows = semantic_search(user_text, filters.limit, filters.location)
        print(f"[DEBUG] semantic search returned {len(rows)} rows")

    # -----------------------------
    # Fallback: only when no location filter — don't return unrelated events
    # when the user clearly asked for a specific location
    # -----------------------------
    if not rows and not filters.location:
        conn = _conn()
        rows = conn.execute("""
            SELECT * FROM events
            WHERE verified = 1 AND cancelled = 0
            ORDER BY search_count DESC, date_iso ASC
            LIMIT ?
        """, (filters.limit,)).fetchall()
        conn.close()

    # -----------------------------
    # Format output
    # -----------------------------
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

# -----------------------------
# Return
# -----------------------------
    return {
        "query": user_text,
        "count": len(events),
        "explanation": explanation,   # 👈 ADD THIS LINE
        "events": events
    }

def generate_explanation(user_text, events):
    if not events:
        return "No exact matches found, showing popular events."

    titles = [e["title"] for e in events[:3]]

    prompt = f"""
User query: {user_text}
Top results: {titles}

Explain in 1 sentence why these events are relevant.
"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-nano",
            temperature=0.3,
            input=prompt
        )
        return resp.output_text.strip()
    except:
        return "Showing relevant events based on your query."