import sqlite3
from datetime import datetime

DB_PATH = "events.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        title TEXT,
        link TEXT,
        website_source TEXT,
        date_iso TEXT,
        location TEXT,
        cancelled INTEGER,
        organizer TEXT,
        verified INTEGER,
        verification_error TEXT,
        first_seen TEXT,
        last_seen TEXT,
        search_count INTEGER DEFAULT 0,
        embedding TEXT          
    )
                 
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    timestamp TEXT
    )
    """)
    try:
     conn.execute("ALTER TABLE events ADD COLUMN embedding TEXT")
    except sqlite3.OperationalError:
     pass
    conn.commit()
    conn.close()

def event_exists(event_id):
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone()
    conn.close()
    return row is not None
from openai import OpenAI
import json

client = OpenAI()

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def upsert_event(e):
    now = datetime.utcnow().isoformat()
    embedding_text = f"{e['title']} {e.get('location','')} {e.get('website_source','')}"
    if not e.get("embedding"):
     try:
        e["embedding"] = json.dumps(get_embedding(embedding_text))
     except Exception as ex:
        print("Embedding error:", ex)
        e["embedding"] = None
    conn = get_conn()
    conn.execute("""
    INSERT INTO events (
        id, title, link, website_source, date_iso, location,
        cancelled, organizer, verified, verification_error,
        first_seen, last_seen, embedding
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        title=excluded.title,
        link=excluded.link,
        website_source=excluded.website_source,
        date_iso=excluded.date_iso,
        location=excluded.location,
        cancelled=excluded.cancelled,
        organizer=excluded.organizer,
        verified=excluded.verified,
        verification_error=excluded.verification_error,
        last_seen=excluded.last_seen,
        embedding=excluded.embedding
    """, (
        e["id"], e.get("title"), e["link"], e["website_source"],
        e.get("date_iso"), e.get("location"),
        int(bool(e.get("cancelled", False))),
        e.get("organizer"),
        int(bool(e.get("verified", False))),
        e.get("verification_error"),
        e.get("first_seen", now),
        now,
        e.get("embedding")
    ))
    conn.commit()
    conn.close()
