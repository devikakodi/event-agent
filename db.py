import os
import json
from datetime import datetime

# -------------------------------------------------------
# Use PostgreSQL if DATABASE_URL is set (Render), else SQLite locally
# -------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    import psycopg2
    import psycopg2.extras

    def get_conn():
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        return conn

    PLACEHOLDER = "%s"
    
else:
    import sqlite3

    DB_PATH = "events.db"

    def get_conn():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    PLACEHOLDER = "?"


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            title TEXT,
            link TEXT,
            website_source TEXT,
            date_iso TEXT,
            location TEXT,
            cancelled INTEGER DEFAULT 0,
            organizer TEXT,
            verified INTEGER DEFAULT 0,
            verification_error TEXT,
            first_seen TEXT,
            last_seen TEXT,
            search_count INTEGER DEFAULT 0,
            click_count INTEGER DEFAULT 0,
            embedding TEXT
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS searches (
            id SERIAL PRIMARY KEY,
            query TEXT,
            timestamp TEXT
        )
    """ if DATABASE_URL else """
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            timestamp TEXT
        )
    """)

    # Add missing columns if upgrading from old schema
    if not DATABASE_URL:
        try:
            cur.execute("ALTER TABLE events ADD COLUMN embedding TEXT")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE events ADD COLUMN click_count INTEGER DEFAULT 0")
        except Exception:
            pass

    conn.commit()
    cur.close()
    conn.close()


def event_exists(event_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM events WHERE id = {PLACEHOLDER}", (event_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row is not None


from openai import OpenAI
client = OpenAI()

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def upsert_event(e):
    now = datetime.utcnow().isoformat()
    embedding_text = f"{e['title']} {e.get('location', '')} {e.get('website_source', '')}"
    if not e.get("embedding"):
        try:
            e["embedding"] = json.dumps(get_embedding(embedding_text))
        except Exception as ex:
            print("Embedding error:", ex)
            e["embedding"] = None

    conn = get_conn()
    cur = conn.cursor()
    p = PLACEHOLDER

    cur.execute(f"""
        INSERT INTO events (
            id, title, link, website_source, date_iso, location,
            cancelled, organizer, verified, verification_error,
            first_seen, last_seen, embedding
        )
        VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})
        ON CONFLICT(id) DO UPDATE SET
            title=EXCLUDED.title,
            link=EXCLUDED.link,
            website_source=EXCLUDED.website_source,
            date_iso=EXCLUDED.date_iso,
            location=EXCLUDED.location,
            cancelled=EXCLUDED.cancelled,
            organizer=EXCLUDED.organizer,
            verified=EXCLUDED.verified,
            verification_error=EXCLUDED.verification_error,
            last_seen=EXCLUDED.last_seen,
            embedding=EXCLUDED.embedding
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
    cur.close()
    conn.close()