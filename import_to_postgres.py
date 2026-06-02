import json
import os
from db import get_conn, init_db, PLACEHOLDER

def import_events():
    print("Initializing DB...")
    init_db()

    with open("events_backup.json", "r") as f:
        events = json.load(f)

    print(f"Importing {len(events)} events...")
    conn = get_conn()
    cur = conn.cursor()
    p = PLACEHOLDER

    inserted = 0
    skipped = 0

    for e in events:
        try:
            cur.execute(f"""
                INSERT INTO events (
                    id, title, link, website_source, date_iso, location,
                    cancelled, organizer, verified, verification_error,
                    first_seen, last_seen, search_count, click_count, embedding
                )
                VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})
                ON CONFLICT(id) DO NOTHING
            """, (
                e.get("id"), e.get("title"), e.get("link"), e.get("website_source"),
                e.get("date_iso"), e.get("location"),
                int(e.get("cancelled") or 0),
                e.get("organizer"),
                int(e.get("verified") or 0),
                e.get("verification_error"),
                e.get("first_seen"),
                e.get("last_seen"),
                int(e.get("search_count") or 0),
                int(e.get("click_count") or 0),
                e.get("embedding")
            ))
            inserted += 1
        except Exception as ex:
            print(f"  Skipped {e.get('title')}: {ex}")
            skipped += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done — {inserted} imported, {skipped} skipped")

if __name__ == "__main__":
    import_events()