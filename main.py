from db import init_db, event_exists, upsert_event
from scraper import scrape_events
from verifier import verify_event

def run():
    print("Initializing DB...")
    init_db()
    print("Scraping events...")
    scraped = scrape_events() 
    print(f"Scraped {len(scraped)} events")
    
    inserted = 0
    verified_added = 0

    for i, e in enumerate(scraped):
        print(f"Verifying {i+1}/{len(scraped)}: {e['title']}")
        is_new = not event_exists(e["id"])

        # verify (handles SSL errors internally and won't crash)
        v = verify_event(e)

        # ALWAYS store it so you can see everything
        upsert_event(v)

        if is_new:
            inserted += 1
            if v.get("verified"):
                verified_added += 1

    print(f"Stored {inserted} new scraped events (verified among new: {verified_added}).")

if __name__ == "__main__":
    run()
    import agent
    agent._db_locations_cache = None
