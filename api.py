from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from agent import ask_agent_structured, get_personalized_recommendations
from db import get_conn, init_db
from datetime import datetime
import os
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Track last scrape time
# -----------------------------
_last_scraped: str = "Never"

def _run_scrape():
    global _last_scraped
    try:
        from main import run
        print("[Scheduler] Starting scheduled scrape...")
        run()
        _last_scraped = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        print(f"[Scheduler] Done. Last scraped: {_last_scraped}")
    except Exception as e:
        print(f"[Scheduler] Scrape failed: {e}")

# -----------------------------
# Auto-schedule daily scrape
# -----------------------------
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(_run_scrape, 'interval', hours=24)
    scheduler.start()
    print("[Scheduler] Daily scrape scheduled.")
except Exception as e:
    print(f"[Scheduler] APScheduler not available: {e}")

# -----------------------------
# Keep-alive ping (every 10 min)
# -----------------------------
def _keep_alive():
    import time
    import urllib.request
    url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not url:
        return
    while True:
        try:
            urllib.request.urlopen(url, timeout=10)
            print("[KeepAlive] Pinged.")
        except Exception:
            pass
        time.sleep(600)

if os.getenv("RENDER_EXTERNAL_URL"):
    t = threading.Thread(target=_keep_alive, daemon=True)
    t.start()
    print("[KeepAlive] Started.")


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
def search_events(request: SearchRequest):
    data = ask_agent_structured(request.query)
    return JSONResponse(content={
        "count": data["count"],
        "events": [
            {
                "id": e.get("id", ""),
                "title": e["title"],
                "date": e["date"],
                "location": e["location"],
                "organizer": e.get("organizer"),
                "link": e["link"],
            }
            for e in data["events"]
        ],
        "past_events": [
            {
                "id": e.get("id", ""),
                "title": e["title"],
                "date": e["date"],
                "location": e["location"],
                "organizer": e.get("organizer"),
                "link": e["link"],
            }
            for e in data.get("past_events", [])
        ]
    })


@app.get("/click/{event_id}")
def track_click(event_id: str):
    from agent import _P, _conn, _fetchall, _execute
    p = _P()
    conn = _conn()
    cur = conn.cursor()
    cur.execute(f"SELECT link FROM events WHERE id = {p}", (event_id,))
    row = cur.fetchone()
    if row:
        link = row[0] if not isinstance(row, dict) else row["link"]
        cur.execute(f"UPDATE events SET click_count = COALESCE(click_count, 0) + 1 WHERE id = {p}", (event_id,))
        conn.commit()
        cur.close()
        conn.close()
        return RedirectResponse(url=link)
    cur.close()
    conn.close()
    return JSONResponse(content={"error": "not found"}, status_code=404)


@app.get("/recommend")
def recommend(limit: int = 5, location: str = "New York"):
    rows, user_location, topics, popular = get_personalized_recommendations(limit, location)
    return JSONResponse(content={
        "based_on": {
            "recent_location": user_location,
            "topics": list(topics)[:8],
        },
        "recommendations": [
            {
                "id": r["id"] if isinstance(r, dict) and "id" in r else "",
                "title": r["title"] if isinstance(r, dict) else r[0],
                "date": r["date_iso"] if isinstance(r, dict) else r[1],
                "location": r["location"] if isinstance(r, dict) else r[2],
                "organizer": r.get("organizer") if isinstance(r, dict) else None,
                "link": r["link"] if isinstance(r, dict) else r[3],
                "popularity": r.get("search_count", 0) if isinstance(r, dict) else 0,
            }
            for r in rows
        ],
        "popular": [
            {
                "id": r["id"] if isinstance(r, dict) and "id" in r else "",
                "title": r["title"] if isinstance(r, dict) else r[0],
                "date": r["date_iso"] if isinstance(r, dict) else r[1],
                "location": r["location"] if isinstance(r, dict) else r[2],
                "organizer": r.get("organizer") if isinstance(r, dict) else None,
                "link": r["link"] if isinstance(r, dict) else r[3],
                "popularity": r.get("click_count", 0) if isinstance(r, dict) else 0,
            }
            for r in popular
        ]
    })


@app.get("/status")
def status():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events WHERE verified = 1 AND cancelled = 0")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM events WHERE verified = 1 AND cancelled = 0 AND date_iso >= %s" if os.getenv("DATABASE_URL") else "SELECT COUNT(*) FROM events WHERE verified = 1 AND cancelled = 0 AND date_iso >= ?", (datetime.utcnow().date().isoformat(),))
        upcoming = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT website_source) FROM events")
        sources = cur.fetchone()[0]
        cur.close()
        conn.close()
        return JSONResponse(content={
            "status": "ok",
            "event_count": total,
            "upcoming_events": upcoming,
            "sources": sources,
            "last_scraped": _last_scraped,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AllyCat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f9f9f9; color: #111; }

        .header { background: #1a1a2e; color: white; padding: 20px 40px; display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 20px; font-weight: 600; letter-spacing: 0.3px; }
        .header-meta { font-size: 12px; color: #9ca3af; text-align: right; }
        .header-meta span { display: block; }

        .container { max-width: 720px; margin: 36px auto; padding: 0 20px; }

        .search-bar { display: flex; gap: 8px; margin-bottom: 28px; }
        .search-bar input {
            flex: 1; padding: 11px 15px; border: 1px solid #d1d5db;
            border-radius: 7px; font-size: 14px; outline: none; background: white;
        }
        .search-bar input:focus { border-color: #4f46e5; }
        button {
            padding: 11px 18px; border: none; border-radius: 7px;
            font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap;
        }
        .btn-search { background: #4f46e5; color: white; }
        .btn-search:hover { background: #4338ca; }
        .btn-recommend { background: #059669; color: white; }
        .btn-recommend:hover { background: #047857; }

        .result-header {
            font-size: 14px; color: #6b7280; margin-bottom: 18px;
            padding-bottom: 10px; border-bottom: 1px solid #e5e7eb;
        }
        .result-header strong { color: #111; }

        .section-header {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.6px; color: #6b7280; margin: 24px 0 12px;
        }
        .section-header:first-of-type { margin-top: 0; }

        .event {
            background: white; border: 1px solid #e5e7eb; border-radius: 9px;
            padding: 14px 16px; margin-bottom: 10px;
            display: flex; gap: 14px; align-items: flex-start;
        }
        .event-number {
            font-size: 13px; font-weight: 700; color: #9ca3af;
            min-width: 22px; padding-top: 1px;
        }
        .event-body { flex: 1; }
        .event-title { font-size: 15px; font-weight: 600; color: #111; margin-bottom: 5px; }
        .event-meta { font-size: 13px; color: #6b7280; line-height: 1.7; }
        .event-meta a { color: #4f46e5; text-decoration: none; }
        .event-meta a:hover { text-decoration: underline; }

        .location-box {
            margin-top: 20px; padding: 14px; background: white;
            border: 1px solid #e5e7eb; border-radius: 9px;
        }
        .location-box p { font-size: 13px; color: #6b7280; margin-bottom: 8px; }
        .location-box .row { display: flex; gap: 8px; }
        .location-box input {
            flex: 1; padding: 9px 12px; border: 1px solid #d1d5db;
            border-radius: 7px; font-size: 13px; outline: none;
        }
        .location-box button {
            padding: 9px 16px; background: #059669; color: white;
            border: none; border-radius: 7px; font-size: 13px; font-weight: 600; cursor: pointer;
        }

        .loading { text-align: center; color: #9ca3af; padding: 48px 0; font-size: 14px; }
        .empty   { text-align: center; color: #9ca3af; padding: 48px 0; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>&#128049; AllyCat</h1>
        <div class="header-meta">
            <span id="event-count">Loading...</span>
            <span id="last-updated"></span>
        </div>
    </div>
    <div class="container">
        <div class="search-bar">
            <input id="query" placeholder='Try "AI events in west coast" or "machine learning in Europe"' />
            <button class="btn-search" onclick="search()">Search</button>
            <button class="btn-recommend" onclick="recommend('New York')">Recommend</button>
        </div>
        <div id="results"></div>
    </div>

    <script>
        // Load status on page load
        async function loadStatus() {
            try {
                const res = await fetch("/status");
                const data = await res.json();
                document.getElementById("event-count").textContent = data.event_count + " events across " + data.sources + " sources";
                if (data.last_scraped && data.last_scraped !== "Never") {
                    document.getElementById("last-updated").textContent = "Last updated: " + data.last_scraped;
                }
            } catch(e) {}
        }
        loadStatus();

        async function search() {
            const query = document.getElementById("query").value.trim();
            if (!query) return;
            document.getElementById("results").innerHTML = '<div class="loading">Searching...</div>';
            const res = await fetch("/search", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({query})
            });
            const data = await res.json();
            renderSearch(data, query);
        }

        async function recommend(location) {
            document.getElementById("results").innerHTML = '<div class="loading">Loading recommendations...</div>';
            const loc = location || "";
            const res = await fetch("/recommend?location=" + encodeURIComponent(loc));
            const data = await res.json();
            renderRecommend(data);
        }

        function eventCard(num, e) {
            const organizer = e.organizer ? "<br>&#127970; " + e.organizer : "";
            const link = e.id ? "/click/" + e.id : e.link;
            return `
            <div class="event">
                <div class="event-number">${num}.</div>
                <div class="event-body">
                    <div class="event-title">${e.title}</div>
                    <div class="event-meta">
                        &#128197; ${e.date || 'TBD'}<br>
                        &#128205; ${e.location || 'Unknown'}${organizer}<br>
                        &#128279; <a href="${link}" target="_blank">${e.link}</a>
                    </div>
                </div>
            </div>`;
        }

        function renderSearch(data, query) {
            const container = document.getElementById("results");
            const events = data.events || [];
            const past = data.past_events || [];

            if (!events.length && !past.length) {
                container.innerHTML = '<div class="empty">No events found.</div>';
                return;
            }

            let html = '';
            if (events.length) {
                html += `<div class="result-header">Returned <strong>${data.count} events</strong> for <strong>${query}</strong></div>`;
                events.forEach((e, i) => { html += eventCard(i + 1, e); });
            }

            if (past.length) {
                html += `<div class="result-header" style="margin-top:24px; color:#b45309;">No upcoming events found — here are the last ${past.length} past events:</div>`;
                past.forEach((e, i) => { html += eventCard(i + 1, e); });
            }

            container.innerHTML = html;
        }

        function renderRecommend(data) {
            const container = document.getElementById("results");
            const recs = data.recommendations || [];
            const popular = data.popular || [];
            const based = data.based_on || {};

            if (!recs.length && !popular.length) {
                container.innerHTML = '<div class="empty">No recommendations yet — try searching first!</div>';
                return;
            }

            let basedText = 'your recent searches';
            if (based.topics && based.topics.length) basedText += ` &middot; topics: ${based.topics.slice(0,5).join(', ')}`;
            let html = `<div class="result-header">Based on ${basedText}</div>`;

            if (recs.length) {
                html += `<div class="section-header">Based on your searches</div>`;
                recs.forEach((e, i) => { html += eventCard(i + 1, e); });
            }

            html += `<div class="section-header">Popular events in ${based.recent_location || 'your area'}</div>`;
            if (popular.length) {
                popular.forEach((e, i) => { html += eventCard(i + 1, e); });
            } else {
                html += `<div class="empty" style="padding:16px 0; text-align:left; font-size:13px; color:#9ca3af;">No popular events yet for this area — popularity is based on link clicks.</div>`;
            }

            html += `
            <div class="location-box">
                <p>Show popular events in a specific location:</p>
                <div class="row">
                    <input id="ploc" placeholder="e.g. New York, San Francisco, London..." />
                    <button onclick="recommend(document.getElementById('ploc').value)">Go</button>
                </div>
            </div>`;

            container.innerHTML = html;

            document.getElementById("ploc").addEventListener("keydown", e => {
                if (e.key === "Enter") recommend(document.getElementById("ploc").value);
            });
        }

        document.getElementById("query").addEventListener("keydown", e => {
            if (e.key === "Enter") search();
        });
    </script>
</body>
</html>
"""