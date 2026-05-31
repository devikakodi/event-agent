from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from agent import ask_agent_structured, get_personalized_recommendations
from db import get_conn, PLACEHOLDER

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    p = PLACEHOLDER
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT link FROM events WHERE id = {p}", (event_id,))
    row = cur.fetchone()
    if row:
        link = row[0] if not hasattr(row, 'keys') else row["link"]
        cur.execute(f"UPDATE events SET click_count = COALESCE(click_count, 0) + 1 WHERE id = {p}", (event_id,))
        conn.commit()
        cur.close()
        conn.close()
        return RedirectResponse(url=link)
    cur.close()
    conn.close()
    return JSONResponse(content={"error": "not found"}, status_code=404)


@app.get("/recommend")
def recommend(limit: int =5, location: str = "New York"):
    rows, user_location, topics, popular = get_personalized_recommendations(limit, location)
    return JSONResponse(content={
        "based_on": {
            "recent_location": user_location,
            "topics": list(topics)[:8],
        },
        "recommendations": [
            {
                "id": r["id"] if "id" in r.keys() else "",
                "title": r["title"],
                "date": r["date_iso"],
                "location": r["location"],
                "organizer": r["organizer"] if r["organizer"] else None,
                "link": r["link"],
                "popularity": r["search_count"],
            }
            for r in rows
        ],
        "popular": [
            {
                "id": r["id"] if "id" in r.keys() else "",
                "title": r["title"],
                "date": r["date_iso"],
                "location": r["location"],
                "organizer": r["organizer"] if r["organizer"] else None,
                "link": r["link"],
                "popularity": r["click_count"],
            }
            for r in popular
        ]
    })


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Event Intelligence</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f9f9f9; color: #111; }

        .header { background: #1a1a2e; color: white; padding: 20px 40px; }
        .header h1 { font-size: 20px; font-weight: 600; letter-spacing: 0.3px; }

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
        <h1>&#128269; Event Intelligence</h1>
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