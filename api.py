from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from agent import ask_agent_structured, get_personalized_recommendations

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
                "title": e["title"],
                "date": e["date"],
                "location": e["location"],
                "link": e["link"],
            }
            for e in data["events"]
        ]
    })


@app.get("/recommend")
def recommend(limit: int = 10):
    rows, recent_location, topics = get_personalized_recommendations(limit)
    return JSONResponse(content={
        "based_on": {
            "recent_location": recent_location,
            "topics": list(topics)[:8],
        },
        "recommendations": [
            {
                "title": r["title"],
                "date": r["date_iso"],
                "location": r["location"],
                "link": r["link"],
                "popularity": r["search_count"],
            }
            for r in rows
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

        /* Result header line e.g. "Returned 7 events for events in New York" */
        .result-header {
            font-size: 14px; color: #6b7280; margin-bottom: 18px;
            padding-bottom: 10px; border-bottom: 1px solid #e5e7eb;
        }
        .result-header strong { color: #111; }

        /* Section headers for recommend */
        .section-header {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.6px; color: #6b7280; margin: 24px 0 12px;
        }
        .section-header:first-of-type { margin-top: 0; }

        /* Event row */
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

        .loading { text-align: center; color: #9ca3af; padding: 48px 0; font-size: 14px; }
        .empty   { text-align: center; color: #9ca3af; padding: 48px 0; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔎 Event Intelligence</h1>
    </div>
    <div class="container">
        <div class="search-bar">
            <input id="query" placeholder='Try "AI events in west coast" or "machine learning in Europe"' />
            <button class="btn-search" onclick="search()">Search</button>
            <button class="btn-recommend" onclick="recommend()">Recommend</button>
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

        async function recommend() {
            document.getElementById("results").innerHTML = '<div class="loading">Loading recommendations...</div>';
            const res = await fetch("/recommend");
            const data = await res.json();
            renderRecommend(data);
        }

        function eventCard(num, e) {
            return `
            <div class="event">
                <div class="event-number">${num}.</div>
                <div class="event-body">
                    <div class="event-title">${e.title}</div>
                    <div class="event-meta">
                        📅 ${e.date || 'TBD'}<br>
                        📍 ${e.location || 'Unknown'}<br>
                        🔗 <a href="${e.link}" target="_blank">${e.link}</a>
                    </div>
                </div>
            </div>`;
        }

        function renderSearch(data, query) {
            const container = document.getElementById("results");
            const events = data.events || [];

            if (!events.length) {
                container.innerHTML = '<div class="empty">No events found.</div>';
                return;
            }

            let html = `<div class="result-header">Returned <strong>${data.count} events</strong> for <strong>${query}</strong></div>`;
            events.forEach((e, i) => { html += eventCard(i + 1, e); });
            container.innerHTML = html;
        }

        function renderRecommend(data) {
            const container = document.getElementById("results");
            const recs = data.recommendations || [];
            const based = data.based_on || {};

            if (!recs.length) {
                container.innerHTML = '<div class="empty">No recommendations yet — try searching first!</div>';
                return;
            }

            // Split into personalised vs popular
            // Popular = top by search_count; personalised = rest
            const sorted = [...recs].sort((a, b) => b.popularity - a.popularity);
            const popular = sorted.slice(0, 3);
            const personalised = recs.filter(r => !popular.includes(r));

            let basedText = 'your recent searches';
            if (based.recent_location) basedText += ` in <strong>${based.recent_location}</strong>`;
            if (based.topics && based.topics.length) basedText += ` · topics: ${based.topics.slice(0,5).join(', ')}`;

            let html = `<div class="result-header">Based on ${basedText}</div>`;

            if (personalised.length) {
                html += `<div class="section-header">Based on your searches</div>`;
                personalised.forEach((e, i) => { html += eventCard(i + 1, e); });
            }

            if (popular.length) {
                html += `<div class="section-header">Popular events in your area</div>`;
                popular.forEach((e, i) => { html += eventCard(i + 1, e); });
            }

            container.innerHTML = html;
        }

        document.getElementById("query").addEventListener("keydown", e => {
            if (e.key === "Enter") search();
        });
    </script>
</body>
</html>
"""