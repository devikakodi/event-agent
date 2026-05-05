# Intelligent Event Discovery System

An agentic search platform that aggregates AI and tech events from multiple websites, verifies them using an LLM, and lets users discover them through natural language queries.

Built at NYU in partnership with the AI Alliance and IBM.

🔗 **Interface:** 
<img width="539" height="370" alt="Screenshot 2026-05-05 at 4 11 35 PM" src="https://github.com/user-attachments/assets/30f0e8af-6429-47d3-a3aa-d458c99a4101" />

---

## What It Does

The system scrapes event data from four websites, verifies each event's legitimacy using GPT-4.1 with a confidence threshold of 0.7, stores everything in a SQLite database with vector embeddings, and exposes it through a FastAPI web interface. Users can search in plain English — "AI events in bay area next month" — and get back formatted results with event name, date, location, organizer, and link.

---

## Features

- **Multi-site Scraper** — scrapes 4 sources using 4 strategies in order: JSON-LD structured data, HTML time tags, Alliance regex pattern, and LLM fallback. LLM only extracts what exists on the page, no hallucination. Can work on any website, just edit the list in the code.
<img width="287" height="84" alt="Screenshot 2026-05-05 at 4 04 58 PM" src="https://github.com/user-attachments/assets/425bbc75-dfe3-4b9d-b607-7b477564e134" />

- **LLM Verification** — every scraped event is verified by a separate verifier module using GPT-4.1-nano with a confidence threshold.
<img width="411" height="222" alt="Screenshot 2026-05-05 at 4 05 12 PM" src="https://github.com/user-attachments/assets/f4029211-ab55-4c1f-b12c-30483bbfd466" />

- **Smart Location Search** — LLM resolves any region, city, or continent the user types. Handles "bay area", "east coast", "europe", "asia" with no hardcoding.
<img width="452" height="532" alt="Screenshot 2026-05-05 at 4 14 21 PM" src="https://github.com/user-attachments/assets/f1cd59ae-39dd-4241-b050-c57f2fa6dde9" /> <img width="444" height="367" alt="Screenshot 2026-05-05 at 4 14 51 PM" src="https://github.com/user-attachments/assets/98f5d005-1e2b-4cae-b34d-837e2d84dc77" />



- **Date-range Filtering** — supports natural time expressions like "next week" or "in December". Today's date is passed to the LLM with every query.
<img width="547" height="344" alt="Screenshot 2026-05-05 at 4 08 02 PM" src="https://github.com/user-attachments/assets/c0a49093-eb1a-4e5a-bad2-6ba30650b7b5" />

- **Personalized Recommendations** — logs every search with a timestamp, extracts topic keywords and location phrases from the last 20 searches, and returns matching events.
<img width="494" height="580" alt="Screenshot 2026-05-05 at 4 15 45 PM" src="https://github.com/user-attachments/assets/7f15e48d-4ce0-442c-9386-8f02e7d846de" />


- **Click-based Popularity** — ranks popular events by actual link clicks, not search count. Each click is permanently recorded.
<img width="464" height="197" alt="Screenshot 2026-05-05 at 4 09 28 PM" src="https://github.com/user-attachments/assets/d90e19e2-3553-4edb-85af-b650f0be7f15" />
<img width="432" height="149" alt="Screenshot 2026-05-05 at 4 09 36 PM" src="https://github.com/user-attachments/assets/39b1d07f-d4d5-45de-ab0a-44636dbbb0f2" />
<img width="368" height="361" alt="Screenshot 2026-05-05 at 4 10 02 PM" src="https://github.com/user-attachments/assets/e9d6f515-403c-4f57-ad0b-1f3489dba2a7" />

- **Deployed API** — FastAPI backend with /search and /recommend endpoints, CORS enabled, callable from any external application.
<img width="354" height="156" alt="Screenshot 2026-05-05 at 4 10 27 PM" src="https://github.com/user-attachments/assets/00d9ffad-19b3-419b-9d90-50a1c9497a40" />

---

## Tech Stack

- **Language:** Python
- **Framework:** FastAPI
- **Database:** SQLite
- **AI:** OpenAI GPT-4.1 (location resolution), GPT-4.1-nano (event verification), text-embedding-3-small (semantic search)
- **Libraries:** BeautifulSoup, Pydantic, NumPy
- **Deployment:** Render (auto-deploy from GitHub)

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/devikakodi/event-agent.git
cd event-agent

# Install dependencies
pip install -r requirements.txt

# Run the scraper and build the database
python main.py

# Start the API and web UI
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

---

## Project Structure

| File | Description |
|------|-------------|
| main.py | Orchestrates scrape → verify → store pipeline |
| scraper.py | Fetches events from multiple websites |
| verifier.py | LLM confirms each event is real |
| db.py | SQLite storage and vector embeddings |
| agent.py | Query parsing and location resolution |
| api.py | FastAPI endpoints and web UI |

---

## Next Steps

- Auto-scrape on a schedule so the database stays fresh automatically
- Migrate to PostgreSQL for persistent storage across deployments
- Add more event sources (one-line change with current scraper architecture)
- Improve Render cold start speed for faster live demo loading

---
