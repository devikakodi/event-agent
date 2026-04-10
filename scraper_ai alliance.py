import requests
import re
import hashlib
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser

URL = "https://thealliance.ai/events"
SOURCE = "thealliance.ai/events"

DATE_RE = re.compile(r"^([A-Z][a-z]{2})\s+(\d{1,2})\s+(.*)$")  # "Nov 13 rest..."

def stable_id(link, title):
    return hashlib.sha256(f"{SOURCE}|{link}|{title}".encode()).hexdigest()[:32]

def scrape_events():
    html = requests.get(URL, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")

    events = []

    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href")

        if not href:
            continue
        if " @ " not in text:
            continue

        m = DATE_RE.match(text)
        if not m:
            continue

        month_abbr, day_str, rest = m.group(1), m.group(2), m.group(3)

        # Split on the LAST " @ " in case title contains "@"
        title_part, location_part = rest.rsplit(" @ ", 1)
        title = title_part.strip()
        location = location_part.strip()

        link = href if href.startswith("http") else f"https://thealliance.ai{href}"

        # Best-effort year: assume current year at scrape time (LLM verifier can override)
        try:
            date_iso = parser.parse(f"{month_abbr} {day_str} {datetime.now().year}").date().isoformat()
        except Exception:
            date_iso = None

        events.append({
            "id": stable_id(link, title),
            "title": title,
            "link": link,
            "website_source": SOURCE,
            "date_iso": date_iso,
            "location": location,
            "first_seen": datetime.utcnow().isoformat()
        })

    return events
