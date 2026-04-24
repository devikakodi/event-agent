import re
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser as dateparser
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Add any event website here.
# Each entry needs just a URL.
# The scraper will figure out the rest.
# -----------------------------
EVENT_SOURCES = [
    "https://thealliance.ai/events",
    "https://www.edge-ai-vision.com/the-alliance/events/",  
    "https://aaai.org/conference/aaai/",                    
    "https://techequity-ai.org/ai-alliance-dev-2025/"
]


def stable_id(source: str, link: str, title: str) -> str:
    return hashlib.sha256(f"{source}|{link}|{title}".encode()).hexdigest()[:32]


def _extract_source_name(url: str) -> str:
    """Turn a URL into a short source label e.g. 'lu.ma/sf'"""
    url = re.sub(r'^https?://(www\.)?', '', url)
    return url.rstrip('/')


# -----------------------------
# Strategy 1: JSON-LD (most reliable — used by Eventbrite, many modern sites)
# -----------------------------
def _extract_from_jsonld(soup: BeautifulSoup, base_url: str, source: str) -> list:
    events = []
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue
            # Unwrap @graph
            if "@graph" in item:
                items += [x for x in item["@graph"] if isinstance(x, dict)]
                continue

            t = item.get("@type", "")
            types = t if isinstance(t, list) else [t]
            if not any("event" in str(x).lower() for x in types):
                continue

            title = item.get("name", "").strip()
            link = item.get("url", "").strip()
            start = item.get("startDate", "")
            location_obj = item.get("location", {})

            if not title or not link:
                continue

            # Parse date
            date_iso = None
            if start:
                try:
                    date_iso = dateparser.parse(start).date().isoformat()
                except Exception:
                    pass

            # Parse location
            location = ""
            if isinstance(location_obj, dict):
                name = location_obj.get("name", "")
                addr = location_obj.get("address", {})
                if isinstance(addr, dict):
                    city = addr.get("addressLocality", "")
                    state = addr.get("addressRegion", "")
                    location = f"{city}, {state}".strip(", ") if city or state else name
                elif isinstance(addr, str):
                    location = addr
                if not location:
                    location = name
            elif isinstance(location_obj, str):
                location = location_obj

            if not link.startswith("http"):
                link = base_url.rstrip("/") + "/" + link.lstrip("/")

            events.append({
                "id": stable_id(source, link, title),
                "title": title,
                "link": link,
                "website_source": source,
                "date_iso": date_iso,
                "location": location or None,
                "first_seen": datetime.utcnow().isoformat()
            })

    return events


# -----------------------------
# Strategy 2: <time> tags + nearby title/link
# -----------------------------
def _extract_from_time_tags(soup: BeautifulSoup, base_url: str, source: str) -> list:
    events = []

    for time_tag in soup.find_all("time"):
        dt = time_tag.get("datetime", "")
        date_iso = None
        if dt:
            try:
                date_iso = dateparser.parse(dt).date().isoformat()
            except Exception:
                pass

        # Walk up to find a parent with a link and title
        parent = time_tag.parent
        for _ in range(6):  # look up to 6 levels
            if parent is None:
                break
            a = parent.find("a", href=True)
            if a:
                title = a.get_text(" ", strip=True)
                href = a.get("href", "")
                if title and len(title) > 5:
                    if not href.startswith("http"):
                        href = base_url.rstrip("/") + "/" + href.lstrip("/")
                    events.append({
                        "id": stable_id(source, href, title),
                        "title": title,
                        "link": href,
                        "website_source": source,
                        "date_iso": date_iso,
                        "location": None,
                        "first_seen": datetime.utcnow().isoformat()
                    })
                    break
            parent = parent.parent if hasattr(parent, 'parent') else None

    return events


# -----------------------------
# Strategy 3: thealliance.ai specific (keep original logic)
# -----------------------------
DATE_RE = re.compile(r"^([A-Z][a-z]{2})\s+(\d{1,2})\s+(.*)$")

def _extract_alliance(soup: BeautifulSoup, base_url: str, source: str) -> list:
    events = []
    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href")
        if not href or " @ " not in text:
            continue
        m = DATE_RE.match(text)
        if not m:
            continue
        month_abbr, day_str, rest = m.group(1), m.group(2), m.group(3)
        title_part, location_part = rest.rsplit(" @ ", 1)
        title = title_part.strip()
        location = location_part.strip()
        link = href if href.startswith("http") else f"{base_url.rstrip('/')}{href}"
        try:
            date_iso = dateparser.parse(f"{month_abbr} {day_str} {datetime.now().year}").date().isoformat()
        except Exception:
            date_iso = None
        events.append({
            "id": stable_id(source, link, title),
            "title": title,
            "link": link,
            "website_source": source,
            "date_iso": date_iso,
            "location": location,
            "first_seen": datetime.utcnow().isoformat()
        })
    return events


# -----------------------------
# Strategy 4: LLM fallback — for sites none of the above can parse
# -----------------------------
def _extract_via_llm(page_text: str, base_url: str, source: str) -> list:
    """
    Send page text to LLM, ask it to extract events as JSON.
    Used as a last resort for unusual site structures.
    """
    page_text = page_text[:6000]  # keep prompt reasonable

    prompt = f"""Extract all events from the following webpage text.
Return a JSON array of events. Each event should have:
- title (string)
- date (YYYY-MM-DD or null)
- location (city/venue string or null)
- link (full URL or path, or null)

Only include real events with at least a title.
Return ONLY a JSON array, no explanation, no markdown.

Page URL: {base_url}
Page text:
{page_text}"""

    try:
        resp = client.responses.create(
            model="gpt-4.1-nano",
            temperature=0,
            input=prompt
        )
        raw = resp.output_text.strip().replace("```json", "").replace("```", "").strip()
        items = json.loads(raw)

        events = []
        for item in items:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            link = item.get("link") or base_url
            if link and not link.startswith("http"):
                link = base_url.rstrip("/") + "/" + link.lstrip("/")
            events.append({
                "id": stable_id(source, link, item["title"]),
                "title": item["title"].strip(),
                "link": link,
                "website_source": source,
                "date_iso": item.get("date"),
                "location": item.get("location"),
                "first_seen": datetime.utcnow().isoformat()
            })
        return events

    except Exception as ex:
        print(f"  LLM extraction failed for {source}: {ex}")
        return []


# -----------------------------
# Deduplicate within a scrape run
# -----------------------------
def _deduplicate(events: list) -> list:
    seen = set()
    result = []
    for e in events:
        key = e["id"]
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


# -----------------------------
# Main scraper — tries each strategy in order
# -----------------------------
def scrape_url(url: str) -> list:
    source = _extract_source_name(url)
    base_url = re.match(r'https?://[^/]+', url).group(0)

    print(f"  Fetching {url}...")
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
    except Exception as ex:
        print(f"  ❌ Failed to fetch {url}: {ex}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # Try each strategy, use first one that yields results
    events = []

    # Strategy 1: JSON-LD
    events = _extract_from_jsonld(soup, base_url, source)
    if events:
        print(f"  ✅ JSON-LD: found {len(events)} events")
        return _deduplicate(events)

    # Strategy 2: thealliance.ai pattern (date @ location in anchor text)
    events = _extract_alliance(soup, base_url, source)
    if events:
        print(f"  ✅ Alliance pattern: found {len(events)} events")
        return _deduplicate(events)

    # Strategy 3: <time> tags
    events = _extract_from_time_tags(soup, base_url, source)
    if events:
        print(f"  ✅ Time tags: found {len(events)} events")
        return _deduplicate(events)

    # Strategy 4: LLM fallback
    print(f"  ⚠️  No pattern matched, trying LLM extraction...")
    events = _extract_via_llm(page_text, base_url, source)
    if events:
        print(f"  ✅ LLM: found {len(events)} events")
        return _deduplicate(events)

    print(f"  ❌ No events found at {url}")
    return []


# -----------------------------
# Entry point — scrapes all sources
# -----------------------------
def scrape_events() -> list:
    all_events = []
    for url in EVENT_SOURCES:
        print(f"\nScraping: {url}")
        events = scrape_url(url)
        all_events.extend(events)

    print(f"\nTotal scraped across all sources: {len(all_events)}")
    return all_events