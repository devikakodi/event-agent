import os
import json
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You verify whether a webpage represents a real event.

Return JSON ONLY with:
- is_event (boolean)
- cancelled (boolean)
- organizer (string or null)
- confidence (0..1)

Rules:
- Do NOT guess dates.
- Only decide if it's an event, cancelled, and organizer if clear.
"""

def _extract_date_from_time_tags(soup: BeautifulSoup):
    # <time datetime="2025-10-22T09:00:00Z">
    for t in soup.find_all("time"):
        dt = t.get("datetime")
        if not dt:
            continue
        try:
            return dateparser.parse(dt).date().isoformat()
        except Exception:
            pass
    return None

def _extract_date_from_jsonld(soup: BeautifulSoup):
    # Look for JSON-LD with @type Event and startDate
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for s in scripts:
        raw = (s.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        # JSON-LD can be dict or list
        candidates = data if isinstance(data, list) else [data]

        for obj in candidates:
            if not isinstance(obj, dict):
                continue

            # Sometimes wrapped in @graph
            if "@graph" in obj and isinstance(obj["@graph"], list):
                candidates.extend([x for x in obj["@graph"] if isinstance(x, dict)])

            t = obj.get("@type") or obj.get("type")
            # @type can be list
            is_event = False
            if isinstance(t, str) and t.lower() == "event":
                is_event = True
            elif isinstance(t, list) and any(isinstance(x, str) and x.lower() == "event" for x in t):
                is_event = True

            if not is_event:
                continue

            start = obj.get("startDate") or obj.get("start_date")
            if not start:
                continue
            try:
                return dateparser.parse(start).date().isoformat()
            except Exception:
                continue

    return None

def _deterministic_date_extract(html: str):
    soup = BeautifulSoup(html, "html.parser")
    d = _extract_date_from_jsonld(soup)
    if d:
        return d
    d = _extract_date_from_time_tags(soup)
    return d

def verify_event(event):
    url = event["link"]

    # Fetch page safely
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text
    except requests.exceptions.SSLError as ex:
        return {
            **event,
            "verified": False,
            "cancelled": False,
            "organizer": None,
            "verification_error": f"ssl_error: {ex}",
        }
    except requests.exceptions.RequestException as ex:
        return {
            **event,
            "verified": False,
            "cancelled": False,
            "organizer": None,
            "verification_error": f"request_error: {ex}",
        }

    # Deterministic date (fixes wrong years)
    extracted_date = _deterministic_date_extract(html)
    if extracted_date:
        event["date_iso"] = extracted_date

    # Use Nano only for: is_event / cancelled / organizer
    # Keep context short and clean
    page_text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    page_text = page_text[:4500]

    try:
        response = client.responses.create(
            model="gpt-4.1-nano",
            temperature=0,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": page_text}
            ],
        )
        data = json.loads(response.output_text.strip())
        print("LLM response:", data)
    except Exception as ex:
        return {
            **event,
            "verified": False,
            "cancelled": False,
            "organizer": None,
            "verification_error": f"llm_error: {ex}",
        }

    verified = bool(data.get("is_event")) and float(data.get("confidence", 0)) >= 0.7

    return {
        **event,
        "verified": verified,
        "cancelled": bool(data.get("cancelled", False)),
        "organizer": data.get("organizer"),
        "verification_error": None if verified else "not_confident_or_not_event",
    }