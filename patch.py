with open('agent.py', 'r') as f:
    content = f.read()

# The core fix: after resolving locations via LLM, extract just the city names
# and search by substring. This means "San Jose, CA" becomes "%San Jose%"
# which catches "San Jose, CA", "Datariders, San Jose, CA" etc.
# Currently we do LIKE '%San Jose, CA%' which misses "Datariders, San Jose, CA"
# because that string doesn't contain "San Jose, CA" — wait, it does.
# Real issue: LLM is only returning 10-11 locations but DB has 26 events.
# Fix: extract core city from each resolved location and use that for LIKE search.

old = '''    if resolved_locs:
        loc_clauses = " OR ".join(["location LIKE ?" for _ in resolved_locs])
        where.append(f"({loc_clauses})")
        params.extend([f"%{v}%" for v in resolved_locs])'''

new = '''    if resolved_locs:
        # Extract core city names for broader matching
        # e.g. "IBM, 425 Market, San Francisco, CA" -> search for "%San Francisco%"
        # This catches venue-prefixed locations like "Datariders, Mountain View, CA"
        import re
        city_patterns = set()
        for loc in resolved_locs:
            # Split on comma, find the part that looks like a city (no digits, len > 3)
            parts = [p.strip() for p in loc.split(',')]
            for part in parts:
                if len(part) > 3 and not any(c.isdigit() for c in part) and len(part) < 30:
                    # Skip state abbreviations like "CA", "NY", "WA"
                    if not re.match(r'^[A-Z]{2}$', part):
                        city_patterns.add(part)
        
        # Use city patterns for LIKE search — much broader matching
        if city_patterns:
            loc_clauses = " OR ".join(["location LIKE ?" for _ in city_patterns])
            where.append(f"({loc_clauses})")
            params.extend([f"%{p}%" for p in city_patterns])
            print(f"[DEBUG] city_patterns used for SQL: {sorted(city_patterns)}")'''

if old in content:
    content = content.replace(old, new)
    print("✅ Fix: location matching now uses city substrings")
else:
    print("❌ Pattern not found")
    idx = content.find('if resolved_locs:')
    print(repr(content[idx:idx+300]))

with open('agent.py', 'w') as f:
    f.write(content)