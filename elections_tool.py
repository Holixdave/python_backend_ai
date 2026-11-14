# elections_tool.py — get_election_results
# ─────────────────────────────────────────────────────────────────────────────
# Provider chain:
#   1. AP Elections API v2 — paid, needs AP_ELECTIONS_API_KEY (US races only).
#        !! The AP adapter was written from AP's public v2 docs WITHOUT a contract key, so it has never run
#        !! against the live API. It is isolated in _ap() / _ap_card(); expect to adjust field names on first
#        !! real response. Any mismatch fails soft (falls through to the web fallback) — it can't break a turn.
#   2. Free fallback — the existing search_web tool, returned WITH a clear "unverified, not official" label.
#      Works for any country (e.g. Nigeria/INEC) because it is just a web search.
# The model is told never to name a winner unless a source says the race has been called.
# ─────────────────────────────────────────────────────────────────────────────
import os
from datetime import date, timedelta
from typing import Optional

from card_tools import ProviderError, TTLCache, clip, fail, http_json, ok, run_chain, to_float, to_int

_cache = TTLCache(ttl_s=45)
_OFFICE_IDS = {"president": "P", "senate": "S", "us_senate": "S", "house": "H", "us_house": "H", "governor": "G",
               "ballot_measure": "I", "measure": "I"}
_RULES = ("Do NOT name a winner or give numbers that are not in this result. If it does not say the race was "
          "called, say it has NOT been called, even if one side leads. Mention where the figures came from.")


def _us_general_date(year: int) -> str:
    """First Tuesday after the first Monday of November."""
    d = date(year, 11, 1)
    first_monday = d + timedelta(days=(0 - d.weekday()) % 7)
    return (first_monday + timedelta(days=1)).isoformat()


def _ap(office: str, state: str, year: int, district: Optional[int], race_name: str):
    key = os.getenv("AP_ELECTIONS_API_KEY", "").strip()
    office_id = _OFFICE_IDS.get(office)
    if not key or not office_id:
        return None
    params = {"apikey": key, "officeID": office_id, "raceTypeID": "G", "format": "json", "national": "false"}
    if state:
        params["statePostal"] = state.upper()[:2]
    if district is not None and office_id == "H":
        params["seatNum"] = district
    data = http_json("GET", f"https://api.ap.org/v2/elections/{_us_general_date(year)}", params=params)
    races = data.get("races") or []
    if race_name and office_id == "I":
        races = [r for r in races if race_name.lower() in str(r.get("seatName") or r.get("officeName") or "").lower()] or races
    return [c for c in (_ap_card(r) for r in races[:4]) if c] or None


def _ap_card(race: dict) -> Optional[dict]:
    units = race.get("reportingUnits") or []
    if not units:
        return None
    unit = next((u for u in units if str(u.get("level", "")).lower() in ("state", "")), units[0])
    cands, total = [], 0
    for c in unit.get("candidates") or []:
        votes = to_int(c.get("voteCount")) or 0
        total += votes
        cands.append({"name": " ".join(x for x in (c.get("first"), c.get("last")) if x) or c.get("last") or "?",
                      "party": c.get("party"), "votes": votes, "winner": str(c.get("winner") or "").upper() in ("X", "TRUE", "W"),
                      "incumbent": bool(c.get("incumbent"))})
    for c in cands:
        c["pct"] = round(100 * c["votes"] / total, 1) if total else None
    cands.sort(key=lambda c: -c["votes"])
    return {"type": "election", "race": clip(race.get("seatName") or race.get("officeName") or "Race", 80),
            "state": race.get("statePostal") or unit.get("statePostal"),
            "called": any(c["winner"] for c in cands), "candidates": cands[:8],
            "precincts_reporting_pct": to_float(unit.get("precinctsReportingPct")),
            "total_votes": total, "provider": "ap", "attribution": "Source: The Associated Press"}


def _web(office: str, state: str, year: int, race_name: str, country: str):
    try:
        from gpt2_functions import search_web          # lazy: pulls in Firebase/keys
    except Exception:
        raise ProviderError("web search unavailable")
    what = race_name or f"{office.replace('_', ' ')}"
    query = f"{year} {country if country and country.upper() != 'US' else ''} {state} {what} election results".replace("  ", " ").strip()
    res = search_web(query, max_results=6)
    text, sources = (res if isinstance(res, (list, tuple)) and len(res) == 2 else (str(res), []))
    if not text or not str(text).strip():
        return None
    return {"text": clip(text, 3500), "sources": (sources or [])[:6], "query": query}


def get_election_results(office: str, state: str = "", year: Optional[int] = None, district: Optional[int] = None,
                         race_name: str = "", country: str = "US") -> dict:
    """
    Look up the current or final result of ONE election race or ballot measure. Not for polls, predictions or
    "who will win". US races use the Associated Press feed when available; otherwise (and for other countries)
    this searches the web and says clearly that the figures are unofficial.

    Args:
        office: president, senate, house, governor, ballot_measure, or other
        state: US state postal code or name (leave empty for national or non-US races)
        year: election year (default: the current year)
        district: US House district number, if relevant
        race_name: measure name/number, or the office and place for 'other' (e.g. "Lagos governorship")
        country: US by default; give the country for any other election
    """
    office = str(office or "").strip().lower().replace(" ", "_")
    if not office:
        return fail("office is required (president, senate, house, governor, ballot_measure, other).")
    from datetime import datetime
    year = to_int(year) or datetime.now().year
    state, country = (state or "").strip(), (country or "US").strip()
    ck = (office, state.lower(), year, district, race_name.lower(), country.lower())
    hit = _cache.get(ck)
    if hit:
        return hit

    if country.upper() in ("US", "USA", "UNITED STATES") and office in _OFFICE_IDS:
        try:
            _, cards = run_chain([("ap", _ap)], office, state, year, to_int(district), race_name)
            lines = []
            for c in cards:
                lead = c["candidates"][0] if c["candidates"] else None
                lines.append(f"{c['race']}: " + ("CALLED — " + next(x["name"] for x in c["candidates"] if x["winner"])
                             if c["called"] else "NOT called yet") + (f"; leader {lead['name']} {lead['pct']}%" if lead and lead["pct"] is not None else ""))
            return _cache.put(ck, ok("AP results. " + " | ".join(lines) + " " + _RULES, cards[0], all_races=cards[1:]))
        except ProviderError as e:
            print(f"[ELECTIONS] AP unavailable ({e}); using web fallback")

    try:
        web = _web(office, state, year, race_name, country)
    except ProviderError as e:
        web = None
        print(f"[ELECTIONS] web fallback failed: {e}")
    if not web:
        return fail("No live election feed is available and the web search found nothing. Tell the user you can't "
                    "look up this result right now. Do not guess.")
    return _cache.put(ck, ok("UNOFFICIAL: no election feed was available, so this is from a web search (not AP, "
                             "not an official count). Tell the user that. " + _RULES, None, provider="web",
                             verified=False, search=web["text"], sources=web["sources"], query=web["query"]))
