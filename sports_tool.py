# sports_tool.py — get_sports (scores, standings)
# ─────────────────────────────────────────────────────────────────────────────
# Provider chain:
#   1. API-Football (api-sports.io) — paid/keyed (API_FOOTBALL_KEY), soccer leagues only
#   2. ESPN public site API — free, no key, every league below
# Both are normalised to the same card shape, so the app doesn't care which one answered.
# NOTE: the ESPN endpoints are public but unofficial; if ESPN changes them this tool degrades to an error
# message (never a crash) and the model falls back to search_web.
# ─────────────────────────────────────────────────────────────────────────────
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from card_tools import ProviderError, TTLCache, clip, fail, http_json, ok, run_chain, to_int

_cache_scores = TTLCache(ttl_s=30)         # live scores move; keep it short
_cache_table = TTLCache(ttl_s=600)

# league key -> (espn "sport/league", api-football league id or None, display name)
LEAGUES = {
    "nfl": ("football/nfl", None, "NFL"), "nba": ("basketball/nba", None, "NBA"),
    "mlb": ("baseball/mlb", None, "MLB"), "nhl": ("hockey/nhl", None, "NHL"),
    "wnba": ("basketball/wnba", None, "WNBA"),
    "ncaafb": ("football/college-football", None, "College football"),
    "ncaamb": ("basketball/mens-college-basketball", None, "College basketball"),
    "epl": ("soccer/eng.1", 39, "Premier League"), "la_liga": ("soccer/esp.1", 140, "La Liga"),
    "serie_a": ("soccer/ita.1", 135, "Serie A"), "bundesliga": ("soccer/ger.1", 78, "Bundesliga"),
    "ligue_1": ("soccer/fra.1", 61, "Ligue 1"), "mls": ("soccer/usa.1", 253, "MLS"),
    "champions_league": ("soccer/uefa.champions", 2, "Champions League"),
    "europa_league": ("soccer/uefa.europa", 3, "Europa League"),
    "world_cup": ("soccer/fifa.world", 1, "World Cup"),
}
_ALIASES = {"premier league": "epl", "english premier league": "epl", "laliga": "la_liga", "la liga": "la_liga",
            "serie a": "serie_a", "ligue 1": "ligue_1", "ucl": "champions_league", "uefa champions league": "champions_league",
            "champions league": "champions_league", "europa league": "europa_league", "fifa world cup": "world_cup",
            "college football": "ncaafb", "college basketball": "ncaamb", "football": "nfl", "soccer": "epl"}
MAX_GAMES = 20
MAX_ROWS = 30


def _norm_league(league: str) -> Optional[str]:
    s = str(league or "").strip().lower().replace("-", "_")
    s = _ALIASES.get(s.replace("_", " "), s)
    return s if s in LEAGUES else None


def _status(state: str, completed: bool, detail: str) -> tuple:
    state = (state or "").lower()
    if completed or state == "post":
        return ("postponed" if "postpon" in detail.lower() or "cancel" in detail.lower() else "final"), detail or "Final"
    if state == "in":
        return "live", detail or "Live"
    return "upcoming", detail or "Scheduled"


# ── ESPN (free) ─────────────────────────────────────────────────────────────
def _espn_scores(league: str, date: str, days: int):
    path = LEAGUES[league][0]
    params = {}
    if date:
        d0 = datetime.strptime(date, "%Y-%m-%d")
        d1 = d0 + timedelta(days=max(0, days - 1))
        params["dates"] = d0.strftime("%Y%m%d") if days <= 1 else f"{d0:%Y%m%d}-{d1:%Y%m%d}"
    data = http_json("GET", f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard", params=params)
    games = []
    for ev in data.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        st = (ev.get("status") or comp.get("status") or {}).get("type") or {}
        status, status_text = _status(st.get("state"), bool(st.get("completed")), st.get("detail") or st.get("description") or "")
        sides = {}
        for c in comp.get("competitors") or []:
            t = c.get("team") or {}
            sides[c.get("homeAway") or ("home" if not sides else "away")] = {
                "name": t.get("displayName") or t.get("name") or "TBD", "abbr": t.get("abbreviation"),
                "logo": t.get("logo"), "score": to_int(c.get("score")) if status != "upcoming" else None}
        if "home" not in sides or "away" not in sides:
            continue
        games.append({"id": str(ev.get("id")), "status": status, "status_text": status_text, "start": ev.get("date"),
                      "home": sides["home"], "away": sides["away"],
                      "venue": ((comp.get("venue") or {}).get("fullName"))})
    return games


_ESPN_STATS = [("rank", "#"), ("gamesPlayed", "P"), ("wins", "W"), ("ties", "D"), ("draws", "D"), ("losses", "L"),
               ("winPercent", "PCT"), ("gamesBehind", "GB"), ("pointDifferential", "GD"), ("pointsDiff", "GD"),
               ("points", "PTS")]


def _espn_standings(league: str):
    path = LEAGUES[league][0]
    data = http_json("GET", f"https://site.api.espn.com/apis/v2/sports/{path}/standings")
    groups = data.get("children") or [data]
    rows = []
    for g in groups:
        entries = ((g.get("standings") or {}).get("entries")) or []
        for e in entries:
            t = e.get("team") or {}
            stats = {}
            for s in e.get("stats") or []:
                for key, label in _ESPN_STATS:
                    if s.get("name") == key and label not in stats:
                        stats[label] = s.get("displayValue", s.get("value"))
            if "#" not in stats:
                stats["#"] = str(len(rows) + 1)
            rows.append({"group": g.get("name") if len(groups) > 1 else None, "team": t.get("displayName") or "?",
                         "abbr": t.get("abbreviation"), "logo": ((t.get("logos") or [{}])[0]).get("href"), "stats": stats})
    return rows or None


# ── API-Football (paid) ─────────────────────────────────────────────────────
def _af_headers():
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    return {"x-apisports-key": key} if key else None


def _af_season(league: str, today: datetime) -> int:
    return today.year if league in ("mls", "world_cup") or today.month >= 7 else today.year - 1


def _af_scores(league: str, date: str, days: int):
    headers, lid = _af_headers(), LEAGUES[league][1]
    if not headers or not lid:
        return None
    today = datetime.now(timezone.utc)
    d0 = datetime.strptime(date, "%Y-%m-%d") if date else today
    params = {"league": lid, "season": _af_season(league, d0)}
    if days > 1:
        params.update({"from": d0.strftime("%Y-%m-%d"), "to": (d0 + timedelta(days=days - 1)).strftime("%Y-%m-%d")})
    else:
        params["date"] = d0.strftime("%Y-%m-%d")
    data = http_json("GET", "https://v3.football.api-sports.io/fixtures", params=params, headers=headers)
    games = []
    live_codes = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT"}
    for fx in data.get("response") or []:
        f, teams, goals = fx.get("fixture") or {}, fx.get("teams") or {}, fx.get("goals") or {}
        st = f.get("status") or {}
        code = st.get("short") or ""
        if code in ("FT", "AET", "PEN"):
            status = "final"
        elif code in live_codes:
            status = "live"
        elif code in ("PST", "CANC", "ABD", "AWD", "WO"):
            status = "postponed"
        else:
            status = "upcoming"
        text = st.get("long") or code
        if status == "live" and st.get("elapsed"):
            text = f"{st['elapsed']}'"
        home, away = teams.get("home") or {}, teams.get("away") or {}
        games.append({"id": str(f.get("id")), "status": status, "status_text": text, "start": f.get("date"),
                      "home": {"name": home.get("name"), "abbr": None, "logo": home.get("logo"),
                               "score": goals.get("home") if status != "upcoming" else None},
                      "away": {"name": away.get("name"), "abbr": None, "logo": away.get("logo"),
                               "score": goals.get("away") if status != "upcoming" else None},
                      "venue": (f.get("venue") or {}).get("name")})
    return games or None


def _af_standings(league: str):
    headers, lid = _af_headers(), LEAGUES[league][1]
    if not headers or not lid:
        return None
    season = _af_season(league, datetime.now(timezone.utc))
    data = http_json("GET", "https://v3.football.api-sports.io/standings", params={"league": lid, "season": season},
                     headers=headers)
    rows = []
    for item in data.get("response") or []:
        tables = (item.get("league") or {}).get("standings") or []
        for table in tables:
            for r in table:
                t, a = r.get("team") or {}, r.get("all") or {}
                rows.append({"group": r.get("group") if len(tables) > 1 else None, "team": t.get("name") or "?",
                             "abbr": None, "logo": t.get("logo"),
                             "stats": {"#": r.get("rank"), "P": a.get("played"), "W": a.get("win"), "D": a.get("draw"),
                                       "L": a.get("lose"), "GD": r.get("goalsDiff"), "PTS": r.get("points")}})
    return rows or None


def _match_team(game: dict, needle: str) -> bool:
    hay = " ".join(str(game[s].get(k) or "") for s in ("home", "away") for k in ("name", "abbr")).lower()
    return needle in hay


def get_sports(league: str, data_type: str = "scores", team: str = "", date: str = "", days: int = 1) -> dict:
    """
    Get live and recent scores, upcoming fixtures, or the league table for a sports league and show them as a card.

    Args:
        league: nfl, nba, mlb, nhl, wnba, ncaafb, ncaamb, epl, la_liga, serie_a, bundesliga, ligue_1, mls, champions_league, europa_league or world_cup
        data_type: scores (live, recent and upcoming games) or standings (the table)
        team: optional team name to filter games, e.g. "Arsenal"
        date: optional start date YYYY-MM-DD for scores (default: the current scoreboard)
        days: with date, how many days to cover (1-14)
    """
    key = _norm_league(league)
    if not key:
        return fail(f"Unknown league '{league}'. Use one of: {', '.join(LEAGUES)}.")
    kind = "standings" if str(data_type).strip().lower().startswith("stand") else "scores"
    name = LEAGUES[key][2]
    date = (date or "").strip()
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return fail("date must look like 2026-11-14.")
    days = max(1, min(to_int(days) or 1, 14))
    needle = (team or "").strip().lower()

    if kind == "standings":
        ck = ("table", key)
        hit = _cache_table.get(ck)
        try:
            provider, rows = (hit if hit else run_chain([("api-football", _af_standings), ("espn", _espn_standings)], key))
        except ProviderError as e:
            return fail(f"Couldn't get the {name} table right now ({e}). Tell the user, or try search_web.")
        _cache_table.put(ck, (provider, rows))
        if needle:
            rows = [r for r in rows if needle in f"{r['team']} {r.get('abbr') or ''}".lower()] or rows
        rows = rows[:MAX_ROWS]
        columns = [c for c in ("#", "P", "W", "D", "L", "PCT", "GB", "GD", "PTS") if any(c in r["stats"] for r in rows)]
        card = {"type": "standings", "title": f"{name} table", "league": key, "columns": columns, "rows": rows,
                "provider": provider}
        top = ", ".join(f"{r['stats'].get('#', i + 1)}. {r['team']}" for i, r in enumerate(rows[:3]))
        return ok(f"Showing the {name} table card (top: {top}). The card has the full table — don't retype it.", card)

    ck = ("scores", key, date, days)
    hit = _cache_scores.get(ck)
    try:
        provider, games = hit if hit else run_chain([("api-football", _af_scores), ("espn", _espn_scores)], key, date, days)
    except ProviderError as e:
        return fail(f"Couldn't get {name} scores right now ({e}). Tell the user, or try search_web.")
    _cache_scores.put(ck, (provider, games))
    if needle:
        games = [g for g in games if _match_team(g, needle)]
    order = {"live": 0, "upcoming": 1, "final": 2, "postponed": 3}
    games = sorted(games, key=lambda g: (order.get(g["status"], 9), g.get("start") or ""))[:MAX_GAMES]
    if not games:
        return ok(f"No {name} games found" + (f" for '{team}'" if team else "") + " in that window. Say so plainly.")
    card = {"type": "scores", "title": clip(f"{name} · {team}" if team else name, 60), "league": key, "games": games,
            "provider": provider, "fetched_at": datetime.now(timezone.utc).isoformat()}
    brief = "; ".join(f"{g['away']['name']} {g['away']['score'] if g['away']['score'] is not None else ''} @ "
                      f"{g['home']['name']} {g['home']['score'] if g['home']['score'] is not None else ''} "
                      f"({g['status_text']})".replace("  ", " ") for g in games[:4])
    return ok(f"Showing {len(games)} {name} game(s) on a card: {brief}. The card has every game — don't retype them all.", card)
