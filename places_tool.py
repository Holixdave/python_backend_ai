# places_tool.py — search_places
# ─────────────────────────────────────────────────────────────────────────────
# Provider chain (first that returns results wins):
#   1. Google Places API (New) text search — only if GOOGLE_PLACES_API_KEY (or GOOGLE_API_KEY) is set
#   2. Photon (komoot) on OpenStreetMap data — free, no key, supports a location bias
#   3. Nominatim (OpenStreetMap) — free, no key, 1 request/second policy (throttled here)
# Returns a "places" card the app draws on a map. `attribution` MUST be shown under the map.
# ─────────────────────────────────────────────────────────────────────────────
import math
import os
from typing import Optional

from card_tools import (ProviderError, Throttle, TTLCache, clip, fail, http_json, ok, run_chain, to_float,
                        to_int)

MAX_PLACES = 8
_cache = TTLCache(ttl_s=600)
_nominatim_gate = Throttle(1.05)


def _key() -> str:
    return (os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _km(a_lat, a_lng, b_lat, b_lng) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _osm_link(lat, lng) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map=17/{lat}/{lng}"


def _google_link(lat, lng) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"


# ── providers: each returns a list of place dicts, or None/raises ProviderError ──
def _google(query: str, near: str, lat: Optional[float], lng: Optional[float], limit: int):
    key = _key()
    if not key:
        return None
    text = f"{query} near {near}" if near and lat is None else query
    body = {"textQuery": text, "maxResultCount": limit}
    if lat is not None and lng is not None:
        body["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": 5000.0}}
    mask = ("places.id,places.displayName,places.formattedAddress,places.location,places.rating,"
            "places.userRatingCount,places.googleMapsUri,places.currentOpeningHours.openNow,"
            "places.nationalPhoneNumber,places.primaryTypeDisplayName")
    data = http_json("POST", "https://places.googleapis.com/v1/places:searchText", json_body=body,
                     headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": mask})
    out = []
    for p in data.get("places") or []:
        loc = p.get("location") or {}
        la, lo = to_float(loc.get("latitude")), to_float(loc.get("longitude"))
        if la is None or lo is None:
            continue
        out.append({
            "name": (p.get("displayName") or {}).get("text") or "Place", "address": p.get("formattedAddress") or "",
            "lat": la, "lng": lo, "rating": to_float(p.get("rating")), "rating_count": to_int(p.get("userRatingCount")),
            "open_now": (p.get("currentOpeningHours") or {}).get("openNow"), "phone": p.get("nationalPhoneNumber"),
            "category": (p.get("primaryTypeDisplayName") or {}).get("text"),
            "maps_url": p.get("googleMapsUri") or _google_link(la, lo), "place_id": p.get("id"),
        })
    return out or None


def _addr_from_props(p: dict) -> str:
    parts = [p.get("housenumber"), p.get("street"), p.get("district") or p.get("suburb"), p.get("city") or p.get("town"),
             p.get("state"), p.get("country")]
    return ", ".join(str(x) for x in parts if x)


def _photon(query: str, near: str, lat: Optional[float], lng: Optional[float], limit: int):
    params = {"q": query if not near or lat is not None else f"{query} {near}", "limit": limit}
    if lat is not None and lng is not None:
        params.update({"lat": lat, "lon": lng})
    data = http_json("GET", "https://photon.komoot.io/api/", params=params)
    out = []
    for f in data.get("features") or []:
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lo, la = to_float(coords[0]), to_float(coords[1])
        props = f.get("properties") or {}
        if la is None or lo is None or not props.get("name"):
            continue
        out.append({"name": props["name"], "address": _addr_from_props(props), "lat": la, "lng": lo, "rating": None,
                    "rating_count": None, "open_now": None, "phone": None,
                    "category": props.get("osm_value"), "maps_url": _osm_link(la, lo),
                    "place_id": f"osm:{props.get('osm_type', '')}{props.get('osm_id', '')}"})
    return out or None


def _nominatim(query: str, near: str, lat: Optional[float], lng: Optional[float], limit: int):
    _nominatim_gate.wait()
    params = {"q": query if not near else f"{query} {near}", "format": "jsonv2", "limit": limit, "addressdetails": 0}
    if lat is not None and lng is not None:                 # prefer (not force) results around the point
        d = 0.1
        params.update({"viewbox": f"{lng - d},{lat + d},{lng + d},{lat - d}", "bounded": 0})
    data = http_json("GET", "https://nominatim.openstreetmap.org/search", params=params)
    out = []
    for r in data if isinstance(data, list) else []:
        la, lo = to_float(r.get("lat")), to_float(r.get("lon"))
        if la is None or lo is None:
            continue
        out.append({"name": r.get("name") or (r.get("display_name") or "Place").split(",")[0],
                    "address": r.get("display_name") or "", "lat": la, "lng": lo, "rating": None, "rating_count": None,
                    "open_now": None, "phone": None, "category": r.get("type"),
                    "maps_url": _osm_link(la, lo), "place_id": f"osm:{r.get('osm_type', '')}{r.get('osm_id', '')}"})
    return out or None


_PROVIDERS = [("google", _google), ("photon", _photon), ("nominatim", _nominatim)]
_ATTRIBUTION = {"google": "Map data and place details © Google", "photon": "© OpenStreetMap contributors",
                "nominatim": "© OpenStreetMap contributors"}


def _geocode_near(near: str):
    """'Yaba, Lagos' -> (lat, lng) using the free chain (cached). None if it can't be found."""
    key = ("geo", near.lower().strip())
    hit = _cache.get(key)
    if hit is not None:
        return hit or None
    for fn in (_photon, _nominatim):
        try:
            res = fn(near, "", None, None, 1)
        except ProviderError:
            continue
        if res:
            return _cache.put(key, (res[0]["lat"], res[0]["lng"]))
    _cache.put(key, ())
    return None


def search_places(query: str, near: str = "", latitude: Optional[float] = None, longitude: Optional[float] = None,
                  limit: int = 6) -> dict:
    """
    Find real places (restaurants, pharmacies, schools, ATMs, landmarks...) and show them on a map card.

    Args:
        query: what to look for, e.g. "pharmacy" or "jollof restaurants"
        near: a place name or area to search around, e.g. "Yaba, Lagos" (use this when the user names an area)
        latitude: the user's latitude, only if you really have it
        longitude: the user's longitude, only if you really have it
        limit: how many places (1-8)
    """
    query = (query or "").strip()
    if not query:
        return fail("query is empty — say what kind of place to look for.")
    limit = max(1, min(to_int(limit) or 6, MAX_PLACES))
    lat, lng = to_float(latitude), to_float(longitude)
    near = (near or "").strip()
    if (lat is None or lng is None) and near:
        pt = _geocode_near(near)
        lat, lng = pt if pt else (None, None)
    if lat is None or lng is None:
        lat = lng = None

    ck = ("search", query.lower(), near.lower(), None if lat is None else round(lat, 3),
          None if lng is None else round(lng, 3), limit)
    hit = _cache.get(ck)
    if hit:
        return hit
    try:
        provider, places = run_chain(_PROVIDERS, query, near, lat, lng, limit)
    except ProviderError as e:
        return fail(f"Couldn't search places right now ({e}). Tell the user, or try search_web.")

    for p in places:
        if lat is not None:
            p["distance_km"] = round(_km(lat, lng, p["lat"], p["lng"]), 2)
    if lat is not None:
        places.sort(key=lambda p: p["distance_km"])
    places = places[:limit]
    for p in places:
        p["name"], p["address"] = clip(p["name"], 100), clip(p["address"], 200)
    center = {"lat": lat, "lng": lng} if lat is not None else {"lat": places[0]["lat"], "lng": places[0]["lng"]}
    card = {"type": "places", "title": clip(query if not near else f"{query} · {near}", 80), "center": center,
            "places": places, "provider": provider, "attribution": _ATTRIBUTION[provider]}
    names = "; ".join(f"{p['name']}" + (f" ({p['distance_km']} km)" if "distance_km" in p else "") for p in places[:5])
    result = ok(f"Showing {len(places)} place(s) on a map card: {names}. The card has addresses and links — "
                f"don't re-list them all; give 1-2 picks with a reason.", card)
    return _cache.put(ck, result)
