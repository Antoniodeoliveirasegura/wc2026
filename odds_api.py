"""
The Odds API client for World Cup betting odds.

Key is read from the ODDS_API_KEY env var or a gitignored .env file — NEVER hardcoded or
committed. Results are cached to odds_cache.json with a TTL to conserve the free 500/month
quota (each live call costs quota). Normalizes the API's markets/outcomes into the same
(market, token) keys betting.market_probs() emits, so model probs and book odds join cleanly.

Only the cheap bulk markets (h2h, spreads, totals) are pulled; richer markets need per-event
calls. Best (highest) decimal odds across books are kept — line shopping.
"""
from __future__ import annotations
import os, json, time, urllib.request, urllib.parse

API = "https://api.the-odds-api.com/v4"
SPORT = "soccer_fifa_world_cup"
_HERE = os.path.dirname(__file__)
CACHE = os.path.join(_HERE, "odds_cache.json")
ENV = os.path.join(_HERE, ".env")

LINES_SPREAD = {0.5, 1.5, 2.5}
LINES_TOTAL = {0.5, 1.5, 2.5, 3.5, 4.5}
SHARP_BOOK = "pinnacle"          # the sharp benchmark — its de-vigged line ~= true probability

# The Odds API team name -> our martj42 dataset name (only where they differ).
ALIAS = {
    "Czechia": "Czech Republic", "Korea Republic": "South Korea", "USA": "United States",
    "Türkiye": "Turkey", "Cabo Verde": "Cape Verde", "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo", "Curacao": "Curaçao", "IR Iran": "Iran",
}


def _alias(n: str) -> str:
    return ALIAS.get(n, n)


def _key() -> str:
    k = os.environ.get("ODDS_API_KEY")
    if not k and os.path.exists(ENV):
        for line in open(ENV, encoding="utf-8"):
            if line.strip().startswith("ODDS_API_KEY"):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        raise RuntimeError("ODDS_API_KEY not set. Put it in the environment or a gitignored "
                           ".env file (ODDS_API_KEY=...). Never commit the key.")
    return k


def fetch_raw(markets=("h2h", "spreads", "totals"), regions="us,eu", ttl=1800, force=False) -> list:
    """Fetch (or reuse cached) WC odds. ttl seconds before a refresh; force re-pulls now."""
    if not force and os.path.exists(CACHE):
        c = json.load(open(CACHE, encoding="utf-8"))
        if time.time() - c.get("_ts", 0) < ttl:
            return c["data"]
    q = urllib.parse.urlencode({"apiKey": _key(), "regions": regions,
                                "markets": ",".join(markets), "oddsFormat": "decimal"})
    with urllib.request.urlopen(f"{API}/sports/{SPORT}/odds/?{q}") as r:
        data = json.load(r)
        remaining = r.headers.get("x-requests-remaining")
    json.dump({"_ts": time.time(), "_remaining": remaining, "data": data},
              open(CACHE, "w", encoding="utf-8"))
    return data


def remaining_quota() -> str | None:
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8")).get("_remaining")
    return None


def _outcomes(mk: str, outcomes: list, home: str):
    """Yield (market, token, price) for one bookmaker market, in betting.market_probs() keys."""
    for o in outcomes:
        nm, pt, price = o["name"], o.get("point"), o.get("price")
        if not price:
            continue
        if mk == "h2h":
            tok = "DRAW" if nm == "Draw" else ("HOME" if _alias(nm) == home else "AWAY")
            yield ("moneyline", tok, float(price))
        elif mk == "spreads" and pt is not None and abs(pt) in LINES_SPREAD:
            side = "HOME" if _alias(nm) == home else "AWAY"
            yield (f"spread_{abs(pt)}", f"{side}_{'-' if pt < 0 else '+'}{abs(pt)}", float(price))
        elif mk == "totals" and pt is not None and pt in LINES_TOTAL:
            yield (f"total_{pt}", f"{'OVER' if nm == 'Over' else 'UNDER'}_{pt}", float(price))


def normalize(data: list) -> dict:
    """-> {(home, away): {"commence": iso, "best": {(market,token): best_decimal},
    "sharp": {(market,token): pinnacle_decimal}}}. `best` = best price across all books
    (line shopping, the price you'd bet at); `sharp` = Pinnacle only (the fair-line benchmark)."""
    games: dict = {}
    for e in data:
        home, away = _alias(e["home_team"]), _alias(e["away_team"])
        best: dict = {}
        sharp: dict = {}
        for bk in e.get("bookmakers", []):
            is_sharp = bk.get("key") == SHARP_BOOK
            for mkt in bk.get("markets", []):
                for k0, k1, price in _outcomes(mkt["key"], mkt["outcomes"], home):
                    key = (k0, k1)
                    if key not in best or price > best[key]:
                        best[key] = price
                    if is_sharp:
                        sharp[key] = price
        games[(home, away)] = {"commence": e.get("commence_time"), "best": best, "sharp": sharp}
    return games


if __name__ == "__main__":
    data = fetch_raw()
    g = normalize(data)
    sharp_games = sum(1 for info in g.values() if info["sharp"])
    print(f"events: {len(g)} | with Pinnacle: {sharp_games} | quota remaining: {remaining_quota()}")
    for (h, a), info in list(g.items())[:3]:
        print(f"  {h} vs {a}: best={len(info['best'])} sel, pinnacle={len(info['sharp'])} sel")
