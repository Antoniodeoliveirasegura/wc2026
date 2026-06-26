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


def fetch_raw(markets=("h2h", "spreads", "totals"), regions="us", ttl=1800, force=False) -> list:
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


def normalize(data: list) -> dict:
    """-> {(home, away): {"commence": iso, "odds": {(market, token): best_decimal}}}.
    Tokens match betting.market_probs(): moneyline HOME/DRAW/AWAY, spread_X HOME_-X/AWAY_+X,
    total_X OVER_X/UNDER_X. Best price across books is kept."""
    games: dict = {}
    for e in data:
        home, away = _alias(e["home_team"]), _alias(e["away_team"])
        book: dict = {}

        def better(k, price):
            if price and (k not in book or price > book[k]):
                book[k] = float(price)

        for bk in e.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                mk = mkt["key"]
                for o in mkt["outcomes"]:
                    nm, pt, price = o["name"], o.get("point"), o.get("price")
                    if mk == "h2h":
                        tok = "DRAW" if nm == "Draw" else ("HOME" if _alias(nm) == home else "AWAY")
                        better(("moneyline", tok), price)
                    elif mk == "spreads" and pt is not None and abs(pt) in LINES_SPREAD:
                        side = "HOME" if _alias(nm) == home else "AWAY"
                        tok = f"{side}_{'-' if pt < 0 else '+'}{abs(pt)}"
                        better((f"spread_{abs(pt)}", tok), price)
                    elif mk == "totals" and pt is not None and pt in LINES_TOTAL:
                        tok = f"{'OVER' if nm == 'Over' else 'UNDER'}_{pt}"
                        better((f"total_{pt}", tok), price)
        games[(home, away)] = {"commence": e.get("commence_time"), "odds": book}
    return games


if __name__ == "__main__":
    data = fetch_raw()
    g = normalize(data)
    print(f"events: {len(g)} | quota remaining: {remaining_quota()}")
    for (h, a), info in list(g.items())[:3]:
        print(f"  {h} vs {a}: {len(info['odds'])} priced selections")
        for k, v in list(info["odds"].items())[:6]:
            print(f"     {k} = {v}")
