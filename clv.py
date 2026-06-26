"""
Closing Line Value (CLV) tracking — the only way to validate a betting model when you have
no historical odds (research consensus: CLV is the gold-standard validator, stabilises in
200-500 bets, works without settled results). The closing line is the market's sharpest
estimate; if our entry price consistently beats it, the model has real edge.

CLV% = entry_odds / closing_odds - 1   (positive = we got a better price than the market's close)

Mechanism (driven by repeated recommend_bets.py runs over the tournament):
  * first time a pick appears -> log its entry odds + timestamp
  * each later run before kickoff -> refresh the provisional "last seen" odds
  * once the game has kicked off -> freeze that last price as the closing line, compute CLV

Run: python clv.py    # print the running CLV summary
"""
from __future__ import annotations
import os, json, datetime

LOG = os.path.join(os.path.dirname(__file__), "clv_log.json")


def _load() -> dict:
    return json.load(open(LOG, encoding="utf-8")) if os.path.exists(LOG) else {}


def _save(d: dict) -> None:
    json.dump(d, open(LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _parse(iso: str | None):
    if not iso:
        return None
    try:
        return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def update(games: list[dict], now: datetime.datetime | None = None) -> dict:
    """Log entry prices for new picks and freeze closing prices for games that have kicked off.
    `games` are recommend_bets.py game dicts (gameId/home/away/commence + topBets with odds)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    log = _load()
    for g in games:
        ko = _parse(g.get("commence"))
        for b in g.get("topBets", []):
            if b.get("recommendation") == "avoid" or not b.get("odds"):
                continue
            key = f'{g["gameId"]}|{b["market"]}|{b["selection"]}'
            rec = log.get(key)
            if rec is None:
                log[key] = {"game": f'{g["home"]} v {g["away"]}', "market": b["market"],
                            "selection": b["selection"], "entry_odds": b["odds"],
                            "entry_time": now.isoformat(timespec="minutes"),
                            "commence": g.get("commence"), "model_p": b["modelProbability"],
                            "last_odds": b["odds"], "closing_odds": None, "clv": None}
            elif rec["closing_odds"] is None:
                if ko and now >= ko:                       # kicked off -> freeze the close
                    close = rec.get("last_odds") or b["odds"]
                    rec["closing_odds"] = close
                    rec["clv"] = round(rec["entry_odds"] / close - 1.0, 4)
                else:                                      # still open -> track latest price
                    rec["last_odds"] = b["odds"]
    _save(log)
    return log


def summary() -> dict:
    log = _load()
    closed = [r for r in log.values() if r.get("clv") is not None]
    by_market: dict[str, list[float]] = {}
    for r in closed:
        by_market.setdefault(r["market"], []).append(r["clv"])
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"logged": len(log), "settled": len(closed),
            "avg_clv": avg([r["clv"] for r in closed]),
            "by_market": {m: {"n": len(v), "avg_clv": avg(v)} for m, v in by_market.items()}}


if __name__ == "__main__":
    s = summary()
    print(f"picks logged: {s['logged']} | settled (kicked off): {s['settled']}")
    if s["settled"]:
        print(f"average CLV: {s['avg_clv']:+.2%}   "
              f"({'positive = real edge' if s['avg_clv'] > 0 else 'negative = behind the market'})")
        for m, st in sorted(s["by_market"].items(), key=lambda kv: -kv[1]["avg_clv"]):
            print(f"  {m:<16} n={st['n']:<3} avg CLV {st['avg_clv']:+.2%}")
    else:
        print("No games have kicked off since logging began — CLV will populate as games close.")
        print("Re-run recommend_bets.py over the next days/hours to accumulate closings.")
