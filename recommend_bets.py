"""
Value-bet recommendations: model market probabilities vs live book odds -> bets.json.

For each upcoming WC game we build the Dixon-Coles score matrix (squad-value + altitude
adjusted), derive every market's model probability, join to the book's best odds, then run
betting.recommend() (hard filter + de-correlate + cap). Output is the small per-game set
the betting spec asks for.

Run: ODDS_API_KEY=... python recommend_bets.py     (or put the key in a gitignored .env)
"""
from __future__ import annotations
import os, json, datetime
from collections import defaultdict
import numpy as np
import wc_model as wc
import marketvalue as mvmod
import altitude as altmod
import simulate as sim
import betting as bet
import odds_api
import clv


def label_side(market: str, token: str, home: str, away: str) -> tuple[str, str]:
    """Human label + correlation side (the team it leans, or OVER/UNDER, or '')."""
    if market == "moneyline":
        return {"HOME": (f"{home} win", home), "AWAY": (f"{away} win", away),
                "DRAW": ("Draw", "")}[token]
    if market == "double_chance":
        return {"HOME_DRAW": (f"{home} or draw", home), "DRAW_AWAY": (f"{away} or draw", away),
                "HOME_AWAY": ("Either team to win", "")}[token]
    if market.startswith("spread_"):
        team = home if token.startswith("HOME") else away
        line = token.split("_")[1]
        return (f"{team} {line}", team)
    if market.startswith("total_"):
        line = market.split("_")[1]
        over = token.startswith("OVER")
        return (f"{'Over' if over else 'Under'} {line} goals", "OVER" if over else "UNDER")
    if market.startswith("team_total_"):
        line = market.split("_")[2]
        team = home if token.startswith("HOME") else away
        over = "OVER" in token
        return (f"{team} {'over' if over else 'under'} {line} goals", team)
    if market == "btts":
        yes = token == "YES"
        return ("Both teams to score" + ("" if yes else " — No"), "OVER" if yes else "UNDER")
    if market == "fh_result":
        return {"HOME": (f"{home} ahead at half", home), "AWAY": (f"{away} ahead at half", away),
                "DRAW": ("Level at half", "")}[token]
    if market.startswith("fh_total_"):
        line = market.split("_")[2]
        over = token.startswith("OVER")
        return (f"1st half {'over' if over else 'under'} {line} goals", "OVER" if over else "UNDER")
    if market == "first_to_score":
        return {"HOME": (f"{home} to score first", home), "AWAY": (f"{away} to score first", away),
                "NONE": ("No goal scored", "UNDER")}[token]
    return (f"{market} {token}", "")


def _devig_group(market: str, token: str) -> str | None:
    """Which outcomes share a market for de-vigging. None = ungrouped (no de-vig)."""
    if market == "moneyline":
        return "ml"                                       # HOME/DRAW/AWAY (3-way)
    if market.startswith("total_") or market == "btts":
        return market                                     # OVER/UNDER or YES/NO (2-way)
    if market.startswith("spread_"):
        side, val = token.split("_")
        v = float(val)
        return f"sp_{v if side == 'HOME' else -v}"        # 2 sides of one handicap line
    return None


def devig_book(book: dict) -> dict:
    """{(market,token): odds} -> {(market,token): de-vigged fair probability} (power method)."""
    groups: dict[str, list] = defaultdict(list)
    for mt in book:
        g = _devig_group(*mt)
        if g:
            groups[g].append(mt)
    fair = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        dv = bet.devig_power([book[mt] for mt in members])
        if dv:
            for mt, p in zip(members, dv):
                fair[mt] = p
    return fair


def candidates_for(home, away, M, lam, mu, best, sharp) -> list[bet.Candidate]:
    sharp_fair = devig_book(sharp)                        # Pinnacle de-vigged = the benchmark
    best_fair = devig_book(best)                          # fallback where Pinnacle didn't price it
    cands = []
    for market, sels in bet.market_probs(M, lam, mu).items():
        for token, p in sels.items():
            o = best.get((market, token))
            if not o:
                continue                         # no book price -> can't compute edge -> skip
            key = (market, token)
            label, side = label_side(market, token, home, away)
            cands.append(bet.Candidate(market, token, label, float(p), float(o), side,
                                       fair=sharp_fair.get(key) or best_fair.get(key),
                                       sharp_odds=sharp.get(key),
                                       pin_fair=sharp_fair.get(key)))     # Pinnacle-only -> market value
    return cands


def game_lambdas(adj, home, away):
    i, j = adj["idx"][home], adj["idx"][away]
    return (float(np.exp(adj["attack"][i] - adj["defense"][j])),
            float(np.exp(adj["attack"][j] - adj["defense"][i])))


ANCHOR_W = 0.5      # geometric weight on the Pinnacle de-vigged line in the anchored forecast.

def anchored_forecast(M, lam, mu, sharp_fair, w=ANCHOR_W):
    """Second, experimental forecast: the model moneyline geometrically blended with
    Pinnacle's de-vigged line (the sharpest market estimate of true probability). Published
    alongside the pure-model forecast and judged live by CLV — research says the closing
    line is near-unbeatable, so anchoring to it is calibration insurance, not edge.
    Returns {model, market, anchored} as [H,D,A]; market/anchored fall back to model when
    Pinnacle didn't price the 3-way."""
    ml = bet.market_probs(M, lam, mu)["moneyline"]
    pm = [ml["HOME"], ml["DRAW"], ml["AWAY"]]
    keys = [("moneyline", "HOME"), ("moneyline", "DRAW"), ("moneyline", "AWAY")]
    if not all(k in sharp_fair for k in keys):
        return {"model": pm, "market": None, "anchored": pm}
    pk = [sharp_fair[k] for k in keys]
    g = [p ** (1 - w) * q ** w for p, q in zip(pm, pk)]
    s = sum(g)
    return {"model": pm, "market": pk, "anchored": [x / s for x in g]}


if __name__ == "__main__":
    df = wc.load()
    m = sim.get_model(df)
    zmap, confmap = mvmod.setup(m)
    fcity = sim.fixture_cities(df)
    games_odds = odds_api.normalize(odds_api.fetch_raw())

    out, skipped = [], []
    for (home, away), info in games_odds.items():
        if home not in m["idx"] or away not in m["idx"]:
            skipped.append((home, away)); continue
        city = fcity.get(frozenset((home, away)), "")
        adj = altmod.alt_adjust(mvmod.mv_adjust(m, zmap, confmap, home, away), home, away, city)
        M = wc.score_matrix(adj, home, away, neutral=True, maxg=sim.MAXG)
        lam, mu = game_lambdas(adj, home, away)
        gid = f"{home}-{away}".lower().replace(" ", "-")
        rec = bet.recommend(gid, candidates_for(home, away, M, lam, mu,
                                                info["best"], info["sharp"]))
        rec.update(home=home, away=away, commence=info.get("commence"),
                   advance=bet.advance_probs(M, lam, mu),    # knockout: who wins the tie (ET+pens)
                   forecast=anchored_forecast(M, lam, mu, devig_book(info["sharp"])))
        out.append(rec)

    out.sort(key=lambda r: (-len(r["recommendedBets"]),
                            -max([b["edge"] for b in r["recommendedBets"]], default=0)))
    payload = {"generatedAt": datetime.datetime.now().isoformat(timespec="minutes"),
               "quotaRemaining": odds_api.remaining_quota(), "games": out}
    with open(os.path.join(os.path.dirname(__file__), "bets.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    clv.update(out)                                        # log entry prices / freeze closings
    cs = clv.summary()

    n_bets = sum(len(r["recommendedBets"]) for r in out)
    print(f"games priced: {len(out)} | skipped (name mismatch): {len(skipped)} "
          f"| quota: {odds_api.remaining_quota()}")
    print(f"total recommended bets: {n_bets}\n")
    for r in out:
        if r["recommendedBets"]:
            print(f"{r['home']} vs {r['away']}:")
            for b in r["recommendedBets"]:
                print(f"  BET  {b['selection']:<26} edge={b['edge']:+.0%} {b['confidence']}  "
                      f"(model {b['modelProbability']:.0%} vs mkt {b['sportsbookImpliedProbability']:.0%})")
    print(f"\nCLV log: {cs['logged']} picks tracked, {cs['settled']} settled"
          + (f", avg CLV {cs['avg_clv']:+.2%}" if cs['settled'] else " (CLV populates as games kick off)"))
    if skipped:
        print("skipped (team name not in model):", skipped)
