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
import numpy as np
import wc_model as wc
import marketvalue as mvmod
import altitude as altmod
import simulate as sim
import betting as bet
import odds_api


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


def candidates_for(home, away, M, lam, mu, book) -> list[bet.Candidate]:
    cands = []
    for market, sels in bet.market_probs(M, lam, mu).items():
        for token, p in sels.items():
            o = book.get((market, token))
            if not o:
                continue                         # no book price -> can't compute edge -> skip
            label, side = label_side(market, token, home, away)
            cands.append(bet.Candidate(market, token, label, float(p), float(o), side))
    return cands


def game_lambdas(adj, home, away):
    i, j = adj["idx"][home], adj["idx"][away]
    return (float(np.exp(adj["attack"][i] - adj["defense"][j])),
            float(np.exp(adj["attack"][j] - adj["defense"][i])))


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
        rec = bet.recommend(gid, candidates_for(home, away, M, lam, mu, info["odds"]))
        rec.update(home=home, away=away, commence=info.get("commence"))
        out.append(rec)

    out.sort(key=lambda r: (-len(r["recommendedBets"]),
                            -max([b["edge"] for b in r["recommendedBets"]], default=0)))
    payload = {"generatedAt": datetime.datetime.now().isoformat(timespec="minutes"),
               "quotaRemaining": odds_api.remaining_quota(), "games": out}
    with open(os.path.join(os.path.dirname(__file__), "bets.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

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
    if skipped:
        print("\nskipped (team name not in model):", skipped)
