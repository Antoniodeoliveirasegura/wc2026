"""
Decisive diagnostic: does the market-value prior help on CROSS-confederation
matches (Caley's thesis) more than within-confederation ones?

Within-confederation pairs share many opponents -> DC is well-identified -> MV adds
nothing (confirmed in marketvalue.py). Cross-confederation pairs are weakly linked
-> MV should help. The 2026 WC group games (cross-confederation, out-of-sample for a
DC fit < 2023-06) sit in the cross set, so this is the relevant test.

Caveat: DC is stale for the late OOS window, but staleness hits both subsets, so the
within-vs-cross CONTRAST is what we read, not the absolute level.

Run: python mv_connectivity.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import wc_model as wc
import marketvalue as mvmod

DC_CUT = pd.Timestamp("2023-06-01")

def confed_map():
    return mvmod.load_conf()

if __name__ == "__main__":
    df = wc.load()
    mv = mvmod.load_mv()
    conf = confed_map()
    m = wc.fit_dixon_coles(df[df.date < DC_CUT], ref_date=DC_CUT)
    z, _ = mvmod.mv_zscores(m["teams"], mv)

    post = df[df.date >= DC_CUT].copy()
    post = post[post.home_team.isin(m["idx"]) & post.away_team.isin(m["idx"])]
    post = post[~post.tournament.str.contains("friendly", case=False)]
    HT, AT, NEU = post.home_team.values, post.away_team.values, post.neutral.values
    Y = np.array([mvmod.outcome(r.home_score, r.away_score) for r in post.itertuples(index=False)])

    def cf(team):
        return conf.get(mvmod.ALIAS.get(team, team))
    hc = np.array([cf(t) for t in HT], dtype=object)
    ac = np.array([cf(t) for t in AT], dtype=object)
    both = np.array([h is not None and a is not None for h, a in zip(hc, ac)])
    cross = both & (hc != ac)
    within = both & (hc == ac)

    def probs(beta):
        m2 = mvmod.adjusted(m, z, beta)
        return np.array([wc.wdl(m2, HT[k], AT[k], neutral=bool(NEU[k])) for k in range(len(post))])
    betas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    cache = {b: probs(b) for b in betas}

    for label, mask in [("within-confederation", within), ("cross-confederation", cross)]:
        base = mvmod.rps(cache[0.0][mask], Y[mask])
        best = min(betas, key=lambda b: mvmod.rps(cache[b][mask], Y[mask]))
        print(f"\n{label}  (n={mask.sum()}):")
        for b in betas:
            r = mvmod.rps(cache[b][mask], Y[mask])
            print(f"  beta={b:.2f}  RPS={r:.4f}  ({r-base:+.4f})" + ("  <- best" if b == best else ""))
