"""
Honest out-of-sample test on the 2026 World Cup games ALREADY PLAYED.

The live model is fit on all data (incl. these games), so grading it on them would
leak. Instead: fit ONLY on matches before the tournament (< 2026-06-11), then predict
the played WC games the model never saw. Reports W/D/L RPS / log-loss / accuracy and
the exact-scoreline hit rate (the metric behind the score-prediction feature).

Run: python test_wc2026.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import wc_model as wc
import marketvalue as mvmod

WC_START = pd.Timestamp("2026-06-11")

def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2

def rps(P, Y):
    oh = np.eye(3)[Y]; cp = np.cumsum(P, 1); ca = np.cumsum(oh, 1)
    return np.mean(np.sum((cp - ca) ** 2, 1) / 2)

def logloss(P, Y):
    return -np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)))

if __name__ == "__main__":
    df = wc.load()
    train = df[df.date < WC_START]
    test = df[(df.date >= WC_START) & (df.tournament == "FIFA World Cup")]
    print(f"train: {len(train):,} matches (< {WC_START.date()})  |  test: {len(test)} played WC games")
    m = wc.fit_dixon_coles(train, ref_date=WC_START)            # never sees the WC games
    zmap, confmap = mvmod.setup(m)

    rows = [r for r in test.itertuples(index=False)
            if r.home_team in m["idx"] and r.away_team in m["idx"]]
    Y = np.array([outcome(r.home_score, r.away_score) for r in rows])

    def evaluate(use_mv):
        P, exact = [], 0
        for r in rows:
            mm = mvmod.mv_adjust(m, zmap, confmap, r.home_team, r.away_team) if use_mv else m
            P.append(list(wc.wdl(mm, r.home_team, r.away_team, neutral=bool(r.neutral))))
            M = wc.score_matrix(mm, r.home_team, r.away_team, neutral=bool(r.neutral), maxg=8)
            pi, pj = np.unravel_index(int(np.argmax(M)), M.shape)
            exact += (pi == r.home_score and pj == r.away_score)
        P = np.array(P)
        return rps(P, Y), logloss(P, Y), np.mean(np.argmax(P, 1) == Y), exact

    base = np.bincount(Y, minlength=3) / len(Y)
    Pb = np.tile(base, (len(Y), 1))

    print(f"\nout-of-sample on the {len(Y)} played 2026 World Cup games:")
    for label, mv in [("Dixon-Coles", False), ("DC + market value", True)]:
        r_, ll, acc, ex = evaluate(mv)
        print(f"  {label:<18} RPS={r_:.4f}  logloss={ll:.4f}  acc={acc:.1%}  exact={ex}/{len(Y)} ({ex/len(Y):.0%})")
    print(f"  {'base rate':<18} RPS={rps(Pb, Y):.4f}  logloss={logloss(Pb, Y):.4f}")
    print("  reference: bookmaker-level RPS on competitive matches ~0.18-0.20")
