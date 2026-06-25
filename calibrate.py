"""
Does isotonic calibration of the Dixon-Coles W/D/L probabilities help?

Strictly temporal, leak-free:
  DC fit on   < 2023-06   ->   calibrate on 2023-06..2024-06   ->   test on 2024-06+
Per-class isotonic regression (one-vs-rest), renormalized. Reports competitive RPS
raw vs calibrated. If calibrated < raw, wire it into the pipeline.

Run: python calibrate.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import wc_model as wc

DC_CUT = pd.Timestamp("2023-06-01")
CAL_CUT = pd.Timestamp("2024-06-01")

def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2

def rps(P, Y):
    oh = np.eye(3)[Y]; cp = np.cumsum(P, 1); ca = np.cumsum(oh, 1)
    return np.mean(np.sum((cp - ca) ** 2, 1) / 2)

def fit_calibrators(Pc, Yc):
    cals = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(Pc[:, c], (Yc == c).astype(float))
        cals.append(ir)
    return cals

def apply_calibrators(cals, P):
    out = np.column_stack([cals[c].predict(P[:, c]) for c in range(3)])
    out = np.clip(out, 1e-9, None)
    return out / out.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    df = wc.load()
    m = wc.fit_dixon_coles(df[df.date < DC_CUT], ref_date=DC_CUT)

    post = df[df.date >= DC_CUT].copy()
    post = post[post.home_team.isin(m["idx"]) & post.away_team.isin(m["idx"])]
    P = np.array([wc.wdl(m, r.home_team, r.away_team, neutral=bool(r.neutral))
                  for r in post.itertuples(index=False)])
    Y = np.array([outcome(r.home_score, r.away_score) for r in post.itertuples(index=False)])
    comp = ~post.tournament.str.contains("friendly", case=False).values
    is_cal = (post.date < CAL_CUT).values
    te = ~is_cal

    cals = fit_calibrators(P[is_cal], Y[is_cal])
    P_te_cal = apply_calibrators(cals, P[te])

    c = comp[te]
    raw = rps(P[te][c], Y[te][c])
    cal = rps(P_te_cal[c], Y[te][c])
    print(f"test = competitive matches >= {CAL_CUT.date()}  (n={c.sum()})")
    print(f"  raw Dixon-Coles    RPS={raw:.4f}")
    print(f"  + isotonic calib   RPS={cal:.4f}   (delta {cal-raw:+.4f})")
    print("verdict:", "calibration helps -> wire it in" if cal < raw else "no help -> skip")
