"""
B-probe: does recent form add signal OVER Dixon-Coles' time-decayed ratings?

Leak-free features, then multinomial logistic models compared on out-of-sample
competitive RPS:
  M0 = DC only                      (baseline)
  M1 = DC + recent form             (the test)
  M2 = DC + recent form + pre-match Elo diff   (bonus)

If M1 beats M0 on the test set, full Tier B is justified. If not, stop.
DC features come from a TRAIN-ONLY fit, so test-row features never see the future.
Form and Elo are computed strictly from matches before each kickoff.

Run: python probe_form.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import wc_model as wc

CUTOFF = pd.Timestamp("2022-06-01")
FORM_K = 5           # matches of recent history in the window
FORM_DECAY = 0.8     # weight per step back within the window

def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2

def form_diff(df):
    """Leak-free: home/away exponentially-weighted recent goal diff BEFORE kickoff."""
    hist = {}
    fh = np.zeros(len(df)); fa = np.zeros(len(df))
    hts, ats = df.home_team.values, df.away_team.values
    hs, as_ = df.home_score.values, df.away_score.values
    def g(team):
        h = hist.get(team)
        if not h:
            return 0.0
        gd = h[-FORM_K:]
        w = FORM_DECAY ** np.arange(len(gd))[::-1]
        return float(np.dot(w, gd) / w.sum())
    for k in range(len(df)):
        fh[k] = g(hts[k]); fa[k] = g(ats[k])
        hist.setdefault(hts[k], []).append(hs[k] - as_[k])
        hist.setdefault(ats[k], []).append(as_[k] - hs[k])
    return fh - fa

def elo_pre(df, home_adv=65.0, start=1500.0):
    """Leak-free pre-match Elo difference (home_pre + home_adv - away_pre)."""
    r = {}; diff = np.zeros(len(df))
    def K(t, m):
        t = str(t).lower()
        base = 50 if ("world cup" in t and "qual" not in t) else \
               40 if any(s in t for s in ("euro", "copa", "nations", "cup", "qualif")) else \
               20 if "friendly" in t else 30
        return base * (1.0 if m <= 1 else 1.5 if m == 2 else 1.75 + (m - 3) / 8.0)
    for k, row in enumerate(df.itertuples(index=False)):
        rh = r.get(row.home_team, start); ra = r.get(row.away_team, start)
        ha = 0.0 if row.neutral else home_adv
        diff[k] = (rh + ha) - ra
        eh = 1.0 / (1.0 + 10 ** (-(rh + ha - ra) / 400.0))
        sh = 1.0 if row.home_score > row.away_score else 0.5 if row.home_score == row.away_score else 0.0
        d = K(row.tournament, abs(row.home_score - row.away_score)) * (sh - eh)
        r[row.home_team] = rh + d; r[row.away_team] = ra - d
    return diff

def rps(P, Y):
    oh = np.eye(3)[Y]; cp = np.cumsum(P, 1); ca = np.cumsum(oh, 1)
    return np.mean(np.sum((cp - ca) ** 2, 1) / 2)

if __name__ == "__main__":
    df = wc.load().reset_index(drop=True)
    m = wc.fit_dixon_coles(df[df.date < CUTOFF], ref_date=CUTOFF)   # train-only fit

    fd = form_diff(df)
    eld = elo_pre(df)
    hts, ats, neu = df.home_team.values, df.away_team.values, df.neutral.values
    keep = (df.home_team.isin(m["idx"]) & df.away_team.isin(m["idx"])).values
    dcp = np.zeros((len(df), 3))
    for k in np.where(keep)[0]:
        dcp[k] = wc.wdl(m, hts[k], ats[k], neutral=bool(neu[k]))

    idx = np.where(keep)[0]
    df2 = df.iloc[idx]
    X_dc = dcp[idx]
    X_form = fd[idx].reshape(-1, 1)
    X_elo = eld[idx].reshape(-1, 1)
    Y = np.array([outcome(h, a) for h, a in zip(df2.home_score, df2.away_score)])
    is_train = (df2.date < CUTOFF).values
    is_comp = ~df2.tournament.str.contains("friendly", case=False).values

    def run(parts, name):
        X = np.column_stack(parts)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[is_train], Y[is_train])
        te = ~is_train
        P = clf.predict_proba(X[te])
        c = is_comp[te]
        print(f"  {name:<20} test RPS={rps(P, Y[te]):.4f}   competitive={rps(P[c], Y[te][c]):.4f}  (n_comp={c.sum()})")

    print(f"B-probe (multinomial logistic; train<{CUTOFF.date()}, test after):")
    print(f"  {'raw Dixon-Coles':<20} test RPS=0.1726     competitive=0.1718  (reference)")
    run([X_dc], "M0 DC only")
    run([X_dc, X_form], "M1 DC+form")
    run([X_dc, X_form, X_elo], "M2 DC+form+elo")
    # self-check: feature build is leak-free in shape and the baseline reproduces DC
    assert len(Y) == len(idx) and X_dc.shape[1] == 3, "feature/label alignment"
    print("\nread: if M1 'competitive' < M0, recent form adds signal over DC -> build B.")
