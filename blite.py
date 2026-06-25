"""
Tier B-lite: Dixon-Coles + cheap leak-free features -> gradient-boosted W/D/L.

Every feature derives from the results data we already have (no scraping):
  DC match probs (train-only fit), recent form, pre-match Elo diff,
  rest-day diff, tournament stage, neutral flag.

FIFA ranking deliberately omitted: it's Elo-derived (redundant with our Elo
feature, which is current) and the free source ends 2024-09 (stale for 2026).

__main__ runs expanding-window time-CV (DC refit per fold) vs the DC baseline,
then trains a final model on all data and saves blite.pkl for the simulator.

Run: python blite.py
"""
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import wc_model as wc

FORM_K, FORM_DECAY = 5, 0.8
REST_CAP = 45.0
ART = os.path.join(os.path.dirname(__file__), "blite.pkl")

def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2

def _ewgd(hist):
    if not hist:
        return 0.0
    gd = hist[-FORM_K:]
    w = FORM_DECAY ** np.arange(len(gd))[::-1]
    return float(np.dot(w, gd) / w.sum())

def _form(df):
    """Leak-free EW recent goal diff; returns (home-away) per match + current map."""
    hist = {}; fh = np.zeros(len(df)); fa = np.zeros(len(df))
    ht, at = df.home_team.values, df.away_team.values
    hs, as_ = df.home_score.values, df.away_score.values
    for k in range(len(df)):
        fh[k] = _ewgd(hist.get(ht[k], [])); fa[k] = _ewgd(hist.get(at[k], []))
        hist.setdefault(ht[k], []).append(int(hs[k] - as_[k]))
        hist.setdefault(at[k], []).append(int(as_[k] - hs[k]))
    return fh - fa, {t: _ewgd(v) for t, v in hist.items()}

def _elo(df, home_adv=65.0, start=1500.0):
    """Leak-free pre-match Elo diff per match + current rating map."""
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
    return diff, r

def _rest(df):
    """Leak-free days-since-last-match diff (capped)."""
    last = {}; rh = np.full(len(df), REST_CAP); ra = np.full(len(df), REST_CAP)
    ht, at, dt = df.home_team.values, df.away_team.values, df.date.values
    for k in range(len(df)):
        if ht[k] in last: rh[k] = min(REST_CAP, (dt[k] - last[ht[k]]) / np.timedelta64(1, "D"))
        if at[k] in last: ra[k] = min(REST_CAP, (dt[k] - last[at[k]]) / np.timedelta64(1, "D"))
        last[ht[k]] = dt[k]; last[at[k]] = dt[k]
    return rh - ra

def _stage(df):
    t = df.tournament.astype(str).str.lower()
    return np.where(t.str.contains("world cup") & ~t.str.contains("qual"), 3.0,
           np.where(t.str.contains("euro|copa|nations|africa|asian|gold cup|qualif", regex=True), 2.0,
           np.where(t.str.contains("friendly"), 0.0, 1.0)))

def global_feats(df):
    fdiff, form_now = _form(df)
    ediff, elo_now = _elo(df)
    feats = dict(form=fdiff, elo=ediff, rest=_rest(df), stage=_stage(df),
                 neutral=df.neutral.values.astype(float))
    return feats, form_now, elo_now

def _dcp(df, m):
    ht, at, neu = df.home_team.values, df.away_team.values, df.neutral.values
    keep = (df.home_team.isin(m["idx"]) & df.away_team.isin(m["idx"])).values
    P = np.zeros((len(df), 3))
    for k in np.where(keep)[0]:
        P[k] = wc.wdl(m, ht[k], at[k], neutral=bool(neu[k]))
    return P, keep

def _X(dcp, gf):
    return np.column_stack([dcp, gf["form"], gf["elo"], gf["rest"], gf["stage"], gf["neutral"]])

def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)

def rps(P, Y):
    oh = np.eye(3)[Y]; cp = np.cumsum(P, 1); ca = np.cumsum(oh, 1)
    return np.mean(np.sum((cp - ca) ** 2, 1) / 2)

def predict_wdl(art, home, away, neutral=True, rest_diff=0.0, stage=3.0):
    """W/D/L for any matchup using saved artifacts (for the simulator)."""
    dcp = np.array(wc.wdl(art["dc"], home, away, neutral=neutral))
    form = art["form_now"].get(home, 0.0) - art["form_now"].get(away, 0.0)
    elo = (art["elo_now"].get(home, 1500.0) - art["elo_now"].get(away, 1500.0)
           + (0.0 if neutral else 65.0))
    X = np.array([[dcp[0], dcp[1], dcp[2], form, elo, rest_diff, stage, 0.0 if neutral else 1.0]])
    return art["gbm"].predict_proba(X)[0]

if __name__ == "__main__":
    df = wc.load().reset_index(drop=True)
    gf, form_now, elo_now = global_feats(df)
    Y = np.array([outcome(h, a) for h, a in zip(df.home_score, df.away_score)])
    comp = ~df.tournament.str.contains("friendly", case=False).values

    cuts = [pd.Timestamp(c) for c in ["2022-06-01", "2023-06-01", "2024-06-01"]]
    ends = cuts[1:] + [df.date.max() + pd.Timedelta(days=1)]
    print("expanding-window time-CV (competitive RPS; DC refit per fold):")
    dc_r, gb_r = [], []
    for cut, end in zip(cuts, ends):
        m = wc.fit_dixon_coles(df[df.date < cut], ref_date=cut)
        P, keep = _dcp(df, m)
        X = _X(P, gf)
        tr = (df.date < cut).values & keep
        te = (df.date >= cut).values & (df.date < end).values & keep
        clf = _gbm().fit(X[tr], Y[tr])
        Pg = clf.predict_proba(X[te])
        c = comp[te]
        d_, g_ = rps(P[te][c], Y[te][c]), rps(Pg[c], Y[te][c])
        dc_r.append(d_); gb_r.append(g_)
        print(f"  test {cut.date()}..{end.date()}  DC={d_:.4f}  GBM={g_:.4f}  (n_comp={c.sum()})")
    print(f"  MEAN  DC={np.mean(dc_r):.4f}   GBM={np.mean(gb_r):.4f}   "
          f"(delta {np.mean(gb_r)-np.mean(dc_r):+.4f})")

    # final model on all data -> save for the simulator
    m_full = wc.fit_dixon_coles(df, ref_date=df.date.max())
    Pf, keepf = _dcp(df, m_full)
    clf = _gbm().fit(_X(Pf, gf)[keepf], Y[keepf])
    pickle.dump(dict(gbm=clf, dc=m_full, form_now=form_now, elo_now=elo_now), open(ART, "wb"))
    assert os.path.exists(ART)
    print(f"\nsaved {os.path.basename(ART)} (GBM + DC + form/Elo maps) for the simulator")
