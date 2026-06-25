"""
Does xG beat actual goals as a strength signal for international matches?

International xG only exists for recent major tournaments (StatsBomb open data:
WC 2018/2022, Euros, Copa, AFCON) — there is NO qualifier/friendly/historical xG,
so an xG rating can't be built for the full 49k-match model. But we CAN test the
premise on the World Cups themselves.

Test: within each WC, for every match, build each team's leave-one-out "form" from
their OTHER matches in that tournament — once from goal difference, once from xG
difference. Whichever better predicts the held-out match outcome (lower log-loss)
is the better strength signal.

Run: python xg_test.py
"""
from __future__ import annotations
import os, json, urllib.request, collections
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPS = [(43, 3, "WC2018"), (43, 106, "WC2022")]
CACHE = os.path.join(os.path.dirname(__file__), "xg_matches.csv")

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)

def build():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    rows = []
    for comp, season, name in COMPS:
        matches = fetch_json(f"{BASE}/matches/{comp}/{season}.json")
        for mt in matches:
            mid = mt["match_id"]
            ht = mt["home_team"]["home_team_name"]; at = mt["away_team"]["away_team_name"]
            try:
                ev = fetch_json(f"{BASE}/events/{mid}.json")
            except Exception:
                continue
            hxg = axg = 0.0
            for e in ev:
                if e.get("type", {}).get("name") == "Shot":
                    xg = e.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
                    tm = e.get("team", {}).get("name")
                    if tm == ht: hxg += xg
                    elif tm == at: axg += xg
            rows.append(dict(tourn=name, home=ht, away=at,
                             hs=mt["home_score"], as_=mt["away_score"],
                             hxg=round(hxg, 3), axg=round(axg, 3)))
        print(f"  {name}: {sum(r['tourn']==name for r in rows)} matches")
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df

def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2

def loo_form(df, for_col, against_col):
    """Leave-one-out per-team mean (for - against) within tournament, per match,
    returned as home_form - away_form."""
    rec = collections.defaultdict(list)   # (tourn, team) -> list of (row_idx, diff)
    for k, r in df.iterrows():
        rec[(r.tourn, r.home)].append((k, r[for_col[0]] - r[against_col[0]]))
        rec[(r.tourn, r.away)].append((k, r[for_col[1]] - r[against_col[1]]))
    def mean_excl(key, k):
        vals = [d for (kk, d) in rec[key] if kk != k]
        return float(np.mean(vals)) if vals else 0.0
    feat = np.zeros(len(df))
    for k, r in df.iterrows():
        feat[k] = mean_excl((r.tourn, r.home), k) - mean_excl((r.tourn, r.away), k)
    return feat.reshape(-1, 1)

def logloss(P, Y):
    return -np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)))

if __name__ == "__main__":
    df = build().reset_index(drop=True)
    Y = np.array([outcome(r.hs, r.as_) for r in df.itertuples(index=False)])
    print(f"\ncorpus: {len(df)} World Cup matches with xG")

    Xg = loo_form(df, ("hs", "as_"), ("as_", "hs"))           # goal-difference form
    Xx = loo_form(df, ("hxg", "axg"), ("axg", "hxg"))         # xG-difference form

    for name, X in [("goals-form", Xg), ("xG-form", Xx)]:
        clf = LogisticRegression(max_iter=2000).fit(X, Y)
        P = clf.predict_proba(X)
        print(f"  {name:<11} log-loss={logloss(P, Y):.4f}  acc={np.mean(np.argmax(P,1)==Y):.3f}")
    print("\nlower log-loss = better strength signal.")
