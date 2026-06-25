"""
Does a Transfermarkt squad-value prior improve our Dixon-Coles model?

Externally this is the single biggest documented upgrade in comparable WC models
(amirdaraee/world-cup-predictions: held-out log-loss 0.854 -> 0.818). Here we test
it on OUR model, leak-free in time:

  DC fit < 2023-06  ->  tune beta on 2023-06..2024-06  ->  test on 2024-06+

Adjustment: attack[i] += beta*z, defense[i] += beta*z, where z = zscore(log squad
market value). Caveat (as in the source project): market values are CURRENT, so
grading past matches has a mild, documented look-ahead.

Data: dcaribou/transfermarkt-datasets national_teams table (squad value + confed).
Run: python marketvalue.py
"""
from __future__ import annotations
import os, urllib.request
import numpy as np
import pandas as pd
import wc_model as wc

MV_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/national_teams.csv.gz"
MV_FILE = os.path.join(os.path.dirname(__file__), "national_teams.csv.gz")
DC_CUT = pd.Timestamp("2023-06-01")
CAL_CUT = pd.Timestamp("2024-06-01")
ALIAS = {"South Korea": "Korea, South", "Bosnia and Herzegovina": "Bosnia-Herzegovina"}

def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2

def rps(P, Y):
    oh = np.eye(3)[Y]; cp = np.cumsum(P, 1); ca = np.cumsum(oh, 1)
    return np.mean(np.sum((cp - ca) ** 2, 1) / 2)

def logloss(P, Y):
    return -np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)))

# Public approximations filling gaps in the source (EUR). The 4 priciest squads
# (ES/FR/EN/PT) are NULL in the dataset; the rest are teams absent from its 124 rows.
MV_FILL = {
    "Spain": 1.40e9, "France": 1.30e9, "England": 1.40e9, "Portugal": 1.05e9,
    "Turkey": 4.0e8, "Ivory Coast": 2.8e8, "DR Congo": 1.7e8,
    "Cape Verde": 5.5e7, "Haiti": 3.5e7, "Curaçao": 2.5e7,
}

def load_mv():
    if not os.path.exists(MV_FILE):
        urllib.request.urlretrieve(MV_URL, MV_FILE)
    t = pd.read_csv(MV_FILE, compression="gzip")
    out = {str(r.country_name): float(r.total_market_value)
           for r in t.itertuples(index=False) if pd.notna(r.total_market_value)}
    for k, v in MV_FILL.items():
        out.setdefault(k, v)        # fill only where the source is missing
    return out

def mv_zscores(teams, mv):
    vals = np.array([mv.get(ALIAS.get(t, t), np.nan) for t in teams])
    logv = np.log(vals)
    known = ~np.isnan(logv)
    z = np.zeros(len(teams))
    z[known] = (logv[known] - logv[known].mean()) / logv[known].std()
    return z, known

def adjusted(m, z, beta):
    m2 = dict(m)
    m2["attack"] = m["attack"] + beta * z
    m2["defense"] = m["defense"] + beta * z
    return m2

# --- connectivity-weighted prior (validated on held-out matches; squad value helps
# everywhere, more across confederations: within best ~0.10, cross best ~0.18) ---
BETA_CROSS, BETA_WITHIN = 0.18, 0.10

def load_conf():
    if not os.path.exists(MV_FILE):
        urllib.request.urlretrieve(MV_URL, MV_FILE)
    t = pd.read_csv(MV_FILE, compression="gzip")
    return {str(r.country_name): str(r.confederation) for r in t.itertuples(index=False)}

def setup(m):
    """Return (zmap, confmap) keyed by martj42 team names for the model's teams."""
    z, _ = mv_zscores(m["teams"], load_mv())
    conf = load_conf()
    zmap = {t: float(z[i]) for i, t in enumerate(m["teams"])}
    confmap = {t: conf.get(ALIAS.get(t, t)) for t in m["teams"]}
    return zmap, confmap

def mv_adjust(m, zmap, confmap, home, away):
    """Per-pair connectivity-weighted squad-value adjustment -> adjusted model.
    Applied ACROSS confederations (where DC is weakly identified), off within."""
    beta = BETA_CROSS if confmap.get(home) != confmap.get(away) else BETA_WITHIN
    if beta == 0.0:
        return m
    att, deff = m["attack"].copy(), m["defense"].copy()
    for t in (home, away):
        i = m["idx"][t]; zz = zmap.get(t, 0.0)
        att[i] += beta * zz; deff[i] += beta * zz
    m2 = dict(m); m2["attack"], m2["defense"] = att, deff
    return m2

def mv_wdl(m, zmap, confmap, home, away, neutral=True):
    """W/D/L with the connectivity-weighted squad-value prior."""
    return wc.wdl(mv_adjust(m, zmap, confmap, home, away), home, away, neutral=neutral)

if __name__ == "__main__":
    df = wc.load()
    mv = load_mv()
    m = wc.fit_dixon_coles(df[df.date < DC_CUT], ref_date=DC_CUT)
    z, known = mv_zscores(m["teams"], mv)
    print(f"MV coverage of rated teams: {known.sum()}/{len(m['teams'])}")

    post = df[df.date >= DC_CUT].copy()
    post = post[post.home_team.isin(m["idx"]) & post.away_team.isin(m["idx"])]
    Y = np.array([outcome(r.home_score, r.away_score) for r in post.itertuples(index=False)])
    comp = ~post.tournament.str.contains("friendly", case=False).values
    val = (post.date < CAL_CUT).values
    te = ~val
    HT, AT, NEU = post.home_team.values, post.away_team.values, post.neutral.values

    def probs(beta):
        m2 = adjusted(m, z, beta)
        return np.array([wc.wdl(m2, HT[k], AT[k], neutral=bool(NEU[k])) for k in range(len(post))])

    betas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    cache = {b: probs(b) for b in betas}
    print("\ntune beta on validation (competitive 2023-06..2024-06):")
    best = min(betas, key=lambda b: rps(cache[b][val & comp], Y[val & comp]))
    for b in betas:
        print(f"  beta={b:.2f}  val RPS={rps(cache[b][val & comp], Y[val & comp]):.4f}"
              + ("  <- best" if b == best else ""))

    cte = comp & te
    P0, Pb = cache[0.0][cte], cache[best][cte]
    print(f"\nTEST competitive >= {CAL_CUT.date()} (n={cte.sum()}):")
    print(f"  plain DC          RPS={rps(P0, Y[cte]):.4f}  logloss={logloss(P0, Y[cte]):.4f}")
    print(f"  + MV prior b={best:.2f}   RPS={rps(Pb, Y[cte]):.4f}  logloss={logloss(Pb, Y[cte]):.4f}")
    print("verdict:", "MV prior helps -> wire it in" if rps(Pb, Y[cte]) < rps(P0, Y[cte]) else "no help on this test set")
