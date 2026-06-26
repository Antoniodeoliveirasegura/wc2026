"""
Prototype the cheap, model-pure accuracy levers from the research pass and measure
each against the Dixon-Coles baseline, on the SAME out-of-sample split as validate()
(train < 2022-06-01, test after). Lower RPS = better.

Levers tested (all keep the "no scraped odds" purity):
  L1  DC + Elo log-opinion-pool blend   (geometric, weight sweep)
  L2  dynamic rho  (scale DC low-score correction by goal-expectation imbalance)
  L3  Elo *trajectory* over current Elo  (the SDR premise, tested cheaply)

Market-odds blend (the big lever) is NOT here: there are no historical international
bookmaker odds in the repo to backtest against. It can only be a predict-time blend.

Run: python experiments.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import wc_model as wc

CUTOFF = pd.Timestamp("2022-06-01")
TRAJ_DAYS = 180          # window for the Elo-trajectory feature (L3)


def outcome(h, a):
    return 0 if h > a else 1 if h == a else 2


def rps(P, Y):
    oh = np.eye(3)[Y]; cp = np.cumsum(P, 1); ca = np.cumsum(oh, 1)
    return np.mean(np.sum((cp - ca) ** 2, 1) / 2)


def logloss(P, Y):
    return -np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)))


def _K(t, m):
    t = str(t).lower()
    base = 50 if ("world cup" in t and "qual" not in t) else \
           40 if any(s in t for s in ("euro", "copa", "nations", "cup", "qualif")) else \
           20 if "friendly" in t else 30
    return base * (1.0 if m <= 1 else 1.5 if m == 2 else 1.75 + (m - 3) / 8.0)


def elo_features(df, home_adv=65.0, start=1500.0):
    """Leak-free per-match: current pre-match Elo diff AND the TRAJ_DAYS trajectory
    diff (rating now minus rating ~TRAJ_DAYS ago), home minus away. All from the past."""
    r = {}
    hist = {}                     # team -> list of (date, rating_after)
    diff = np.zeros(len(df))
    traj = np.zeros(len(df))
    dates = df.date.values
    def rating_days_ago(team, now, days):
        h = hist.get(team)
        if not h:
            return start
        cutoff = now - np.timedelta64(days, "D")
        past = start
        for d, rr in h:                       # h is chronological; small per team
            if d <= cutoff:
                past = rr
            else:
                break
        return past
    for k, row in enumerate(df.itertuples(index=False)):
        rh = r.get(row.home_team, start); ra = r.get(row.away_team, start)
        ha = 0.0 if row.neutral else home_adv
        diff[k] = (rh + ha) - ra
        th = rh - rating_days_ago(row.home_team, dates[k], TRAJ_DAYS)
        ta = ra - rating_days_ago(row.away_team, dates[k], TRAJ_DAYS)
        traj[k] = th - ta
        eh = 1.0 / (1.0 + 10 ** (-(rh + ha - ra) / 400.0))
        sh = 1.0 if row.home_score > row.away_score else 0.5 if row.home_score == row.away_score else 0.0
        d = _K(row.tournament, abs(row.home_score - row.away_score)) * (sh - eh)
        rh2, ra2 = rh + d, ra - d
        r[row.home_team] = rh2; r[row.away_team] = ra2
        hist.setdefault(row.home_team, []).append((dates[k], rh2))
        hist.setdefault(row.away_team, []).append((dates[k], ra2))
    return diff, traj


def wdl_dynrho(m, home, away, neutral=True, maxg=10):
    """DC W/D/L with rho scaled by goal-expectation imbalance (L2).
    rho_eff = rho * (1 - |lam-mu|/(lam+mu)): full draw-correction for even games."""
    i, j = m["idx"][home], m["idx"][away]
    log_lam = m["attack"][i] - m["defense"][j] + (0.0 if neutral else m["home_adv"])
    log_mu = m["attack"][j] - m["defense"][i]
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    rho = m["rho"] * (1 - abs(lam - mu) / (lam + mu))
    ks = np.arange(maxg + 1)
    M = np.outer(wc._pois(ks, lam), wc._pois(ks, mu))
    M[0, 0] *= 1 - lam * mu * rho
    M[0, 1] *= 1 + lam * rho
    M[1, 0] *= 1 + mu * rho
    M[1, 1] *= 1 - rho
    M = M / M.sum()
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def geo_blend(p1, p2, w):
    g = np.clip(p1, 1e-12, 1) ** w * np.clip(p2, 1e-12, 1) ** (1 - w)
    return g / g.sum(1, keepdims=True)


if __name__ == "__main__":
    df = wc.load().reset_index(drop=True)
    m = wc.fit_dixon_coles(df[df.date < CUTOFF], ref_date=CUTOFF)   # train-only fit

    eld, elt = elo_features(df)
    hts, ats, neu = df.home_team.values, df.away_team.values, df.neutral.values
    keep = (df.home_team.isin(m["idx"]) & df.away_team.isin(m["idx"])).values
    dcp = np.zeros((len(df), 3)); dynp = np.zeros((len(df), 3))
    for k in np.where(keep)[0]:
        dcp[k] = wc.wdl(m, hts[k], ats[k], neutral=bool(neu[k]))
        dynp[k] = wdl_dynrho(m, hts[k], ats[k], neutral=bool(neu[k]))

    idx = np.where(keep)[0]
    df2 = df.iloc[idx]
    Y = np.array([outcome(h, a) for h, a in zip(df2.home_score, df2.away_score)])
    is_train = (df2.date < CUTOFF).values
    te = ~is_train
    P_dc, P_dyn = dcp[idx], dynp[idx]
    Xe, Xt = eld[idx].reshape(-1, 1), elt[idx].reshape(-1, 1)

    # Elo -> calibrated W/D/L via train-only multinomial logistic (current diff)
    clf_e = LogisticRegression(max_iter=3000).fit(Xe[is_train], Y[is_train])
    P_elo = clf_e.predict_proba(Xe)

    def line(name, P):
        print(f"  {name:<26} RPS={rps(P[te], Y[te]):.4f}  logloss={logloss(P[te], Y[te]):.4f}")

    print(f"out-of-sample (train<{CUTOFF.date()}, test after; n_test={te.sum()}):\n")
    print("BASELINE")
    line("DC (current)", P_dc)
    line("Elo-logistic alone", P_elo)

    print("\nL1  DC x Elo log-opinion-pool (geometric, weight on DC)")
    best = (1.0, rps(P_dc[te], Y[te]))
    for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
        P = geo_blend(P_dc, P_elo, w)
        r = rps(P[te], Y[te])
        star = "  <- DC-only" if w == 1.0 else ("  *best*" if r < best[1] else "")
        if r < best[1] and w != 1.0:
            best = (w, r)
        print(f"     w={w:.1f}  RPS={r:.4f}{star}")
    print(f"     best blend w={best[0]:.1f}: RPS={best[1]:.4f}  "
          f"(delta vs DC {best[1]-rps(P_dc[te],Y[te]):+.4f})")

    print("\nL2  dynamic rho")
    line("DC + dynamic rho", P_dyn)
    print(f"     delta vs DC {rps(P_dyn[te],Y[te])-rps(P_dc[te],Y[te]):+.4f}")

    print("\nL3  Elo trajectory vs current Elo (multinomial logistic, train-fit)")
    P_cur = P_elo
    clf_t = LogisticRegression(max_iter=3000).fit(
        np.column_stack([Xe, Xt])[is_train], Y[is_train])
    P_traj = clf_t.predict_proba(np.column_stack([Xe, Xt]))
    line("Elo current only", P_cur)
    line("Elo current + trajectory", P_traj)
    print(f"     delta {rps(P_traj[te],Y[te])-rps(P_cur[te],Y[te]):+.4f}  "
          f"(negative = trajectory adds signal)")

    # self-check: DC baseline here matches a direct validate() of the same split
    vm, _, _ = wc.validate(df, cutoff=str(CUTOFF.date()))
    assert abs(rps(P_dc[te], Y[te]) - vm["rps"]) < 0.02, \
        f"DC baseline drift vs validate(): {rps(P_dc[te],Y[te]):.4f} vs {vm['rps']:.4f}"
    print(f"\nself-check ok: DC baseline {rps(P_dc[te],Y[te]):.4f} ~ validate() {vm['rps']:.4f}")
