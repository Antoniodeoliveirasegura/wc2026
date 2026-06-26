"""
Is the model's goal LEVEL systematically low vs the sharp market? (totals calibration probe)

The model's W/D/L is RPS-validated, but live odds showed its totals skew Under / draws inflate
— symptoms of low expected goals (lambda). Pinnacle's de-vigged O/U line is the sharp estimate
of true total goals. Here we MEASURE the gap per game: model E[total] vs Pinnacle-implied
E[total], fit a single GOAL_SCALE on (lambda+mu), and show what it does to the draw rate.

Deploy only if the gap is systematic. Run: python totals_calibrate.py
"""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from scipy.optimize import brentq
import wc_model as wc
import marketvalue as mvmod
import altitude as altmod
import simulate as sim
import betting as bet
import odds_api


def implied_lambda_total(p_over: float, line: float) -> float | None:
    """Total-goal Poisson rate whose P(total > line) equals p_over."""
    k = int(line)                                   # over 2.5 -> P(X >= 3) = sf(2)
    if not 0.01 < p_over < 0.99:
        return None
    try:
        return brentq(lambda L: poisson.sf(k, L) - p_over, 0.1, 8.0)
    except ValueError:
        return None


def model_lambdas(adj, home, away):
    i, j = adj["idx"][home], adj["idx"][away]
    return (float(np.exp(adj["attack"][i] - adj["defense"][j])),
            float(np.exp(adj["attack"][j] - adj["defense"][i])))


def model_pover(adj, home, away, line, scale=1.0):
    s2 = dict(adj)                                  # scale both rates by `scale` (lambda *= s)
    if scale != 1.0:
        s2["attack"] = adj["attack"] + np.log(scale)
    M = wc.score_matrix(s2, home, away, neutral=True, maxg=sim.MAXG)
    G = M.shape[0] - 1
    tot = np.add.outer(np.arange(G + 1), np.arange(G + 1))
    return float(M[tot > line].sum())


def draw_rate(adj, home, away, scale=1.0):
    s2 = dict(adj)
    if scale != 1.0:
        s2["attack"] = adj["attack"] + np.log(scale)
    return float(wc.wdl(s2, home, away, neutral=True)[1])


if __name__ == "__main__":
    df = wc.load()
    m = sim.get_model(df)
    zmap, confmap = mvmod.setup(m)
    fcity = sim.fixture_cities(df)
    games = odds_api.normalize(odds_api.fetch_raw())

    rows = []
    for (home, away), info in games.items():
        if home not in m["idx"] or away not in m["idx"]:
            continue
        sharp = info["sharp"]
        ov, un = sharp.get(("total_2.5", "OVER_2.5")), sharp.get(("total_2.5", "UNDER_2.5"))
        if not ov or not un:
            continue
        dv = bet.devig_power([ov, un])
        if not dv:
            continue
        pin_over = dv[0]
        pin_lt = implied_lambda_total(pin_over, 2.5)
        if not pin_lt:
            continue
        city = fcity.get(frozenset((home, away)), "")
        adj = altmod.alt_adjust(mvmod.mv_adjust(m, zmap, confmap, home, away), home, away, city)
        lam, mu = model_lambdas(adj, home, away)
        rows.append((home, away, lam + mu, pin_lt, model_pover(adj, home, away, 2.5), pin_over, adj))

    if not rows:
        print("no Pinnacle O/U 2.5 lines available")
        raise SystemExit

    ratios = np.array([r[3] / r[2] for r in rows])             # pinnacle / model total rate
    scale = float(np.median(ratios))
    mod_lt = np.array([r[2] for r in rows]); pin_lt = np.array([r[3] for r in rows])
    print(f"games with Pinnacle O/U 2.5: {len(rows)}")
    print(f"model mean E[total]:    {mod_lt.mean():.2f}")
    print(f"Pinnacle mean E[total]: {pin_lt.mean():.2f}   "
          f"(gap {pin_lt.mean()-mod_lt.mean():+.2f} goals/game)")
    print(f"per-game ratio pinnacle/model: median {scale:.3f}  (mean {ratios.mean():.3f}, "
          f"sd {ratios.std():.3f})")
    print(f"\n=> fitted GOAL_SCALE = {scale:.3f}")

    over_err0 = np.mean([abs(r[4] - r[5]) for r in rows])
    over_err1 = np.mean([abs(model_pover(r[6], r[0], r[1], 2.5, scale) - r[5]) for r in rows])
    dr0 = np.mean([draw_rate(r[6], r[0], r[1]) for r in rows])
    dr1 = np.mean([draw_rate(r[6], r[0], r[1], scale) for r in rows])
    print(f"\nmean |model P(Over2.5) - Pinnacle|:  {over_err0:.3f} -> {over_err1:.3f}  (scaled)")
    print(f"mean model draw probability:         {dr0:.3f} -> {dr1:.3f}  "
          f"(scaled; lower = less draw-inflation)")

    base = wc.validate(df, cutoff="2022-06-01")[0]["rps"]
    print(f"\nout-of-sample W/D/L RPS (unscaled model): {base:.4f}")
    print("  (scale is a WC-totals adjustment; deploy on totals markets, check moneyline RPS holds)")

    assert 0.5 < scale < 2.0, f"implausible scale {scale} — check de-vig / lambda mapping"
    print("\nself-check ok: scale in sane range")
