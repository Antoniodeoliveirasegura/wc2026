"""
Does altitude add signal BEYOND Dixon-Coles team strength? (Tier-1 premise test.)

The 2026 question: Mexico City (2240m) and Guadalajara (1566m) host games; sea-level
visitors are said to be penalized. Before wiring any altitude term into the sim, test
the premise on historical internationals.

Method: fit DC, take each match's expected goal diff E[home]-E[away] = lam-mu, and the
RESIDUAL = actual_gd - expected_gd. Regress the residual on venue altitude. If altitude
helps the (acclimatized) home side beyond what strength explains, the residual rises with
venue altitude. NOTE: DC partly absorbs altitude into the ratings of high-altitude teams
(Bolivia etc.), so any effect we measure is a LOWER BOUND on the true venue effect.

Run: python altitude_test.py
"""
from __future__ import annotations
import unicodedata
import numpy as np
import pandas as pd
from scipy import stats
import wc_model as wc

# Elevation (m) for high-altitude football cities; everything else defaults to ~sea level.
# Only high venues carry signal, so sea-level precision elsewhere is irrelevant.
ALT_M = {
    # Bolivia
    "la paz": 3640, "el alto": 4150, "oruro": 3700, "potosi": 4067, "sucre": 2810,
    "cochabamba": 2560, "tarija": 1870,
    # Ecuador / Colombia / Peru
    "quito": 2850, "ambato": 2577, "riobamba": 2754, "bogota": 2640, "tunja": 2820,
    "pasto": 2527, "manizales": 2160, "cusco": 3400, "arequipa": 2335, "huancayo": 3250,
    "juliaca": 3825,
    # Mexico / Central America (2026-relevant)
    "mexico city": 2240, "ciudad de mexico": 2240, "toluca": 2660, "puebla": 2135,
    "pachuca": 2400, "guadalajara": 1566, "queretaro": 1820, "leon": 1815,
    "san luis potosi": 1860, "aguascalientes": 1880, "guatemala city": 1500,
    "quetzaltenango": 2330, "san jose": 1170, "tegucigalpa": 990,
    # Africa
    "addis ababa": 2355, "asmara": 2325, "nairobi": 1795, "johannesburg": 1753,
    "pretoria": 1339, "bloemfontein": 1395, "kampala": 1190, "kigali": 1567,
    "harare": 1490, "windhoek": 1655, "antananarivo": 1280, "lusaka": 1279,
    "mbabane": 1243, "sanaa": 2250, "sana'a": 2250, "nakuru": 1850,
    # Asia / other
    "tehran": 1200, "thimphu": 2334, "kathmandu": 1400, "ulaanbaatar": 1300,
    "bishkek": 800, "almaty": 850, "addis": 2355,
}


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    return s


def venue_alt(city):
    return ALT_M.get(norm(city), 0.0)


def expected_gd(m, home, away, neutral):
    """DC expected home-goal-diff E[home]-E[away] = lam - mu for one match."""
    i, j = m["idx"][home], m["idx"][away]
    lam = np.exp(m["attack"][i] - m["defense"][j] + (0.0 if neutral else m["home_adv"]))
    mu = np.exp(m["attack"][j] - m["defense"][i])
    return lam - mu


def report(label, res, alt, mask):
    r, a = res[mask], alt[mask]
    if len(r) < 30:
        print(f"  {label:<42} n={len(r):<5} (too few)")
        return
    lr = stats.linregress(a, r)
    print(f"  {label:<42} n={len(r):<5} slope={lr.slope*1000:+.3f} gd/1000m  "
          f"p={lr.pvalue:.3f}  mean_resid={r.mean():+.3f}")


if __name__ == "__main__":
    df = wc.load()
    m = wc.fit_dixon_coles(df, ref_date=df.date.max())

    keep = df.home_team.isin(m["idx"]) & df.away_team.isin(m["idx"])
    d = df[keep].copy()
    d["alt"] = d.city.map(venue_alt).values
    d["egd"] = [expected_gd(m, r.home_team, r.away_team, bool(r.neutral))
                for r in d.itertuples(index=False)]
    d["resid"] = (d.home_score - d.away_score) - d.egd
    res, alt, neu = d.resid.values, d.alt.values, d.neutral.values.astype(bool)

    print(f"matches: {len(d):,}  |  at venues >1500m: {(alt > 1500).sum():,}  "
          f">2500m: {(alt > 2500).sum():,}")
    print("\nresidual = actual goal-diff minus DC expectation; +slope => altitude helps home beyond strength")
    report("ALL matches", res, alt, np.ones(len(d), bool))
    report("non-neutral only (home acclimatized)", res, alt, ~neu)
    report("neutral only", res, alt, neu)

    print("\nbucketed mean residual (non-neutral):")
    nn = ~neu
    for lo, hi in [(0, 250), (250, 1000), (1000, 1500), (1500, 2500), (2500, 9999)]:
        mk = nn & (alt >= lo) & (alt < hi)
        if mk.sum():
            print(f"  venue {lo:>4}-{hi:<4}m  n={mk.sum():<5} mean_resid={res[mk].mean():+.3f}")

    # away-side penalty: away goals minus DC expectation, sea-level vs high-altitude venue
    aw_resid = (d.away_score.values - np.exp(
        [m["attack"][m["idx"][r.away_team]] - m["defense"][m["idx"][r.home_team]]
         for r in d.itertuples(index=False)]))
    lo_mask, hi_mask = nn & (alt < 250), nn & (alt > 2000)
    print(f"\naway-goals residual (non-neutral): sea-level={aw_resid[lo_mask].mean():+.3f} "
          f"(n={lo_mask.sum()})  vs >2000m={aw_resid[hi_mask].mean():+.3f} (n={hi_mask.sum()})")
    print("  negative at altitude = visitors score fewer than strength predicts (the 2026 worry)")

    # self-check: altitude lookup actually matched some high venues
    assert (alt > 1500).sum() > 50, "altitude lookup matched too few venues — check city spellings"
    print("\nself-check ok: altitude lookup matched real high-venue matches")
