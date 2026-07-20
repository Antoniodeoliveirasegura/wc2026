"""
Reconstructed PRE-TOURNAMENT title odds for the 2026 World Cup.

Everything the tournament revealed is held out: BOTH ratings that feed the match model --
Dixon-Coles and the pi-ratings -- are fit only on matches before 2026-06-11, and the
simulation starts from an empty table, playing all 72 group games and then the knockouts.
What it does use is what was already known at the draw: the 48-team field, the group
composition, and the venue schedule.

IMPORTANT -- this is a RECONSTRUCTION, not a published prediction. It is computed after the
fact from a model that never saw a tournament result, which makes it an honest backtest of
what the method would have said. It is NOT evidence that this call was made in advance.

Deterministic given the same inputs (random.seed(0)), and the result can never change now
that the tournament is over, so it is cached to pre_wc_odds.json and committed.

Run: python pre_wc_odds.py   ->   pre_wc_odds.json
"""
from __future__ import annotations
import os, json, random, datetime
import numpy as np
import wc_model as wc
import marketvalue as mvmod
import simulate as sim

OUT = os.path.join(os.path.dirname(__file__), "pre_wc_odds.json")


def build():
    df = wc.load()
    pre = df[df.date < sim.WC_START]
    print(f"training on {len(pre):,} matches before {sim.WC_START.date()} "
          f"(holding out {len(df) - len(pre)} played since)")

    m = sim.pre_wc_model(df)                     # DC fit on pre-tournament data only
    zmap, confmap = mvmod.setup(m)
    # The live site blends DC with pi-ratings fit on everything; refit them pre-cutoff too,
    # otherwise the tournament leaks back in through the second rating.
    _, pi_predict = wc.fit_pi_wdl(pre)

    wcdf = sim.wc_games(df)
    gdf, _ = sim.split_games(wcdf)
    groups = sim.groups_from(gdf)                # composition only -- known at the draw
    teams = sorted({t for g in groups for t in g})
    tidx = {t: i for i, t in enumerate(teams)}
    print(f"{len(teams)} teams, {len(groups)} groups")

    A = sim.adv_matrix(m, teams, zmap, confmap, pi_predict)
    A_azteca = sim.adv_matrix(m, teams, zmap, confmap, pi_predict, city="Mexico City",
                              home_nations=sim.KO_HOME | {"Mexico"})
    azteca_group = next((i for i, g in enumerate(groups) if "Mexico" in g), None)

    # Empty table, every group game still to play: the state on the morning of 11 June.
    base = (np.zeros(len(teams), int), np.zeros(len(teams), int), np.zeros(len(teams), int))
    fcity = sim.fixture_cities(df)               # published schedule -- known in advance
    allpairs = [(g[i], g[j]) for g in groups
                for i in range(len(g)) for j in range(i + 1, len(g))]
    assert len(allpairs) == 72, f"expected 72 group games, got {len(allpairs)}"
    rem = [(tidx[a], tidx[b], *sim.score_dist(m, zmap, confmap, a, b,
            fcity.get(frozenset((a, b)), ""))) for a, b in allpairs]

    random.seed(0)
    R = sim.simulate_from_groups([[tidx[t] for t in g] for g in groups], base, rem, A,
                                 A_azteca=A_azteca, azteca_group=azteca_group)
    n = sim.N_SIMS
    assert abs(sum(R["champ"].values()) - n) < 1, "champion counts must sum to the sim count"

    return {"generated": datetime.date.today().isoformat(), "n_sims": n,
            "cutoff": str(sim.WC_START.date()),
            "teams": {t: {k: R[k][tidx[t]] / n
                          for k in ("qualify", "r16", "sf", "final", "champ")}
                      for t in teams}}


if __name__ == "__main__":
    d = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    top = sorted(d["teams"].items(), key=lambda kv: -kv[1]["champ"])[:16]
    print(f"\n  {'team':<18}{'champ':>8}{'final':>8}{'qualify':>9}")
    for t, v in top:
        print(f"  {t:<18}{v['champ']:>7.1%}{v['final']:>8.1%}{v['qualify']:>9.0%}")
    print(f"\nwrote {os.path.basename(OUT)} ({len(d['teams'])} teams, {d['n_sims']:,} sims)")
