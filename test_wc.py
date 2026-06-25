"""Fast behavioural tests for the WC model (no data download, no slow MLE fit).

Run: pytest -q
"""
import numpy as np
import pandas as pd
import wc_model as wc
import blite
import simulate as sim


def _toy_model():
    return {"teams": ["A", "B"], "idx": {"A": 0, "B": 1},
            "attack": np.array([0.6, -0.6]), "defense": np.array([0.3, -0.3]),
            "home_adv": 0.3, "rho": -0.05}


def test_wdl_sums_to_one_and_favours_stronger():
    ph, pdraw, pa = wc.wdl(_toy_model(), "A", "B", neutral=True)
    assert abs(ph + pdraw + pa - 1.0) < 1e-9
    assert ph > pa


def test_score_matrix_normalises_and_nonnegative():
    M = wc.score_matrix(_toy_model(), "A", "B")
    assert abs(M.sum() - 1.0) < 1e-9
    assert (M >= 0).all()


def test_home_advantage_increases_win_prob():
    m = _toy_model()
    ph_home, _, _ = wc.wdl(m, "A", "B", neutral=False)
    ph_neutral, _, _ = wc.wdl(m, "A", "B", neutral=True)
    assert ph_home > ph_neutral


def test_rps_perfect_is_zero_worst_is_one():
    Y = np.array([0])
    assert blite.rps(np.array([[1.0, 0.0, 0.0]]), Y) < 1e-9
    assert abs(blite.rps(np.array([[0.0, 0.0, 1.0]]), Y) - 1.0) < 1e-9


def test_elo_ranks_stronger_team_higher():
    rows = [{"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
             "home_team": "Strong", "away_team": "Weak", "home_score": 3,
             "away_score": 0, "tournament": "Friendly", "neutral": True}
            for i in range(10)]
    elo = wc.fit_elo(pd.DataFrame(rows))
    assert elo["Strong"] > elo["Weak"]


def test_dixon_coles_tiny_fit_gives_valid_probs():
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    d0 = pd.Timestamp("2020-01-01")
    rows = []
    for k in range(80):
        i, j = rng.choice(4, 2, replace=False)
        rows.append({"date": d0 + pd.Timedelta(days=k), "home_team": teams[i],
                     "away_team": teams[j], "home_score": int(rng.poisson(1.5)),
                     "away_score": int(rng.poisson(1.1)), "tournament": "Friendly",
                     "neutral": False})
    m = wc.fit_dixon_coles(pd.DataFrame(rows))
    ph, pdraw, pa = wc.wdl(m, "A", "B", neutral=True)
    assert abs(ph + pdraw + pa - 1.0) < 1e-9


def _toy_bracket():
    W = [(i, i) for i in range(12)]
    RU = [(12 + i, i) for i in range(12)]
    TH = [(24 + i, i) for i in range(8)]
    return W, RU, TH


def test_draw_r32_uses_all_32_in_16_matches():
    matches = sim.draw_r32(*_toy_bracket())
    assert len(matches) == 16
    flat = [t for pair in matches for t in pair]
    assert len(set(flat)) == 32


def test_simulate_knockouts_distribution():
    n = 32
    A = np.full((n, n), 0.5); np.fill_diagonal(A, 0.0)
    R = sim.simulate_knockouts(list(range(n)), list(range(n)), A, n_sims=200)
    assert sum(R["champ"].values()) == 200
    assert sum(R["qualify"].values()) == 200 * n


def test_simulate_from_groups_qualifies_32():
    n = 48
    A = np.full((n, n), 0.5); np.fill_diagonal(A, 0.0)
    groups_idx = [list(range(g * 4, g * 4 + 4)) for g in range(12)]
    base = (np.arange(n) % 7, np.zeros(n, int), np.zeros(n, int))
    R = sim.simulate_from_groups(groups_idx, base, [], A, n_sims=200)
    assert sum(R["champ"].values()) == 200
    assert sum(R["qualify"].values()) == 200 * 32   # 2 per group + 8 thirds
