"""
Acclimatization-asymmetry altitude penalty (premise validated in altitude_test.py).

Backtest finding: at NON-neutral high-altitude venues the acclimatized home side
overperforms DC by ~+0.20 goal-diff per 1000m (p<0.001) and sea-level visitors score
fewer goals; at NEUTRAL venues the effect vanishes (p=0.40). So it is an acclimatization
asymmetry, not a blanket venue effect: penalize only the team whose home altitude sits
well below the venue, scaling down its expected goals. Two sea-level teams at altitude
net to zero (matches the neutral null).

Only Mexico City (2240m) and Guadalajara/Zapopan (1566m) host 2026 games; every US/CA
venue is ~sea level, so the penalty is a no-op everywhere else.
"""
from __future__ import annotations
import unicodedata

ALT_K = 0.10   # log-lambda penalty per 1000m of upward shock (altitude_test.py lower bound; tunable)

# 2026 host venues that sit at altitude (city spellings as they appear in the fixtures).
ALT_M = {"mexico city": 2240, "ciudad de mexico": 2240, "guadalajara": 1566, "zapopan": 1566}

# Home elevation of the acclimatized WC sides; every other team defaults to ~sea level.
TEAM_HOME_ALT = {"Mexico": 2240, "Colombia": 2640, "Ecuador": 2850}


def _norm(s):
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def venue_alt(city):
    return ALT_M.get(_norm(city), 0.0)


def alt_adjust(m, home, away, city):
    """Return the model with each team's attack reduced by its upward altitude shock.
    No-op at sea-level venues or when both sides are equally (un)acclimatized."""
    V = venue_alt(city)
    if V <= 0:
        return m
    sh = max(0.0, V - TEAM_HOME_ALT.get(home, 0.0)) / 1000.0
    sa = max(0.0, V - TEAM_HOME_ALT.get(away, 0.0)) / 1000.0
    if sh == 0.0 and sa == 0.0:
        return m
    att = m["attack"].copy()
    att[m["idx"][home]] -= ALT_K * sh
    att[m["idx"][away]] -= ALT_K * sa
    m2 = dict(m); m2["attack"] = att
    return m2
