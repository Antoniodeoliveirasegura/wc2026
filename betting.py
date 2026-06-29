"""
Betting recommendation engine: turn model probabilities into a SMALL set of value bets.

Philosophy: generate every candidate market internally, then filter HARD. Per game we
surface at most 3 "bet" + 5 "lean"; everything else is hidden (counted as avoids). A bet
only survives if it has a real edge, decent confidence, a non-noisy market, sane odds, and
isn't just a correlated echo of a better pick already chosen (e.g. don't recommend Team-win
AND Team-1.5 AND Team-scores-first — keep the best one).

Pure + offline: market probabilities come from the Dixon-Coles score matrix (no network here).
Odds are supplied by the caller (odds_api.py). edge = model_prob - implied_prob, where
implied = 1/decimal_odds is the break-even probability you actually get (so edge>0 == +EV).

NOTE: with no historical odds this layer cannot be ROI-backtested; it surfaces live edges,
not a proven-profitable system. Treat output as model disagreement with the market, not advice.

Run: python betting.py    # self-check on a synthetic "Paraguay" game
"""
from __future__ import annotations
from dataclasses import dataclass
from math import factorial
from typing import TypedDict, Literal
import numpy as np

# ----------------------------------------------------------------- tunable thresholds
MIN_EDGE = 0.04          # a model "bet" needs >= 4% edge vs the sharp line
LEAN_EDGE = 0.02         # a "lean" needs >= 2% edge
MARKET_EDGE = 0.01       # a "market value" pick: best price beats Pinnacle fair by >= 1%
                         # (sharp markets are efficient -> pure market value is small and rare)
MAX_MARKET_EDGE = 0.10   # above this a "market edge" is a stale/erroneous line, not real value
                         # (real sharp-market value tops out ~2%) -> ignore as a value signal
MIN_RELIABILITY = 0.60   # markets noisier than this can't be a "bet"
MIN_ODDS, MAX_ODDS = 1.25, 12.0   # skip junk favorites and lottery longshots
MAX_PROB = 0.92                   # skip near-certain selections (Over 0.5 etc.): tiny odds make
                                  # any "edge" line-noise, and the risk/reward is a lay-up
MAX_BETS, MAX_LEANS = 3, 5
FH_SHARE = 0.45          # share of a match's goals expected in the first half (approx)

# The model's W/D/L is RPS-validated but its goal LEVEL runs ~1.3x low vs the sharp market
# (totals_calibrate.py: model E[total] 1.99 vs Pinnacle 2.60; corroborated by ~2.7 WC goals/game).
# GOAL_SCALE calibrates lambda for the goal-SUM markets only (totals/BTTS/team totals/first half),
# leaving the validated outcome/margin model untouched. This makes totals HONEST (removes the
# old Under bias) — but since we calibrated TO the sharp line, we don't trust the model to BEAT
# it, so goal-sum markets are stakeable only on pure MARKET VALUE (a soft price beats Pinnacle).
GOAL_SCALE = 1.30
_SUM_DEPENDENT = ("total_", "team_total_", "btts", "fh_total_")

ET_SHARE = 1 / 3         # extra time is 30 of 90 minutes -> a third of the goal rate

# weights for the ranking score (edge-dominant; others break ties)
W_EDGE, W_KELLY, W_CONF, W_REL, W_MKT = 3.0, 1.0, 0.10, 0.05, 2.0

Confidence = Literal["low", "medium", "high"]
Verdict = Literal["bet", "lean", "avoid"]

# Market reliability (0-1): how trustworthy / non-noisy the market is. Drives confidence and
# the "not too noisy" filter. Moneyline/totals are sharp; halves and first-scorer are noisy.
RELIABILITY = {
    "moneyline": 1.00, "double_chance": 0.92, "btts": 0.82,
    "spread_0.5": 0.88, "spread_1.5": 0.82, "spread_2.5": 0.72,
    "total_1.5": 0.90, "total_2.5": 0.90, "total_3.5": 0.80, "total_0.5": 0.66, "total_4.5": 0.62,
    "team_total_0.5": 0.78, "team_total_1.5": 0.74, "team_total_2.5": 0.60,
    "fh_result": 0.55, "fh_total_0.5": 0.52, "fh_total_1.5": 0.52, "first_to_score": 0.50,
}


class BetJSON(TypedDict):
    market: str
    selection: str
    odds: float
    sharpOdds: float | None
    modelProbability: float
    sportsbookImpliedProbability: float
    edge: float
    marketEdge: float | None
    valueType: str
    confidence: Confidence
    recommendation: Verdict
    reason: str


def devig_power(odds: list[float]) -> list[float] | None:
    """No-vig fair probabilities via the POWER method (Clarke et al. 2017: universally beats
    multiplicative, best for 3-way 1X2). Find k with sum((1/o)^k)=1, then fair_i=(1/o_i)^k.
    Edge vs this fair line ~= true EV%. None if the market isn't a full >=2-outcome set."""
    r = [1.0 / o for o in odds if o and o > 1.0]
    if len(r) != len(odds) or len(r) < 2:
        return None
    s = sum(r)
    if s <= 1.0:                                   # no margin (or arb) -> just normalize
        return [x / s for x in r]
    lo, hi = 1.0, 10.0
    for _ in range(60):                            # bisection on the exponent
        k = (lo + hi) / 2
        if sum(x ** k for x in r) > 1.0:
            lo = k
        else:
            hi = k
    p = [x ** ((lo + hi) / 2) for x in r]
    t = sum(p)
    return [x / t for x in p]


class GameJSON(TypedDict):
    gameId: str
    topBets: list[BetJSON]
    recommendedBets: list[BetJSON]
    leans: list[BetJSON]
    avoidsCount: int


@dataclass(frozen=True)
class Candidate:
    market: str            # reliability key, e.g. "moneyline", "total_2.5"
    token: str             # canonical selection, e.g. "HOME", "OVER_2.5" (joins model<->odds)
    label: str             # human-readable, e.g. "Paraguay win"
    model_p: float
    odds: float            # decimal odds from the book (best available — the price you bet at)
    side: str              # correlation tag: which team it leans, or "OVER"/"UNDER"/""
    fair: float | None = None        # de-vigged benchmark for edge (Pinnacle's line when available)
    sharp_odds: float | None = None  # Pinnacle's price for this selection (CLV reference)
    pin_fair: float | None = None    # Pinnacle-only de-vigged prob (None if Pinnacle didn't price it)

    @property
    def implied(self) -> float:
        return self.fair if self.fair is not None else 1.0 / self.odds

    @property
    def edge(self) -> float:
        return self.model_p - self.implied            # model's disagreement with the (sharp) line

    @property
    def market_edge(self) -> float | None:
        """Pure market value: the best price beats Pinnacle's fair line (model-independent +EV).
        None when Pinnacle didn't price this selection."""
        return (self.pin_fair - 1.0 / self.odds) if self.pin_fair is not None else None

    @property
    def value_type(self) -> str:
        me = self.market_edge
        return "market" if (me is not None and MARKET_EDGE <= me <= MAX_MARKET_EDGE) else "model"

    @property
    def reliability(self) -> float:
        return RELIABILITY.get(self.market, 0.5)

    @property
    def kelly(self) -> float:
        # fraction of bankroll; (p*o - 1)/(o - 1). Captures edge AND price. Clamped >=0.
        return max(0.0, (self.model_p * self.odds - 1.0) / (self.odds - 1.0))

    @property
    def confidence_score(self) -> float:
        base = self.reliability * (0.6 + 0.8 * abs(self.model_p - 0.5))   # decisive => more confident
        me = self.market_edge
        if self.value_type == "market":              # best price beats Pinnacle fair -> price-confirmed
            base += 0.25
        elif self.edge >= 0.10:                      # implausibly large edge vs a liquid market ->
            base *= 0.55                             # almost certainly model error, not value -> distrust
        return max(0.0, min(1.0, base))

    @property
    def confidence(self) -> Confidence:
        s = self.confidence_score
        return "high" if s >= 0.70 else "medium" if s >= 0.50 else "low"

    @property
    def rank(self) -> float:
        return (W_EDGE * self.edge + W_KELLY * self.kelly + W_CONF * self.confidence_score
                + W_REL * self.reliability + W_MKT * max(0.0, self.market_edge or 0.0))


# ----------------------------------------------------------------- model market probabilities
def market_probs(M: np.ndarray, lam: float, mu: float) -> dict[str, dict[str, float]]:
    """All candidate markets' model probabilities, derived from the DC score matrix M
    (M[i,j] = P(home i, away j)) plus the goal expectations lam (home), mu (away).
    Returns {market: {token: prob}}. First-half / first-scorer use documented approximations."""
    G = M.shape[0] - 1
    idx = np.arange(G + 1)
    diff = idx[:, None] - idx[None, :]
    tot = idx[:, None] + idx[None, :]
    out: dict[str, dict[str, float]] = {}

    home, draw, away = float(M[diff > 0].sum()), float(M[diff == 0].sum()), float(M[diff < 0].sum())
    out["moneyline"] = {"HOME": home, "DRAW": draw, "AWAY": away}
    out["double_chance"] = {"HOME_DRAW": home + draw, "HOME_AWAY": home + away, "DRAW_AWAY": draw + away}

    for line in (0.5, 1.5, 2.5):
        out[f"spread_{line}"] = {
            f"HOME_-{line}": float(M[diff > line].sum()), f"AWAY_-{line}": float(M[diff < -line].sum()),
            f"HOME_+{line}": float(M[diff > -line].sum()), f"AWAY_+{line}": float(M[diff < line].sum()),
        }
    # Goal-SUM markets use a GOAL_SCALE-calibrated independent-Poisson matrix (the model's
    # neutral-venue lambda runs low vs the sharp totals line); outcome/margin markets above
    # keep the RPS-validated DC matrix M.
    fact = np.array([factorial(int(x)) for x in idx], dtype=float)
    sl, sm = GOAL_SCALE * lam, GOAL_SCALE * mu
    psh = np.exp(-sl) * sl ** idx / fact
    psa = np.exp(-sm) * sm ** idx / fact
    Ms = np.outer(psh, psa); Ms = Ms / Ms.sum()

    for line in (1.5, 2.5, 3.5):                # 0.5/4.5 dropped: near-certain, error-prone, low value
        over = float(Ms[tot > line].sum())
        out[f"total_{line}"] = {f"OVER_{line}": over, f"UNDER_{line}": 1 - over}

    mh, ma = Ms.sum(1), Ms.sum(0)                                # scaled home / away goal marginals
    for line in (0.5, 1.5, 2.5):
        out[f"team_total_{line}"] = {
            f"HOME_OVER_{line}": float(mh[idx > line].sum()), f"HOME_UNDER_{line}": float(mh[idx < line].sum()),
            f"AWAY_OVER_{line}": float(ma[idx > line].sum()), f"AWAY_UNDER_{line}": float(ma[idx < line].sum()),
        }

    btts_yes = float(Ms[1:, 1:].sum())
    out["btts"] = {"YES": btts_yes, "NO": 1 - btts_yes}

    # first half: independent Poisson on FH_SHARE of the SCALED goal rate
    ph = np.exp(-FH_SHARE * sl) * (FH_SHARE * sl) ** idx / fact
    pa = np.exp(-FH_SHARE * sm) * (FH_SHARE * sm) ** idx / fact
    FH = np.outer(ph, pa); FH = FH / FH.sum()
    out["fh_result"] = {"HOME": float(FH[diff > 0].sum()), "DRAW": float(FH[diff == 0].sum()),
                        "AWAY": float(FH[diff < 0].sum())}
    for line in (0.5, 1.5):
        ov = float(FH[tot > line].sum())
        out[f"fh_total_{line}"] = {f"OVER_{line}": ov, f"UNDER_{line}": 1 - ov}

    p_none = float(Ms[0, 0]); rate = sl + sm
    out["first_to_score"] = {
        "HOME": (sl / rate) * (1 - p_none) if rate else 0.0,
        "AWAY": (sm / rate) * (1 - p_none) if rate else 0.0, "NONE": p_none}
    return out


def advance_probs(M: np.ndarray, lam: float, mu: float) -> dict[str, float]:
    """Knockout: who wins the TIE (incl. extra time + penalties), not the 90' result.
    The 90' draw mass is redistributed by who'd survive the ET+pens mini-game: ET is
    modeled as ET_SHARE of the regulation goal rate (independent Poisson), and a
    still-level ET goes to a coin-flip shootout.
    Returns {"HOME","AWAY"} probabilities to advance (sum == 1)."""
    idx = np.arange(M.shape[0])
    diff = idx[:, None] - idx[None, :]
    ph, pd, pa = float(M[diff > 0].sum()), float(M[diff == 0].sum()), float(M[diff < 0].sum())
    fact = np.array([factorial(int(x)) for x in idx], dtype=float)
    el, em = ET_SHARE * lam, ET_SHARE * mu
    eh = np.exp(-el) * el ** idx / fact
    ea = np.exp(-em) * em ** idx / fact
    ET = np.outer(eh, ea); ET = ET / ET.sum()
    et_home, et_draw = float(ET[diff > 0].sum()), float(ET[diff == 0].sum())
    q_home = et_home + 0.5 * et_draw          # ponytail: pens are a 50/50 coin flip; swap in a
                                              # skill-weighted shootout edge if it ever matters
    return {"HOME": ph + pd * q_home, "AWAY": pa + pd * (1 - q_home)}


# ----------------------------------------------------------------- recommendation logic
def _reason(c: Candidate, decorrelated_from: str | None) -> str:
    if c.value_type == "market":
        base = (f"Market value: best price ({c.odds:.2f}) beats Pinnacle's fair line by "
                f"{c.market_edge:+.0%} &mdash; +EV regardless of our model. {c.confidence} confidence.")
    else:
        base = (f"Model {c.model_p:.0%} vs Pinnacle fair {c.implied:.0%} (+{c.edge:.0%}), "
                f"{c.confidence} confidence on a {'sharp' if c.reliability >= 0.8 else 'moderate'} market.")
    if decorrelated_from:
        base += f" Preferred over correlated picks ({decorrelated_from})."
    return base


def _to_json(c: Candidate, verdict: Verdict, reason: str) -> BetJSON:
    return {"market": c.market, "selection": c.label, "odds": round(c.odds, 2),
            "sharpOdds": round(c.sharp_odds, 2) if c.sharp_odds else None,
            "modelProbability": round(c.model_p, 4),
            "sportsbookImpliedProbability": round(c.implied, 4),
            "edge": round(c.edge, 4),
            "marketEdge": round(c.market_edge, 4) if c.market_edge is not None else None,
            "valueType": c.value_type, "confidence": c.confidence,
            "recommendation": verdict, "reason": reason}


def _bet_eligible(c: Candidate) -> bool:
    """Outcome/margin markets are always stakeable. Goal-SUM markets (totals/BTTS/team totals/
    first half) are calibrated but we don't trust the model to beat the sharp totals line, so
    they can be a 'bet' only on pure market value (a soft price beats Pinnacle); else lean."""
    if not c.market.startswith(_SUM_DEPENDENT):
        return True
    return c.value_type == "market"


def _verdict(c: Candidate) -> Verdict:
    if (c.edge >= MIN_EDGE and c.confidence != "low" and c.reliability >= MIN_RELIABILITY
            and _bet_eligible(c)):
        return "bet"
    return "lean" if c.edge >= LEAN_EDGE else "avoid"


def _display_rank(c: Candidate) -> float:
    pen = 1.0 if _bet_eligible(c) else 0.35            # non-value sum-markets shown but downranked
    return c.rank * pen


def top_bets(candidates: list[Candidate], n: int = 5) -> list[BetJSON]:
    """The n best value selections for a game: positive edge, sane odds, de-correlated by side
    (one per team / over-under), ranked. Each labeled bet/lean/avoid honestly."""
    pool = sorted([c for c in candidates
                   if MIN_ODDS <= c.odds <= MAX_ODDS and c.implied <= MAX_PROB and c.edge > 0],
                  key=_display_rank, reverse=True)
    picked: list[Candidate] = []; sides: set[str] = set()
    for c in pool:
        if c.side and c.side in sides:
            continue
        picked.append(c)
        if c.side:
            sides.add(c.side)
        if len(picked) >= n:
            break
    return [_to_json(c, _verdict(c), _reason(c, None)) for c in picked]


def recommend(game_id: str, candidates: list[Candidate]) -> GameJSON:
    """Filter -> de-correlate -> rank -> tier into <=3 bets, <=5 leans, rest avoid.
    Also emits topBets: the 5 best de-correlated selections for display."""
    sane = [c for c in candidates if MIN_ODDS <= c.odds <= MAX_ODDS and c.implied <= MAX_PROB]
    qualifiers = sorted(
        [c for c in sane if c.edge >= MIN_EDGE and c.confidence != "low"
         and c.reliability >= MIN_RELIABILITY and _bet_eligible(c)],
        key=lambda c: c.rank, reverse=True)

    bets: list[BetJSON] = []
    taken_sides: dict[str, str] = {}        # correlation tag -> label already chosen
    overflow: list[Candidate] = []
    for c in qualifiers:
        clash = taken_sides.get(c.side) if c.side else None
        if clash is None and len(bets) < MAX_BETS:
            bets.append(_to_json(c, "bet", _reason(c, None)))
            if c.side:
                taken_sides[c.side] = c.label
        else:
            overflow.append(c)               # correlated echo or past the bet cap -> consider as lean

    chosen = {(b["market"], b["selection"]) for b in bets}
    lean_pool = sorted(
        overflow + [c for c in sane if c.edge >= LEAN_EDGE and (c.market, c.label) not in chosen],
        key=lambda c: c.rank, reverse=True)
    leans: list[BetJSON] = []
    lean_sides = dict(taken_sides)
    seen = set(chosen)
    for c in lean_pool:
        key = (c.market, c.label)
        if key in seen or len(leans) >= MAX_LEANS:
            continue
        clash = lean_sides.get(c.side) if c.side else None
        leans.append(_to_json(c, "lean", _reason(c, clash))); seen.add(key)
        if c.side:
            lean_sides.setdefault(c.side, c.label)

    avoids = len(candidates) - len(bets) - len(leans)
    return {"gameId": game_id, "topBets": top_bets(candidates), "recommendedBets": bets,
            "leans": leans, "avoidsCount": max(0, avoids)}


if __name__ == "__main__":
    # synthetic Paraguay-Australia: model likes Paraguay; book offers many correlated Paraguay bets.
    # Expect: only the best 1-2 Paraguay bets surface; the rest become leans/avoids (not all of them).
    C = Candidate
    cands = [
        C("moneyline", "HOME", "Paraguay win", 0.50, 2.65, "Paraguay"),
        C("spread_0.5", "HOME_-0.5", "Paraguay -0.5", 0.50, 2.60, "Paraguay"),
        C("team_total_0.5", "HOME_OVER_0.5", "Paraguay over 0.5 goals", 0.72, 1.52, "Paraguay"),
        C("first_to_score", "HOME", "Paraguay to score first", 0.46, 2.10, "Paraguay"),
        C("double_chance", "HOME_DRAW", "Paraguay or draw", 0.74, 1.36, "Paraguay"),
        C("total_2.5", "UNDER_2.5", "Under 2.5 goals", 0.61, 1.80, "UNDER"),
        C("moneyline", "AWAY", "Australia win", 0.26, 4.40, "Australia"),   # no edge -> avoid
        C("btts", "NO", "Both teams to score - No", 0.55, 1.95, "UNDER"),
    ]
    out = recommend("paraguay-australia", cands)
    print(f"bets={len(out['recommendedBets'])} leans={len(out['leans'])} avoids={out['avoidsCount']}")
    for b in out["recommendedBets"]:
        print(f"  BET  {b['selection']:<28} edge={b['edge']:+.0%} {b['confidence']}")
    for b in out["leans"]:
        print(f"  lean {b['selection']:<28} edge={b['edge']:+.0%} {b['confidence']}")
    assert len(out["recommendedBets"]) <= MAX_BETS and len(out["leans"]) <= MAX_LEANS
    para_bets = [b for b in out["recommendedBets"] if "Paraguay" in b["selection"]]
    assert len(para_bets) <= 1, "correlated Paraguay bets not de-duplicated"
    assert all(b["edge"] >= MIN_EDGE for b in out["recommendedBets"]), "a bet slipped under min edge"
    print("self-check ok: hard filter + de-correlation + caps hold")

    # advance_probs: redistribute the 90' draw into who wins the tie (ET + pens)
    lam, mu = 1.6, 1.0                                   # home favored
    M = np.outer(np.exp(-lam) * lam ** np.arange(11) / np.array([factorial(i) for i in range(11)]),
                 np.exp(-mu) * mu ** np.arange(11) / np.array([factorial(i) for i in range(11)]))
    M = M / M.sum()
    adv = advance_probs(M, lam, mu)
    idx = np.arange(11); d = idx[:, None] - idx[None, :]
    ph, pa = float(M[d > 0].sum()), float(M[d < 0].sum())
    assert abs(adv["HOME"] + adv["AWAY"] - 1.0) < 1e-9, "advance probs must sum to 1 (no draw)"
    assert adv["HOME"] > ph and adv["AWAY"] > pa, "advance must absorb draw mass for both sides"
    assert adv["HOME"] > adv["AWAY"], "stronger side should be likelier to advance"
    print(f"advance self-check ok: HOME {adv['HOME']:.0%} / AWAY {adv['AWAY']:.0%} "
          f"(from 90' {ph:.0%}/{1-ph-pa:.0%} draw/{pa:.0%})")
