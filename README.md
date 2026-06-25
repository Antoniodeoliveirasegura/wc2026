# wc2026 — World Cup 2026 forecast

A from-scratch statistical model for the 2026 FIFA World Cup: match Win/Draw/Loss
probabilities from a Dixon-Coles scoreline model with a connectivity-weighted
squad-value prior, plus a Monte-Carlo tournament simulator that runs the knockouts
from the (already-played) group stage.

## Results

Out-of-sample accuracy on competitive internationals: **RPS ≈ 0.16–0.17** —
bookmaker-comparable. Current title odds (50k sims):

| Team | Champion | Final | Semi |
|---|---|---|---|
| Spain | 16.3% | 26.3% | 40.2% |
| Argentina | 14.7% | 24.2% | 38.1% |
| England | 11.0% | 19.5% | 32.9% |
| Brazil | 10.9% | 19.4% | 33.1% |
| France | 9.3% | 17.2% | 30.2% |
| Portugal | 8.1% | 15.2% | 27.1% |
| Germany | 5.8% | 12.4% | 24.4% |
| Netherlands | 4.7% | 10.4% | 21.4% |

> No model "calls the winner" — the favorite tops out ~16%. The deliverable is a
> calibrated distribution, not a single pick. Full 32-team table in `forecast.md`.

## How it works

- **Dixon-Coles** (`wc_model.py`): bivariate-Poisson scoreline model, MLE, 3-yr time
  decay (tuned), friendlies down-weighted. Proper W/D/L probabilities.
- **Connectivity-weighted squad-value prior** (`marketvalue.py`): Transfermarkt
  national squad values nudge team strength — more *across* confederations (where
  results-based ratings are weakly linked: the Morocco-vs-Portugal "no common
  opponents" problem), less within. Validated out-of-sample (−0.0013 within /
  −0.0040 across confederations).
- **Elo** (`wc_model.py`): independent dynamic rating, used as a cross-check.
- **Tournament sim** (`simulate.py`): real group standings + 8 best thirds, then
  Monte-Carlo knockouts with the real draw structure (group winners protected,
  same-group separation) and host advantage (USA/Canada/Mexico).

## What we tested and rejected (honest negatives)

Most ideas don't beat a well-tuned Dixon-Coles — documented so they aren't re-tried:

- Cheap features (form, rest, stage) → GBM (`blite.py`): wash vs a refit DC.
- Isotonic calibration (`calibrate.py`): *hurts* — DC is already well-calibrated.
- *Uniform* market-value prior (`marketvalue.py`): no help; it perturbs the
  well-identified within-confederation pairs. Only the **connectivity-weighted**
  version (`mv_connectivity.py` validates it) works.
- xG-based ratings (`xg_test.py`): xG *is* a better strength signal than goals
  (−0.017 log-loss on 127 World Cup matches), but international xG exists only for
  recent major tournaments — undeployable for a 49k-match model. Premise holds,
  the data doesn't.

Grounded in the literature: Groll et al.'s hybrid random forest, Caley's "PADDLIN'"
2026 model, and `amirdaraee/world-cup-predictions` (whose squad-value prior was its
"largest single upgrade").

## Setup

    pip install -r requirements.txt
    python simulate.py         # tournament forecast (auto-downloads data)
    python wc_model.py         # engine + out-of-sample validation
    python mv_connectivity.py  # the market-value validation
    pytest -q                  # tests

Data auto-downloads: international results (`martj42`) + national-team squad values
(`dcaribou/transfermarkt-datasets`); both gitignored. The 4 priciest squads
(ES/FR/EN/PT) are null in the source and filled with public values (see `marketvalue.py`).

## Caveats

- Squad values are CURRENT (a mild, documented look-ahead when grading past matches).
- Knockout draws resolved as a coin-flip for extra-time/penalties; exact official R32
  slotting randomized within the real constraints, not hard-coded.
