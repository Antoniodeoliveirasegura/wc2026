# wc2026 — World Cup 2026 forecast

A from-scratch statistical model for the 2026 FIFA World Cup: match Win/Draw/Loss
probabilities from a Dixon-Coles scoreline model, plus a Monte-Carlo tournament
simulator that runs the knockouts from the (already-played) group stage.

## Results

Out-of-sample accuracy on competitive internationals: **RPS ≈ 0.171** (vs a
0.229 base-rate baseline) — bookmaker-comparable.

Current title odds (50k sims, real draw structure + host advantage):

| Team | Champion | Final | Semi |
|---|---|---|---|
| Spain | 16.0% | 25.9% | 40.1% |
| Argentina | 14.1% | 23.3% | 37.3% |
| England | 11.5% | 20.3% | 34.0% |
| Brazil | 11.2% | 20.0% | 33.8% |
| France | 9.5% | 17.6% | 31.0% |
| Portugal | 7.8% | 14.9% | 26.7% |
| Germany | 5.8% | 12.6% | 25.0% |
| Netherlands | 4.8% | 10.7% | 22.0% |

> No model "calls the winner" — the favorite tops out ~16%. The deliverable is a
> calibrated distribution, not a single pick.

## How it works

- **Dixon-Coles** (`wc_model.py`): bivariate-Poisson scoreline model fit by MLE on
  ~20 years of international results, time-decayed (3-yr half-life, tuned) with
  friendlies down-weighted. Gives proper W/D/L probabilities.
- **Elo** (`wc_model.py`): independent dynamic rating, used as a cross-check.
- **Tournament sim** (`simulate.py`): computes real group standings + the 8 best
  third-placed teams from the played 2026 group stage, then Monte-Carlos the
  knockouts with the real draw structure (group winners protected, same-group
  separation) and host advantage for USA/Canada/Mexico.
- **B-lite experiment** (`blite.py`): a gradient-boosted model on DC + form + Elo +
  rest + stage features. Kept as a **documented negative result** — under proper
  time-CV it matched plain DC (Δ −0.0001): cheap features add nothing over a
  regularly-refit Dixon-Coles.

## Setup

    pip install -r requirements.txt
    python simulate.py      # tournament forecast (auto-downloads data on first run)
    python wc_model.py      # engine + out-of-sample validation
    python blite.py         # the feature/GBM experiment (time-CV)

Data (`results.csv`, ~3.7 MB) auto-downloads from the public
`martj42/international_results` dataset on first run; it is gitignored.

## Honest ceiling

Match-level accuracy is tapped: knob-tuning (≤0.003 RPS) and cheap features (≈0)
are both exhausted. Beyond a tuned Dixon-Coles, the only levers are bookmaker odds
(copying the market, not predicting) or real lineup/injury data (expensive, and
unsourceable for 2026). The tournament-sim fidelity was the last honest gain.

## Caveats

- Knockout draws resolved as a coin-flip for extra-time/penalties.
- Exact official R32 slotting is randomized within the real constraints, not
  hard-coded.
- Market-value tilt is a small hardcoded squad-value adjustment (`MV_WEIGHT`).
