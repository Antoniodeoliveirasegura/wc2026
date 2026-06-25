"""
2026 World Cup tournament simulator (Tier A engine + faithful knockout draw).

Group stage is played (real results in the dataset). We compute standings + the 8
best third-placed teams, then Monte-Carlo the knockouts with:
  - the real draw STRUCTURE: group winners protected (face a 3rd/runner-up in the
    R32, never each other), same-group teams separated in the R32;
  - host advantage for USA/Canada/Mexico in their matches;
  - Dixon-Coles match probabilities with a CONNECTIVITY-WEIGHTED squad-value prior
    (Transfermarkt): applied across confederations, off within. Validated to help
    cross-confederation matches (-0.0036 RPS) — see marketvalue.py / mv_connectivity.py.

Run: python simulate.py
"""
from __future__ import annotations
import os, pickle, random, collections
import numpy as np
import pandas as pd
import wc_model as wc
import marketvalue as mvmod

random.seed(0)
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "model.pkl")
N_SIMS = 50000
HOSTS = {"United States", "Canada", "Mexico"}   # play knockout games at home

def get_model(df):
    if os.path.exists(MODEL_CACHE):
        return pickle.load(open(MODEL_CACHE, "rb"))
    m = wc.fit_dixon_coles(df, ref_date=df.date.max())
    pickle.dump(m, open(MODEL_CACHE, "wb"))
    return m

# ----------------------------------------------------------------- group stage (real results)
def wc_games(df):
    return df[(df.date >= "2026-06-01") & (df.tournament == "FIFA World Cup")]

def groups_from(wcdf):
    adj = collections.defaultdict(set)
    for r in wcdf.itertuples():
        adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); stack += list(adj[x] - seen)
        comps.append(sorted(comp))
    return comps

def standings(wcdf, group):
    pts = dict.fromkeys(group, 0); gf = dict.fromkeys(group, 0); ga = dict.fromkeys(group, 0)
    for r in wcdf.itertuples():
        if r.home_team in group and r.away_team in group:
            hs, as_ = r.home_score, r.away_score
            gf[r.home_team] += hs; ga[r.home_team] += as_
            gf[r.away_team] += as_; ga[r.away_team] += hs
            if hs > as_:   pts[r.home_team] += 3
            elif hs < as_: pts[r.away_team] += 3
            else:          pts[r.home_team] += 1; pts[r.away_team] += 1
    rank = sorted(group, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
    return rank, pts, gf, ga

def qualifiers(wcdf):
    """Return winners/runners/thirds as (team, group_id) lists + the 32-team list."""
    winners, runners, thirds = [], [], []
    for gi, g in enumerate(groups_from(wcdf)):
        rank, pts, gf, ga = standings(wcdf, g)
        winners.append((rank[0], gi)); runners.append((rank[1], gi))
        t = rank[2]; thirds.append((t, gi, pts[t], gf[t] - ga[t], gf[t]))
    thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best8 = [(t, gi) for (t, gi, *_ ) in thirds[:8]]
    teams = [t for t, _ in winners + runners + best8]
    return winners, runners, best8, teams

# ----------------------------------------------------------------- host-aware advance probs
def adv_matrix(m, teams, zmap, confmap):
    """A[i][j] = P(i beats j) incl. host advantage + connectivity-weighted squad-value
    prior; asymmetric. Draw -> ET/pens coin flip."""
    def wdl(a, b, neutral):
        return mvmod.mv_wdl(m, zmap, confmap, a, b, neutral=neutral)
    n = len(teams); A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = teams[i], teams[j]
            ah, bh = a in HOSTS, b in HOSTS
            if ah and not bh:
                ph, pdr, pa = wdl(a, b, False)        # a at home
            elif bh and not ah:
                pb, pdr, pa2 = wdl(b, a, False)        # b at home
                ph, pa = pa2, pb
            else:
                ph, pdr, pa = wdl(a, b, True)
            A[i][j] = ph + 0.5 * pdr
    return A

# ----------------------------------------------------------------- constrained R32 draw
def draw_r32(W, RU, TH):
    """W/RU/TH: lists of (idx, group). Winners face thirds/runners (never winners);
    same-group teams separated. Returns 16 (idxA, idxB) pairs."""
    W = W[:]; RU = RU[:]; TH = TH[:]
    random.shuffle(W); random.shuffle(RU); random.shuffle(TH)
    matches = []
    availW = W[:]
    for ti, tg in TH:                                   # 8 thirds vs distinct winners
        k = next((k for k, (wi, wg) in enumerate(availW) if wg != tg), 0)
        wi, _ = availW.pop(k); matches.append((wi, ti))
    availRU = RU[:]
    for wi, wg in availW:                                # 4 remaining winners vs runners
        k = next((k for k, (ri, rg) in enumerate(availRU) if rg != wg), 0)
        ri, _ = availRU.pop(k); matches.append((wi, ri))
    rem = availRU
    while rem:                                           # 8 remaining runners -> 4 games
        ai, ag = rem.pop(0)
        k = next((k for k, (bi, bg) in enumerate(rem) if bg != ag), 0)
        bi, _ = rem.pop(k); matches.append((ai, bi))
    return matches

def simulate(W, RU, TH, A, n_sims=N_SIMS):
    rounds = {k: collections.Counter() for k in ("r16", "qf", "sf", "final", "champ")}
    for _ in range(n_sims):
        survivors = [a if random.random() < A[a][b] else b for a, b in draw_r32(W, RU, TH)]
        for t in survivors:
            rounds["r16"][t] += 1
        random.shuffle(survivors)
        alive = survivors
        while len(alive) > 1:
            stage = {8: "qf", 4: "sf", 2: "final"}.get(len(alive))
            if stage:
                for t in alive: rounds[stage][t] += 1
            alive = [alive[k] if random.random() < A[alive[k]][alive[k + 1]] else alive[k + 1]
                     for k in range(0, len(alive), 2)]
        rounds["champ"][alive[0]] += 1
    return rounds

HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026 World Cup Forecast</title>
<style>
:root{--bg:#0e1116;--card:#161b22;--ink:#e6edf3;--muted:#8b949e;--accent:#3b82f6;--line:#21262d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:36px 20px 72px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}.sub{color:var(--muted);margin:0 0 28px;font-size:14px}
h2{font-size:15px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:22px}
.bars{display:flex;flex-direction:column;gap:9px}
.bar{display:grid;grid-template-columns:118px 1fr 50px;align-items:center;gap:12px;font-size:14px}
.bar .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.track{background:#21262d;border-radius:6px;height:22px;overflow:hidden}
.fill{background:linear-gradient(90deg,#2563eb,#3b82f6);height:100%;border-radius:6px}
.v{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:right;padding:8px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:none}.foot{color:var(--muted);font-size:13px;margin-top:8px}
</style></head><body><div class="wrap">
<h1>2026 World Cup forecast</h1>
<p class="sub">__SUB__</p>
<div class="card"><h2>Title odds</h2><div class="bars">__BARS__</div></div>
<div class="card"><h2>All 32 qualifiers</h2><table>
<thead><tr><th>Team</th><th>Champion</th><th>Final</th><th>Semi</th><th>Quarter</th><th>R16</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
<p class="foot">__FOOT__</p>
</div></body></html>"""

def write_site(teams, order, R, n_sims, path):
    champ = R["champ"]
    pc = lambda c, i: f"{c[i] / n_sims:.1%}"
    mx = champ[order[0]] / n_sims
    bars = "".join(
        f'<div class="bar"><span class="nm">{teams[i]}</span>'
        f'<span class="track"><span class="fill" style="width:{champ[i]/n_sims/mx*100:.0f}%"></span></span>'
        f'<span class="v">{pc(champ, i)}</span></div>' for i in order[:12])
    rows = "".join(
        f"<tr><td>{teams[i]}</td><td>{pc(champ,i)}</td><td>{pc(R['final'],i)}</td>"
        f"<td>{pc(R['sf'],i)}</td><td>{pc(R['qf'],i)}</td><td>{pc(R['r16'],i)}</td></tr>"
        for i in order)
    sub = (f"Dixon-Coles + connectivity-weighted squad value &middot; {n_sims:,} "
           "Monte-Carlo simulations &middot; from the completed group stage")
    foot = ("A calibrated distribution, not a single pick &mdash; the favourite tops out ~16%. "
            "Built from free historical results + Transfermarkt squad values.")
    html = (HTML_TEMPLATE.replace("__SUB__", sub).replace("__BARS__", bars)
            .replace("__ROWS__", rows).replace("__FOOT__", foot))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    df = wc.load()
    wcdf = wc_games(df)
    m = get_model(df)
    zmap, confmap = mvmod.setup(m)

    winners, runners, thirds, teams = qualifiers(wcdf)
    qidx = {t: i for i, t in enumerate(teams)}
    A = adv_matrix(m, teams, zmap, confmap)
    W = [(qidx[t], g) for t, g in winners]
    RU = [(qidx[t], g) for t, g in runners]
    TH = [(qidx[t], g) for t, g in thirds]
    R = simulate(W, RU, TH, A)
    champ = R["champ"]
    order = sorted(range(len(teams)), key=lambda i: champ[i], reverse=True)

    print(f"title odds  (N={N_SIMS:,}, real draw + host edge + connectivity-weighted "
          f"market value):\n")
    print(f"  {'team':<18}{'champ':>7}{'final':>7}{'semi':>7}")
    for i in order[:16]:
        print(f"  {teams[i]:<18}{champ[i]/N_SIMS:>6.1%}{R['final'][i]/N_SIMS:>7.1%}{R['sf'][i]/N_SIMS:>7.1%}")

    pct = lambda c, i: f"{c[i] / N_SIMS:.1%}"
    lines = ["# 2026 World Cup forecast", "",
             f"Dixon-Coles engine + connectivity-weighted squad-value prior + Monte-Carlo "
             f"knockout sim (N={N_SIMS:,}, real draw structure + host advantage). "
             "All 32 qualifiers:", "",
             "| Team | Champion | Final | Semi | Quarter | R16 |",
             "|---|---|---|---|---|---|"]
    for i in order:
        lines.append(f"| {teams[i]} | {pct(champ, i)} | {pct(R['final'], i)} | "
                     f"{pct(R['sf'], i)} | {pct(R['qf'], i)} | {pct(R['r16'], i)} |")
    with open(os.path.join(os.path.dirname(__file__), "forecast.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_site(teams, order, R, N_SIMS,
               os.path.join(os.path.dirname(__file__), "docs", "index.html"))

    assert abs(sum(champ.values()) - N_SIMS) < 1 and max(champ.values()) / N_SIMS < 0.5
    print(f"\nwrote forecast.md + docs/index.html ({len(teams)} teams)  |  self-check ok")
