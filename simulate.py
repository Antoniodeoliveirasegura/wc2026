"""
2026 World Cup forecaster — simulates the REMAINDER of the tournament from the
current live state in the data, so it stays correct as matches are played:
  * group stage in progress -> sample the remaining group games, then knockouts
  * knockouts in progress    -> drop eliminated teams, simulate what's left

Match model: Dixon-Coles + connectivity-weighted squad-value prior + host edge.
Outputs forecast.md and docs/index.html (the live dashboard).

Run: python simulate.py
"""
from __future__ import annotations
import os, pickle, random, bisect, collections, urllib.request, json, datetime
import numpy as np
import pandas as pd
import wc_model as wc
import marketvalue as mvmod

random.seed(0)
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "model.pkl")
SHOOT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
SHOOT_FILE = os.path.join(os.path.dirname(__file__), "shootouts.csv")
N_SIMS = 20000
HOSTS = {"United States", "Canada", "Mexico"}
MAXG = 8

def get_model(df):
    if os.path.exists(MODEL_CACHE):
        return pickle.load(open(MODEL_CACHE, "rb"))
    m = wc.fit_dixon_coles(df, ref_date=df.date.max())
    pickle.dump(m, open(MODEL_CACHE, "wb"))
    return m

def wc_games(df):
    g = df[(df.date >= "2026-06-01") & (df.tournament == "FIFA World Cup")]
    return g.sort_values("date").reset_index(drop=True)

def split_games(wcdf):
    """Group games = each team's first 3 WC matches; the rest are knockouts."""
    cnt = collections.Counter(); grp, ko = [], []
    for i, r in enumerate(wcdf.itertuples(index=False)):
        is_ko = cnt[r.home_team] >= 3 or cnt[r.away_team] >= 3
        (ko if is_ko else grp).append(i)
        cnt[r.home_team] += 1; cnt[r.away_team] += 1
    return wcdf.iloc[grp].reset_index(drop=True), wcdf.iloc[ko].reset_index(drop=True)

def groups_from(gdf):
    adj = collections.defaultdict(set)
    for r in gdf.itertuples():
        adj[r.home_team].add(r.away_team); adj[r.away_team].add(r.home_team)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        st, comp = [t], set()
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x); comp.add(x); st += list(adj[x] - seen)
        comps.append(sorted(comp))
    return comps

def base_stats(gdf, tidx):
    n = len(tidx)
    pts = np.zeros(n, int); gf = np.zeros(n, int); ga = np.zeros(n, int)
    for r in gdf.itertuples(index=False):
        h, a, hs, as_ = tidx[r.home_team], tidx[r.away_team], r.home_score, r.away_score
        gf[h] += hs; ga[h] += as_; gf[a] += as_; ga[a] += hs
        if hs > as_: pts[h] += 3
        elif hs < as_: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    return pts, gf, ga

def remaining_group_games(groups, gdf):
    played = {frozenset((r.home_team, r.away_team)) for r in gdf.itertuples(index=False)}
    return [(g[i], g[j]) for g in groups for i in range(len(g)) for j in range(i + 1, len(g))
            if frozenset((g[i], g[j])) not in played]

def score_dist(m, zmap, confmap, a, b):
    M = wc.score_matrix(mvmod.mv_adjust(m, zmap, confmap, a, b), a, b, neutral=True, maxg=MAXG)
    cells = [(i, j) for i in range(MAXG + 1) for j in range(MAXG + 1)]
    cum = np.cumsum(M.flatten()); cum = cum / cum[-1]
    return cells, cum

def adv_matrix(m, teams, zmap, confmap):
    n = len(teams); A = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ta, tb = teams[a], teams[b]
            ah, bh = ta in HOSTS, tb in HOSTS
            if ah and not bh:
                ph, pdr, pa = mvmod.mv_wdl(m, zmap, confmap, ta, tb, neutral=False)
            elif bh and not ah:
                pb, pdr, pa2 = mvmod.mv_wdl(m, zmap, confmap, tb, ta, neutral=False)
                ph = pa2
            else:
                ph, pdr, pa = mvmod.mv_wdl(m, zmap, confmap, ta, tb, neutral=True)
            A[a][b] = ph + 0.5 * pdr
    return A

def draw_r32(W, RU, TH):
    """W/RU/TH: (idx, group) tuples. Winners face thirds/runners (never winners);
    same-group teams separated. Returns 16 (idxA, idxB) pairs."""
    W = W[:]; RU = RU[:]; TH = TH[:]
    random.shuffle(W); random.shuffle(RU); random.shuffle(TH)
    matches, availW = [], W[:]
    for ti, tg in TH:
        k = next((k for k, (wi, wg) in enumerate(availW) if wg != tg), 0)
        wi, _ = availW.pop(k); matches.append((wi, ti))
    availRU = RU[:]
    for wi, wg in availW:
        k = next((k for k, (ri, rg) in enumerate(availRU) if rg != wg), 0)
        ri, _ = availRU.pop(k); matches.append((wi, ri))
    rem = availRU
    while rem:
        ai, ag = rem.pop(0)
        k = next((k for k, (bi, bg) in enumerate(rem) if bg != ag), 0)
        bi, _ = rem.pop(k); matches.append((ai, bi))
    return matches

def play_out(matches, A, rounds):
    survivors = [a if random.random() < A[a][b] else b for a, b in matches]
    for t in survivors:
        rounds["r16"][t] += 1
    random.shuffle(survivors); alive = survivors
    while len(alive) > 1:
        stage = {8: "qf", 4: "sf", 2: "final"}.get(len(alive))
        if stage:
            for t in alive:
                rounds[stage][t] += 1
        alive = [alive[k] if random.random() < A[alive[k]][alive[k + 1]] else alive[k + 1]
                 for k in range(0, len(alive), 2)]
    rounds["champ"][alive[0]] += 1

def simulate_from_groups(groups_idx, base, rem_dists, A, n_sims=N_SIMS):
    """Group stage in progress: sample remaining group games -> qualifiers -> knockouts."""
    pts0, gf0, ga0 = base
    rounds = {k: collections.Counter() for k in ("qualify", "r16", "qf", "sf", "final", "champ")}
    for _ in range(n_sims):
        pts = pts0.copy(); gf = gf0.copy(); ga = ga0.copy()
        for ai, bi, cells, cum in rem_dists:
            hg, ag = cells[bisect.bisect_left(cum, random.random())]
            gf[ai] += hg; ga[ai] += ag; gf[bi] += ag; ga[bi] += hg
            if hg > ag: pts[ai] += 3
            elif hg < ag: pts[bi] += 3
            else: pts[ai] += 1; pts[bi] += 1
        W, RU, TH = [], [], []
        for gi, gidx in enumerate(groups_idx):
            rank = sorted(gidx, key=lambda ti: (pts[ti], gf[ti] - ga[ti], gf[ti]), reverse=True)
            W.append((rank[0], gi)); RU.append((rank[1], gi))
            t = rank[2]; TH.append((t, gi, pts[t], gf[t] - ga[t], gf[t]))
        TH.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
        best8 = [(t, gi) for t, gi, *_ in TH[:8]]
        for t, _ in W + RU + best8:
            rounds["qualify"][t] += 1
        play_out(draw_r32(W, RU, best8), A, rounds)
    return rounds

def load_shootouts():
    if not os.path.exists(SHOOT_FILE):
        urllib.request.urlretrieve(SHOOT_URL, SHOOT_FILE)
    s = pd.read_csv(SHOOT_FILE)
    return {(str(r.date), frozenset((r.home_team, r.away_team))): r.winner
            for r in s.itertuples(index=False)}

def simulate_knockouts(qual_idx, alive_idx, A, n_sims=N_SIMS):
    """Knockouts in progress (basic): drop eliminated teams, single-elim among the
    survivors. Refine the exact-bracket paths when the knockouts actually begin."""
    rounds = {k: collections.Counter() for k in ("qualify", "r16", "qf", "sf", "final", "champ")}
    for t in qual_idx:
        rounds["qualify"][t] += n_sims
    for _ in range(n_sims):
        cur = list(alive_idx); random.shuffle(cur)
        while len(cur) > 1:
            stage = {16: "r16", 8: "qf", 4: "sf", 2: "final"}.get(len(cur))
            if stage:
                for t in cur:
                    rounds[stage][t] += 1
            nxt = [cur[k] if random.random() < A[cur[k]][cur[k + 1]] else cur[k + 1]
                   for k in range(0, len(cur) - 1, 2)]
            if len(cur) % 2:
                nxt.append(cur[-1])
            cur = nxt
        rounds["champ"][cur[0]] += 1
    return rounds

# ----------------------------------------------------------------- score predictions
PRED_FILE = os.path.join(os.path.dirname(__file__), "predictions.json")

def predict_score(m, zmap, confmap, a, b):
    """Most-likely exact scoreline (argmax of the squad-value-adjusted DC score grid)."""
    M = wc.score_matrix(mvmod.mv_adjust(m, zmap, confmap, a, b), a, b, neutral=True, maxg=MAXG)
    i, j = np.unravel_index(int(np.argmax(M)), M.shape)
    return int(i), int(j)

def pred_key(a, b):
    return "|".join(sorted((a, b)))

def update_and_grade(m, zmap, confmap, rem, gdf):
    """Lock a predicted score for each UPCOMING (not-yet-played) game and never
    overwrite it -> a prediction can only ever be made before kickoff. Grade locked
    predictions against played results. Returns display rows."""
    preds = json.load(open(PRED_FILE, encoding="utf-8")) if os.path.exists(PRED_FILE) else {}
    today = datetime.date.today().isoformat()
    for a, b in rem:                                       # upcoming games only
        k = pred_key(a, b)
        if k not in preds:                                 # lock once, never overwrite
            ga, gb = predict_score(m, zmap, confmap, a, b)
            preds[k] = {"pred": {a: ga, b: gb}, "locked": today}
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=1)
    played = {pred_key(r.home_team, r.away_team):
              (r.home_team, r.away_team, int(r.home_score), int(r.away_score))
              for r in gdf.itertuples(index=False)}
    rows = []
    for k, p in preds.items():
        a, b = list(p["pred"].keys())
        row = {"a": a, "b": b, "pa": p["pred"][a], "pb": p["pred"][b], "played": k in played}
        if row["played"]:
            h, aw, hs, as_ = played[k]
            act = {h: hs, aw: as_}
            row["act_a"], row["act_b"] = act[a], act[b]
            row["correct"] = (p["pred"].get(h) == hs and p["pred"].get(aw) == as_)
        rows.append(row)
    rows.sort(key=lambda r: (r["played"], r["a"]))         # upcoming first
    return rows

def render_preds(preds):
    if not preds:
        return "<p style='color:var(--muted);font-size:13px;margin:0'>No upcoming games to predict yet.</p>"
    out = []
    for r in preds:
        match = (f'<span class="mt">{r["a"]}</span> <span class="sc">{r["pa"]}&ndash;{r["pb"]}</span> '
                 f'<span class="mt">{r["b"]}</span>')
        if not r["played"]:
            badge = '<span class="badge b-pend">upcoming</span>'
        elif r["correct"]:
            badge = '<span class="badge b-ok">&#10003; exact</span>'
        else:
            badge = f'<span class="badge b-no">actual {r["act_a"]}&ndash;{r["act_b"]}</span>'
        out.append(f'<div class="pred"><span>{match}</span>{badge}</div>')
    return '<div class="preds">' + "".join(out) + "</div>"

HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026 World Cup Forecast</title>
<style>
:root{--bg:#0e1116;--card:#161b22;--ink:#e6edf3;--muted:#8b949e;--accent:#3b82f6;--line:#21262d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:36px 20px 72px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}.sub{color:var(--muted);margin:0 0 6px;font-size:14px}
.phase{display:inline-block;background:#1f2937;color:#9fc5ff;font-size:12px;font-weight:600;
padding:4px 11px;border-radius:999px;margin:0 0 26px}
h2{font-size:15px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:22px}
.bars{display:flex;flex-direction:column;gap:9px}
.bar{display:grid;grid-template-columns:118px 1fr 50px;align-items:center;gap:12px;font-size:14px}
.bar .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.track{background:#21262d;border-radius:6px;height:22px;overflow:hidden}
.fill{background:linear-gradient(90deg,#2563eb,#3b82f6);height:100%;border-radius:6px}
.v{text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:right;padding:8px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:none}.foot{color:var(--muted);font-size:13px;margin-top:8px}
.hint{font-size:11px;color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0}
.preds{display:flex;flex-direction:column;gap:8px}
.pred{display:flex;justify-content:space-between;align-items:center;font-size:14px;gap:12px}
.pred .sc{font-variant-numeric:tabular-nums;font-weight:600;margin:0 2px}
.badge{font-size:12px;padding:3px 9px;border-radius:999px;font-variant-numeric:tabular-nums;white-space:nowrap}
.b-pend{background:#21262d;color:#8b949e}.b-ok{background:#10331f;color:#3fb950}.b-no{background:#3a1d1d;color:#f85149}
</style></head><body><div class="wrap">
<h1>2026 World Cup forecast</h1>
<p class="sub">__SUB__</p>
<div class="phase">__PHASE__</div>
<div class="card"><h2>Title odds</h2><div class="bars">__BARS__</div></div>
<div class="card"><h2>Score predictions <span class="hint">locked before kickoff &middot; green = exact</span></h2>__PREDS__</div>
<div class="card"><h2>All teams — chance of reaching each stage</h2><table>
<thead><tr><th>Team</th><th>Qualify</th><th>Champion</th><th>Final</th><th>Semi</th><th>R16</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
<p class="foot">__FOOT__</p>
</div></body></html>"""

def write_site(teams, order, R, n_sims, phase, preds, path):
    champ = R["champ"]
    pc = lambda c, i: f"{c[i] / n_sims:.1%}" if c[i] else "—"
    mx = max(1, champ[order[0]])
    bars = "".join(
        f'<div class="bar"><span class="nm">{teams[i]}</span>'
        f'<span class="track"><span class="fill" style="width:{champ[i]/mx*100:.0f}%"></span></span>'
        f'<span class="v">{pc(champ, i)}</span></div>' for i in order[:12])
    rows = "".join(
        f"<tr><td>{teams[i]}</td><td>{pc(R['qualify'],i)}</td><td>{pc(champ,i)}</td>"
        f"<td>{pc(R['final'],i)}</td><td>{pc(R['sf'],i)}</td><td>{pc(R['r16'],i)}</td></tr>"
        for i in order)
    sub = ("Dixon-Coles + connectivity-weighted squad value &middot; simulates the rest "
           f"of the tournament &middot; {n_sims:,} Monte-Carlo runs")
    foot = ("A calibrated distribution, not a single pick. Updates automatically as results "
            "come in. Built from free historical results + Transfermarkt squad values.")
    html = (HTML_TEMPLATE.replace("__SUB__", sub).replace("__PHASE__", phase)
            .replace("__BARS__", bars).replace("__ROWS__", rows).replace("__FOOT__", foot)
            .replace("__PREDS__", render_preds(preds)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    df = wc.load()
    m = get_model(df)
    zmap, confmap = mvmod.setup(m)
    wcdf = wc_games(df)
    gdf, kdf = split_games(wcdf)
    groups = groups_from(gdf)
    teams = sorted({t for g in groups for t in g})
    tidx = {t: i for i, t in enumerate(teams)}
    A = adv_matrix(m, teams, zmap, confmap)
    rem = remaining_group_games(groups, gdf)

    if len(kdf) == 0:                                    # group stage in progress / just done
        base = base_stats(gdf, tidx)
        rem_dists = [(tidx[a], tidx[b], *score_dist(m, zmap, confmap, a, b)) for a, b in rem]
        R = simulate_from_groups([[tidx[t] for t in g] for g in groups], base, rem_dists, A)
        phase = (f"Group stage - {len(gdf)} of 72 games played, {len(rem)} remaining"
                 if rem else "Group stage complete - knockouts next")
    else:                                                # knockouts under way
        pts, gf, ga = base_stats(gdf, tidx)
        qual = []
        for g in groups:
            rank = sorted(g, key=lambda t: (pts[tidx[t]], gf[tidx[t]] - ga[tidx[t]], gf[tidx[t]]), reverse=True)
            qual += [rank[0], rank[1]]
        thirds = sorted((g[2] for g in [sorted(gg, key=lambda t: (pts[tidx[t]], gf[tidx[t]]-ga[tidx[t]], gf[tidx[t]]), reverse=True) for gg in groups]),
                        key=lambda t: (pts[tidx[t]], gf[tidx[t]] - ga[tidx[t]], gf[tidx[t]]), reverse=True)[:8]
        qual += thirds
        sh = load_shootouts()
        elim = set()
        for r in kdf.itertuples(index=False):
            if r.home_score > r.away_score: elim.add(r.away_team)
            elif r.away_score > r.home_score: elim.add(r.home_team)
            else:
                w = sh.get((str(pd.Timestamp(r.date).date()), frozenset((r.home_team, r.away_team))), r.home_team)
                elim.add(r.away_team if w == r.home_team else r.home_team)
        qidx = [tidx[t] for t in qual]
        alive = [tidx[t] for t in qual if t not in elim]
        R = simulate_knockouts(qidx, alive, A)
        phase = f"Knockouts - {len(kdf)} games played, {len(alive)} teams alive"

    order = sorted(range(len(teams)), key=lambda i: R["champ"][i], reverse=True)
    print(phase)
    print(f"\n  {'team':<18}{'qualify':>8}{'champ':>8}{'final':>8}")
    for i in order[:16]:
        q = R["qualify"][i] / N_SIMS; c = R["champ"][i] / N_SIMS; f_ = R["final"][i] / N_SIMS
        print(f"  {teams[i]:<18}{q:>7.0%}{c:>8.1%}{f_:>8.1%}")

    pct = lambda c, i: f"{c[i] / N_SIMS:.1%}"
    lines = ["# 2026 World Cup forecast", "", f"_{phase}_", "",
             "| Team | Qualify | Champion | Final | Semi | R16 |", "|---|---|---|---|---|---|"]
    for i in order:
        lines.append(f"| {teams[i]} | {pct(R['qualify'],i)} | {pct(R['champ'],i)} | "
                     f"{pct(R['final'],i)} | {pct(R['sf'],i)} | {pct(R['r16'],i)} |")
    with open(os.path.join(os.path.dirname(__file__), "forecast.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    preds = update_and_grade(m, zmap, confmap, rem, gdf)
    write_site(teams, order, R, N_SIMS, phase, preds,
               os.path.join(os.path.dirname(__file__), "docs", "index.html"))

    assert abs(sum(R["champ"].values()) - N_SIMS) < 1
    print(f"\nwrote forecast.md + docs/index.html | {phase}")
