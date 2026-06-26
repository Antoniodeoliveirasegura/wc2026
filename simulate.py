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
import altitude as altmod

random.seed(0)
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "model.pkl")
SHOOT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
SHOOT_FILE = os.path.join(os.path.dirname(__file__), "shootouts.csv")
N_SIMS = 20000
HOSTS = {"United States", "Canada", "Mexico"}
MAXG = 8
WC_START = pd.Timestamp("2026-06-11")
PRE_WC_CACHE = os.path.join(os.path.dirname(__file__), "pre_wc_model.pkl")

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

def fixture_cities(df):
    """{frozenset(team pair): venue city} for every 2026 WC fixture (played + scheduled).
    Read from the raw results file because wc.load() drops not-yet-played rows."""
    raw = pd.read_csv(wc.DATA)
    f = raw[(raw.date >= "2026-06-01") & (raw.tournament == "FIFA World Cup")]
    return {frozenset((r.home_team, r.away_team)): r.city for r in f.itertuples(index=False)}

def score_dist(m, zmap, confmap, a, b, city=""):
    # altitude penalty applies only at scheduled high-altitude venues (Mexico City /
    # Guadalajara); a no-op elsewhere. Knockout venues aren't assigned yet -> Option B.
    adj = altmod.alt_adjust(mvmod.mv_adjust(m, zmap, confmap, a, b), a, b, city)
    M = wc.score_matrix(adj, a, b, neutral=True, maxg=MAXG)
    cells = [(i, j) for i in range(MAXG + 1) for j in range(MAXG + 1)]
    cum = np.cumsum(M.flatten()); cum = cum / cum[-1]
    return cells, cum

def adv_matrix(m, teams, zmap, confmap, elo_predict, city=""):
    """P(a beats b) for every ordered pair, used by the knockout sims. W/D/L is the
    DC+squad-value model geometrically blended with Elo (wc.geo_blend; validated
    -0.0035 RPS). Host edge applies in non-neutral framings exactly as before.
    city != "" applies the venue altitude penalty (for the Azteca knockout slots)."""
    def blended(home, away, neutral):
        base = altmod.alt_adjust(mvmod.mv_adjust(m, zmap, confmap, home, away), home, away, city)
        dc = np.array(wc.wdl(base, home, away, neutral=neutral))
        return wc.geo_blend(dc, elo_predict(home, away, neutral=neutral))   # [H,D,A]
    n = len(teams); A = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ta, tb = teams[a], teams[b]
            ah, bh = ta in HOSTS, tb in HOSTS
            if bh and not ah:                                   # tb hosts -> frame in tb's home
                H, D, Aw = blended(tb, ta, neutral=False)
                A[a][b] = Aw + 0.5 * D                          # ta is the away side here
            else:                                               # ta hosts (non-neutral), or neutral
                H, D, Aw = blended(ta, tb, neutral=not (ah and not bh))
                A[a][b] = H + 0.5 * D
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

def play_out(matches, A, rounds, A_alt=None, azteca_idx=None):
    """A_alt/azteca_idx (Option B): the R32 Match 79 slot and its R16 leg (Match 92) are
    played at Estadio Azteca, so they use the altitude-adjusted advantage A_alt; everything
    else uses A. azteca_idx is the index in `matches` of the Group-A-winner slot."""
    survivors = []; azteca_alive = None
    for k, (a, b) in enumerate(matches):
        adv = A_alt if (A_alt is not None and k == azteca_idx) else A
        w = a if random.random() < adv[a][b] else b
        survivors.append(w)
        if k == azteca_idx:
            azteca_alive = w                       # plays its R16 at Azteca too (Match 92)
    for t in survivors:
        rounds["r16"][t] += 1
    random.shuffle(survivors); alive = survivors
    r16_round = True
    while len(alive) > 1:
        stage = {8: "qf", 4: "sf", 2: "final"}.get(len(alive))
        if stage:
            for t in alive:
                rounds[stage][t] += 1
        nxt = []
        for k in range(0, len(alive), 2):
            x, y = alive[k], alive[k + 1]
            adv = A_alt if (r16_round and A_alt is not None and azteca_alive in (x, y)) else A
            nxt.append(x if random.random() < adv[x][y] else y)
        alive = nxt; r16_round = False             # rounds after R16 are all sea-level US venues
    rounds["champ"][alive[0]] += 1

def simulate_from_groups(groups_idx, base, rem_dists, A, n_sims=N_SIMS,
                         A_azteca=None, azteca_group=None):
    """Group stage in progress: sample remaining group games -> qualifiers -> knockouts.
    azteca_group (Option B): index of Mexico's group, whose winner plays R32+R16 at Azteca."""
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
        matches = draw_r32(W, RU, best8)
        azteca_idx = None
        if azteca_group is not None and A_azteca is not None:
            aw = next((t for t, gi in W if gi == azteca_group), None)   # Group A winner this sim
            azteca_idx = next((k for k, (a, b) in enumerate(matches) if aw in (a, b)), None)
        play_out(matches, A, rounds, A_alt=A_azteca, azteca_idx=azteca_idx)
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

FLAG = {
    "Spain": "es", "Argentina": "ar", "England": "gb-eng", "Brazil": "br", "France": "fr",
    "Portugal": "pt", "Germany": "de", "Netherlands": "nl", "Belgium": "be", "Colombia": "co",
    "Morocco": "ma", "Switzerland": "ch", "Croatia": "hr", "Uruguay": "uy", "Norway": "no",
    "Japan": "jp", "Mexico": "mx", "United States": "us", "Canada": "ca", "Australia": "au",
    "Ecuador": "ec", "Senegal": "sn", "Iran": "ir", "South Korea": "kr", "Egypt": "eg",
    "Ghana": "gh", "Ivory Coast": "ci", "Algeria": "dz", "Tunisia": "tn", "Cape Verde": "cv",
    "DR Congo": "cd", "Curaçao": "cw", "Haiti": "ht", "Paraguay": "py", "Qatar": "qa",
    "Saudi Arabia": "sa", "Uzbekistan": "uz", "Jordan": "jo", "Iraq": "iq", "Panama": "pa",
    "Scotland": "gb-sct", "New Zealand": "nz", "South Africa": "za", "Czech Republic": "cz",
    "Turkey": "tr", "Austria": "at", "Sweden": "se", "Bosnia and Herzegovina": "ba",
}

def flag(team):
    iso = FLAG.get(team)
    return f'<img src="https://flagcdn.com/24x18/{iso}.png" width="20" height="15" alt="">' if iso else ""

SHORT = {"Bosnia and Herzegovina": "Bosnia", "United States": "USA", "Czech Republic": "Czechia",
         "South Korea": "S. Korea", "South Africa": "S. Africa", "Saudi Arabia": "Saudi",
         "New Zealand": "N. Zealand", "Ivory Coast": "Ivory C."}

def short(t):
    return SHORT.get(t, t)

def predict_score(m, zmap, confmap, a, b):
    """Most-likely exact scoreline (argmax of the squad-value-adjusted DC score grid)."""
    M = wc.score_matrix(mvmod.mv_adjust(m, zmap, confmap, a, b), a, b, neutral=True, maxg=MAXG)
    i, j = np.unravel_index(int(np.argmax(M)), M.shape)
    return int(i), int(j)

def pred_key(a, b):
    return "|".join(sorted((a, b)))

def update_and_grade(m, zmap, confmap, rem, gdf, groups):
    """Lock a predicted score for each UPCOMING game (never overwritten -> only ever
    before kickoff). Organise every group game BY GROUP: played games show the result
    (graded green/red where we had a locked prediction), upcoming games show the
    prediction. Returns [(group_label, rows)]."""
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

    def make_row(a, b):
        k = pred_key(a, b); p = preds.get(k)
        if k in played:
            h, aw, hs, as_ = played[k]; act = {h: hs, aw: as_}
            if p:
                ok = (p["pred"].get(h) == hs and p["pred"].get(aw) == as_)
                return {"a": a, "b": b, "pa": p["pred"][a], "pb": p["pred"][b],
                        "status": "correct" if ok else "wrong", "act_a": act[a], "act_b": act[b]}
            return {"a": a, "b": b, "pa": act[a], "pb": act[b], "status": "result"}
        if p:
            return {"a": a, "b": b, "pa": p["pred"][a], "pb": p["pred"][b], "status": "upcoming"}
        return None

    grouped = []
    for gi, g in enumerate(sorted(groups, key=lambda x: x[0])):
        rows = [r for i in range(len(g)) for j in range(i + 1, len(g))
                if (r := make_row(g[i], g[j]))]
        rows.sort(key=lambda r: r["status"] == "upcoming")      # results/graded first
        grouped.append(("Group " + chr(65 + gi), rows))
    return grouped

def render_preds(grouped):
    if not grouped:
        return "<p style='color:var(--muted);font-size:13px;margin:0'>No games yet.</p>"
    out = []
    for gi, (label, rows) in enumerate(grouped):
        out.append(f'<div class="grp{" first" if gi == 0 else ""}">{label}</div>')
        for r in rows:
            match = (f'<span class="mt">{flag(r["a"])}{short(r["a"])}</span> '
                     f'<span class="sc">{r["pa"]}&ndash;{r["pb"]}</span> '
                     f'<span class="mt">{flag(r["b"])}{short(r["b"])}</span>')
            st = r["status"]
            if st == "upcoming":
                badge, cls = '<span class="badge b-pend">upcoming</span>', "pred"
            elif st == "result":
                badge, cls = '<span class="badge b-done">played</span>', "pred dim"
            elif st == "correct":
                badge, cls = '<span class="badge b-ok">&#10003;</span>', "pred"
            else:
                badge, cls = f'<span class="badge b-no">{r["act_a"]}&ndash;{r["act_b"]}</span>', "pred"
            out.append(f'<div class="{cls}"><span>{match}</span>{badge}</div>')
    return '<div class="preds">' + "".join(out) + "</div>"

# ----------------------------------------------------------------- model hindcast (backtest)
def pre_wc_model(df):
    """Model fit ONLY on pre-tournament data -> honest backtest of played games (no leak)."""
    if os.path.exists(PRE_WC_CACHE):
        return pickle.load(open(PRE_WC_CACHE, "rb"))
    pm = wc.fit_dixon_coles(df[df.date < WC_START], ref_date=WC_START)
    pickle.dump(pm, open(PRE_WC_CACHE, "wb"))
    return pm

def hindcast(df, wcdf):
    """How a pre-tournament model would have called each played WC game (argmax score)."""
    pm = pre_wc_model(df)
    zmap, confmap = mvmod.setup(pm)
    rows, exact, ok = [], 0, 0
    for r in wcdf.sort_values("date").itertuples(index=False):
        h, a = r.home_team, r.away_team
        if h not in pm["idx"] or a not in pm["idx"]:
            continue
        M = wc.score_matrix(mvmod.mv_adjust(pm, zmap, confmap, h, a), h, a,
                            neutral=bool(r.neutral), maxg=MAXG)
        pi, pj = (int(x) for x in np.unravel_index(int(np.argmax(M)), M.shape))
        hs, as_ = int(r.home_score), int(r.away_score)
        is_exact = (pi == hs and pj == as_)
        res, pres = (hs > as_) - (hs < as_), (pi > pj) - (pi < pj)
        exact += is_exact; ok += (res == pres)
        rows.append({"home": h, "away": a, "ph": pi, "pa": pj, "ah": hs, "aa": as_,
                     "exact": is_exact, "ok": res == pres})
    return rows, exact, ok, len(rows)

def render_hindcast(data):
    rows, exact, ok, n = data
    if not n:
        return "<p style='color:var(--muted);font-size:13px'>No games played yet.</p>"
    cards = []
    for r in rows:
        match = (f'{flag(r["home"])}{short(r["home"])} '
                 f'<b class="sc">{r["ph"]}&ndash;{r["pa"]}</b> '
                 f'{flag(r["away"])}{short(r["away"])}')
        if r["exact"]:
            badge = '<span class="badge b-ok">&#10003; exact</span>'
        elif r["ok"]:
            badge = f'<span class="badge b-amber">{r["ah"]}&ndash;{r["aa"]}</span>'
        else:
            badge = f'<span class="badge b-no">{r["ah"]}&ndash;{r["aa"]}</span>'
        cards.append(f'<div class="hc"><span>{match}</span>{badge}</div>')
    summ = f'{exact}/{n} exact scores &middot; {ok}/{n} outcomes right ({ok / n:.0%})'
    return f'<div class="hcsum">{summ}</div><div class="hcgrid">' + "".join(cards) + "</div>"

HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2026 World Cup Forecast</title>
<style>
:root{--bg:#0e1116;--card:#161b22;--ink:#e6edf3;--muted:#8b949e;--accent:#3b82f6;--line:#21262d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:36px 20px 72px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:22px;align-items:start}
.main{display:flex;flex-direction:column;gap:22px;min-width:0}
.aside{min-width:0}
@media(max-width:780px){.layout{grid-template-columns:1fr}}
@media(max-width:640px){
.wrap{padding:22px 13px 56px}
h1{font-size:23px}.sub{font-size:13px;margin-bottom:8px}
.card{padding:16px 13px}.card h2{font-size:13px}
.bar{grid-template-columns:90px 1fr 40px;gap:8px;font-size:13px}
.bar .nm img{margin-right:5px}
table{table-layout:fixed}th:first-child,td:first-child{width:44%;overflow-wrap:break-word}
th,td{padding:7px 4px;font-size:12.5px}.col-opt{display:none}
.hcgrid{grid-template-columns:1fr}.hc{font-size:12px;flex-wrap:wrap}
.pred{font-size:12px}
}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}.sub{color:var(--muted);margin:0 0 6px;font-size:14px}
.phase{display:inline-block;background:#1f2937;color:#9fc5ff;font-size:12px;font-weight:600;
padding:4px 11px;border-radius:999px;margin:0 0 26px}
h2{font-size:15px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:22px}
.bars{display:flex;flex-direction:column;gap:9px}
.bar{display:grid;grid-template-columns:118px 1fr 50px;align-items:center;gap:12px;font-size:14px}
.bar .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.track{background:#21262d;border-radius:6px;height:22px;overflow:hidden}
.fill{display:block;background:linear-gradient(90deg,#2563eb,#60a5fa);height:100%;border-radius:6px}
.nm img,td img,.mt img{border-radius:2px;vertical-align:-2px;margin-right:7px;box-shadow:0 0 0 .5px rgba(255,255,255,.12)}
tbody tr{transition:background .12s}tbody tr:hover{background:rgba(255,255,255,.04)}
.card{transition:border-color .15s}
.v{text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:14px}.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{text-align:right;padding:8px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:none}.foot{color:var(--muted);font-size:13px;margin-top:8px}
.hint{font-size:11px;color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0}
.preds{display:flex;flex-direction:column;gap:6px}
.pred{display:flex;justify-content:space-between;align-items:center;font-size:12.5px;gap:8px;flex-wrap:wrap}
.pred.dim{opacity:.55}.pred .sc{font-variant-numeric:tabular-nums;font-weight:600;margin:0 2px}
.grp{font-size:11px;font-weight:600;color:#60a5fa;text-transform:uppercase;letter-spacing:.04em;margin:14px 0 4px;padding-top:10px;border-top:1px solid var(--line)}
.grp.first{border-top:none;padding-top:0;margin-top:2px}
.badge{font-size:11.5px;padding:3px 8px;border-radius:999px;font-variant-numeric:tabular-nums;white-space:nowrap}
.b-pend{background:#21262d;color:#8b949e}.b-ok{background:#10331f;color:#3fb950}.b-no{background:#3a1d1d;color:#f85149}.b-done{background:#1c2128;color:#6e7681}.b-amber{background:#3a2e14;color:#e3b341}
.hcsum{font-size:13px;color:var(--muted);margin-bottom:14px}
.hcgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:8px}
.hc{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:12.5px;background:#0e1116;border:1px solid var(--line);border-radius:8px;padding:7px 11px}
.hc .sc{font-weight:600;margin:0 3px}
</style></head><body><div class="wrap">
<h1>2026 World Cup forecast</h1>
<p class="sub">__SUB__</p>
<div class="phase">__PHASE__</div>
<div class="layout">
<div class="main">
<div class="card"><h2>Title odds</h2><div class="bars">__BARS__</div></div>
<div class="card"><h2>All teams — chance of reaching each stage</h2><div class="tbl"><table>
<thead><tr><th>Team</th><th>Qualify</th><th>Champion</th><th class="col-opt">Final</th><th class="col-opt">Semi</th><th>R16</th></tr></thead>
<tbody>__ROWS__</tbody></table></div></div>
</div>
<aside class="aside">
<div class="card"><h2>Score predictions <span class="hint">by group &middot; green = exact</span></h2>__PREDS__</div>
</aside>
</div>
<div class="card"><h2>Model hindcast <span class="hint">how it would have called games already played &middot; backtest, no hindsight</span></h2>__HINDCAST__</div>
<p class="foot">__FOOT__</p>
</div></body></html>"""

def write_site(teams, order, R, n_sims, phase, preds, hc, path):
    champ = R["champ"]
    pc = lambda c, i: f"{c[i] / n_sims:.1%}" if c[i] else "—"
    mx = max(1, champ[order[0]])
    bars = "".join(
        f'<div class="bar"><span class="nm">{flag(teams[i])}{teams[i]}</span>'
        f'<span class="track"><span class="fill" style="width:{champ[i]/mx*100:.0f}%"></span></span>'
        f'<span class="v">{pc(champ, i)}</span></div>' for i in order[:12])
    rows = "".join(
        f"<tr><td>{flag(teams[i])}{teams[i]}</td><td>{pc(R['qualify'],i)}</td><td>{pc(champ,i)}</td>"
        f"<td class='col-opt'>{pc(R['final'],i)}</td><td class='col-opt'>{pc(R['sf'],i)}</td><td>{pc(R['r16'],i)}</td></tr>"
        for i in order)
    sub = ("Dixon-Coles + connectivity-weighted squad value, blended with Elo &middot; simulates "
           f"the rest of the tournament &middot; {n_sims:,} Monte-Carlo runs")
    foot = ("A calibrated distribution, not a single pick. Updates automatically as results "
            "come in. Built from free historical results + Transfermarkt squad values.")
    html = (HTML_TEMPLATE.replace("__SUB__", sub).replace("__PHASE__", phase)
            .replace("__BARS__", bars).replace("__ROWS__", rows).replace("__FOOT__", foot)
            .replace("__PREDS__", render_preds(preds))
            .replace("__HINDCAST__", render_hindcast(hc)))
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
    _, elo_predict = wc.fit_elo_wdl(df)
    A = adv_matrix(m, teams, zmap, confmap, elo_predict)
    rem = remaining_group_games(groups, gdf)

    # Option B: the R32 (Match 79) and R16 (Match 92) for Mexico's group winner are at Estadio
    # Azteca (2240m). Build an altitude-adjusted advantage matrix + locate Mexico's group.
    # WC_NO_ALT_KO disables it (for the A/B accuracy check).
    A_azteca = None if os.environ.get("WC_NO_ALT_KO") else \
        adv_matrix(m, teams, zmap, confmap, elo_predict, city="Mexico City")
    azteca_group = next((i for i, g in enumerate(groups) if "Mexico" in g), None) \
        if A_azteca is not None else None

    if len(kdf) == 0:                                    # group stage in progress / just done
        base = base_stats(gdf, tidx)
        fcity = fixture_cities(df)
        rem_dists = [(tidx[a], tidx[b], *score_dist(m, zmap, confmap, a, b,
                      fcity.get(frozenset((a, b)), ""))) for a, b in rem]
        R = simulate_from_groups([[tidx[t] for t in g] for g in groups], base, rem_dists, A,
                                 A_azteca=A_azteca, azteca_group=azteca_group)
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
    preds = update_and_grade(m, zmap, confmap, rem, gdf, groups)
    hc = hindcast(df, wcdf)
    write_site(teams, order, R, N_SIMS, phase, preds, hc,
               os.path.join(os.path.dirname(__file__), "docs", "index.html"))

    assert abs(sum(R["champ"].values()) - N_SIMS) < 1
    print(f"\nwrote forecast.md + docs/index.html | {phase}")
