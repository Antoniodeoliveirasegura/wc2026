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
import betting as bet
import clv

random.seed(0)
MODEL_CACHE = os.path.join(os.path.dirname(__file__), "model.pkl")
SHOOT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
SHOOT_FILE = os.path.join(os.path.dirname(__file__), "shootouts.csv")
N_SIMS = 20000
HOSTS = {"United States", "Canada", "Mexico"}
# Knockout home edge is venue-accurate: in the sim, future knockout venues aren't pinned, but in
# practice the US hosts essentially every knockout tie (Canada's R32 is already in the US; Mexico's
# only home knockout slot is the Azteca R32/R16, handled separately via A_azteca). So the base
# advantage matrix grants a home edge to the US only; the Azteca matrix adds Mexico.
KO_HOME = frozenset({"United States"})
MAXG = 8
WC_START = pd.Timestamp("2026-06-11")
PRE_WC_CACHE = os.path.join(os.path.dirname(__file__), "pre_wc_model.pkl")

def get_model(df):
    # Refit when results.csv is newer than the cache, so the DC ratings always reflect the
    # latest results (in-tournament updating: group games shift strength -> sharper knockouts).
    if os.path.exists(MODEL_CACHE) and os.path.getmtime(MODEL_CACHE) >= os.path.getmtime(wc.DATA):
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

def adv_matrix(m, teams, zmap, confmap, rate_predict, city="", home_nations=KO_HOME):
    """P(a beats b) for every ordered pair, used by the knockout sims. W/D/L is the
    DC+squad-value model geometrically blended with pi-ratings (wc.geo_blend; pi_test.py:
    DC x pi beats the old DC x Elo by ~-0.004 RPS on two held-out cutoffs). Host edge
    applies in non-neutral framings exactly as before.
    city != "" applies the venue altitude penalty (for the Azteca knockout slots)."""
    def blended(home, away, neutral):
        base = altmod.alt_adjust(mvmod.mv_adjust(m, zmap, confmap, home, away), home, away, city)
        dc = np.array(wc.wdl(base, home, away, neutral=neutral))
        return wc.geo_blend(dc, rate_predict(home, away, neutral=neutral))   # [H,D,A]
    n = len(teams); A = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ta, tb = teams[a], teams[b]
            ah, bh = ta in home_nations, tb in home_nations
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

ROUND_BY_COUNT = {16: "Round of 32", 8: "Round of 16", 4: "Quarter-finals",
                  2: "Semi-finals", 1: "Final"}

def upcoming_knockout(df):
    """Scheduled-but-unplayed knockout ties from the raw results file (wc.load drops unscored
    rows). A fixture is a knockout once both teams have their 3 group games behind them, so this
    self-extends round by round as the source posts each round's fixtures."""
    played = collections.Counter()
    for r in wc_games(df).itertuples(index=False):
        played[r.home_team] += 1; played[r.away_team] += 1
    raw = pd.read_csv(wc.DATA)
    f = raw[(raw.date >= "2026-06-01") & (raw.tournament == "FIFA World Cup")
            & (raw.home_score.isna() | raw.away_score.isna())]
    out = []
    for r in f.sort_values("date").itertuples(index=False):
        if played.get(r.home_team, 0) >= 3 and played.get(r.away_team, 0) >= 3:
            out.append({"a": r.home_team, "b": r.away_team,
                        "city": getattr(r, "city", "") or "", "date": str(r.date)[:10],
                        "country": getattr(r, "country", "") or ""})
    return out

def predict_knockout(m, zmap, confmap, fixtures):
    """For each upcoming knockout tie: the most-likely 90' scoreline AND who the model backs
    to ADVANCE (regulation + ET + pens via betting.advance_probs) — since someone must win."""
    idx = np.arange(MAXG + 1)
    rows = []
    for fx in fixtures:
        a, b, city, country = fx["a"], fx["b"], fx["city"], fx.get("country", "")
        if a not in m["idx"] or b not in m["idx"]:
            continue
        # host advantage is venue-accurate: a host nation is "home" only when the tie is actually
        # in its country (Canada playing in the US gets no edge). Else neutral.
        host = a if (a in HOSTS and country == a) else b if (b in HOSTS and country == b) else None
        home, away = (host, b if host == a else a) if host else (a, b)
        adj = altmod.alt_adjust(mvmod.mv_adjust(m, zmap, confmap, home, away), home, away, city)
        M = wc.score_matrix(adj, home, away, neutral=host is None, maxg=MAXG)
        pi, pj = (int(x) for x in np.unravel_index(int(np.argmax(M)), M.shape))
        lam = float((M.sum(1) * idx).sum()); mu = float((M.sum(0) * idx).sum())
        adv = bet.advance_probs(M, lam, mu)                  # HOME = home side, AWAY = away side
        if host == b:                                        # M is host-first -> flip back to (a,b)
            pa, pb, adv_a, adv_b = pj, pi, adv["AWAY"], adv["HOME"]
        else:
            pa, pb, adv_a, adv_b = pi, pj, adv["HOME"], adv["AWAY"]
        winner = a if adv_a >= adv_b else b
        rows.append({"a": a, "b": b, "pa": pa, "pb": pb, "date": fx["date"],
                     "winner": winner, "win_p": max(adv_a, adv_b)})
    return rows

ROUND_SEQ = {1: "Round of 32", 2: "Round of 16", 3: "Quarter-finals",
             4: "Semi-finals", 4.5: "Third-place play-off", 5: "Final"}

def update_and_grade_knockout(m, zmap, confmap, df, wcdf):
    """Knockout bracket BY ROUND, the group-scores treatment for ties: PLAYED ties show the
    actual score + who advanced (penalty winner from shootouts.csv), graded against a
    locked pre-match prediction; UPCOMING ties show the predicted score + backed advancer.
    Predictions lock once (never overwritten), so grading is hindsight-free. Returns
    [(round_label, rows)]."""
    preds = json.load(open(PRED_FILE, encoding="utf-8")) if os.path.exists(PRED_FILE) else {}
    today = datetime.date.today().isoformat()
    _, kdf = split_games(wcdf)
    shoot = load_shootouts()
    up_rows = predict_knockout(m, zmap, confmap, upcoming_knockout(df))
    for pr in up_rows:                                     # lock each upcoming tie once
        k = pred_key(pr["a"], pr["b"])
        if k not in preds:
            preds[k] = {"pred": {pr["a"]: pr["pa"], pr["b"]: pr["pb"]},
                        "adv": pr["winner"], "locked": today}
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=1)

    cnt = collections.Counter(); by_round = collections.defaultdict(list); lost = set()
    for r in kdf.sort_values("date").itertuples(index=False):     # played ties, chronological
        h, a, hs, as_ = r.home_team, r.away_team, int(r.home_score), int(r.away_score)
        ri = max(cnt[h], cnt[a]) + 1; cnt[h] += 1; cnt[a] += 1     # per-team KO ordinal -> round
        # Third-place play-off: the only tie contested by two ALREADY-eliminated teams. It shares
        # the final's ordinal (both sides' 5th knockout game), so without this it lands in the
        # "Final" bucket. 4.5 sorts it between the semis and the final.
        if h in lost and a in lost:
            ri -= 0.5
        key = (str(pd.Timestamp(r.date).date()), frozenset((h, a)))
        winner = h if hs > as_ else a if as_ > hs else shoot.get(key, h)
        lost.add(a if winner == h else h)
        row = {"a": h, "b": a, "act_a": hs, "act_b": as_, "winner": winner, "pens": hs == as_}
        p = preds.get(pred_key(h, a))
        if p and "adv" in p:                                       # graded vs the locked call
            exact = p["pred"].get(h) == hs and p["pred"].get(a) == as_
            row["status"] = "correct" if exact else "adv_ok" if p["adv"] == winner else "wrong"
        else:
            row["status"] = "result"                              # no pre-match lock -> ungraded
        by_round[ri].append(row)
    for pr in up_rows:                                            # upcoming ties after
        ri = max(cnt[pr["a"]], cnt[pr["b"]]) + 1; cnt[pr["a"]] += 1; cnt[pr["b"]] += 1
        by_round[ri].append({"a": pr["a"], "b": pr["b"], "pa": pr["pa"], "pb": pr["pb"],
                             "winner": pr["winner"], "win_p": pr["win_p"], "status": "upcoming"})
    # Latest round first: Final, third-place, semis, quarters, R16, R32.
    return [(ROUND_SEQ.get(ri, f"Round {ri}"), by_round[ri])
            for ri in sorted(by_round, reverse=True)]

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
                     f'<span class="sc">{r["pa"]}-{r["pb"]}</span> '
                     f'<span class="mt">{flag(r["b"])}{short(r["b"])}</span>')
            st = r["status"]
            if st == "upcoming":
                badge, cls = '<span class="badge b-pend">upcoming</span>', "pred"
            elif st == "result":
                badge, cls = '<span class="badge b-done">played</span>', "pred dim"
            elif st == "correct":
                badge, cls = '<span class="badge b-ok">&#10003;</span>', "pred"
            else:
                badge, cls = f'<span class="badge b-no">{r["act_a"]}-{r["act_b"]}</span>', "pred"
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
    """How a pre-tournament model would have called each played WC game (argmax score),
    split into GROUP and KNOCKOUT stages. Knockout games that ended level were decided on
    penalties, so their true winner is read from shootouts.csv (a level score there is NOT
    a draw). Each stage also gets a forced-winner view: ignore the draw, lean to the likelier
    winner, scored only on games that had a winner (every knockout does -> someone advances)."""
    pm = pre_wc_model(df)
    zmap, confmap = mvmod.setup(pm)
    _, kdf = split_games(wcdf)
    ko_keys = {(r.date.strftime("%Y-%m-%d"), frozenset((r.home_team, r.away_team)))
               for r in kdf.itertuples(index=False)}
    shoot = load_shootouts()
    blank = lambda: {"rows": [], "exact": 0, "ok": 0, "dec_ok": 0, "dec_n": 0}
    stages = {"group": blank(), "knockout": blank()}
    for r in wcdf.sort_values("date").itertuples(index=False):
        h, a = r.home_team, r.away_team
        if h not in pm["idx"] or a not in pm["idx"]:
            continue
        key = (r.date.strftime("%Y-%m-%d"), frozenset((h, a)))
        is_ko = key in ko_keys
        M = wc.score_matrix(mvmod.mv_adjust(pm, zmap, confmap, h, a), h, a,
                            neutral=bool(r.neutral), maxg=MAXG)
        pi, pj = (int(x) for x in np.unravel_index(int(np.argmax(M)), M.shape))
        idx = np.arange(M.shape[0]); diff = idx[:, None] - idx[None, :]
        pwin_h, pwin_a = float(M[diff > 0].sum()), float(M[diff < 0].sum())
        lean = h if pwin_h >= pwin_a else a                  # the team it'd back if forced to choose
        lean_p = max(pwin_h, pwin_a)
        hs, as_ = int(r.home_score), int(r.away_score)
        winner = h if hs > as_ else a if as_ > hs else (shoot.get(key) if is_ko else None)
        pens = is_ko and hs == as_ and winner is not None
        res = 0 if winner is None else (1 if winner == h else -1)
        pres = (pi > pj) - (pi < pj)
        is_exact = (pi == hs and pj == as_)
        lean_ok = winner is not None and lean == winner
        st = stages["knockout" if is_ko else "group"]
        st["exact"] += is_exact; st["ok"] += (res == pres)
        if winner is not None:
            st["dec_n"] += 1; st["dec_ok"] += lean_ok
        st["rows"].append({"home": h, "away": a, "ph": pi, "pa": pj, "ah": hs, "aa": as_,
                           "exact": is_exact, "ok": res == pres, "lean": lean, "lean_p": lean_p,
                           "lean_ok": lean_ok, "actual_draw": res == 0, "pens": pens,
                           "winner": winner})
    return {k: (v["rows"], v["exact"], v["ok"], len(v["rows"]), v["dec_ok"], v["dec_n"])
            for k, v in stages.items()}

def render_hindcast(data):
    rows, exact, ok, n, dec_ok, dec_n = data
    if not n:
        return "<p style='color:var(--muted);font-size:13px'>No games played yet.</p>"
    cards = []
    for r in rows:
        match = (f'{flag(r["home"])}{short(r["home"])} '
                 f'<b class="sc">{r["ph"]}-{r["pa"]}</b> '
                 f'{flag(r["away"])}{short(r["away"])}')
        # when the predicted scoreline is a draw, show who it'd back if forced to call a winner
        if r["ph"] == r["pa"]:
            cls = "" if r["actual_draw"] else (" lean-ok" if r["lean_ok"] else " lean-no")
            match += (f'<span class="lean{cls}">&rarr; {short(r["lean"])} '
                      f'{r["lean_p"]*100:.0f}%</span>')
        if r.get("pens"):                       # level after ET -> advanced on penalties
            match += f'<span class="lean">pens: {short(r["winner"])}</span>'
        if r["exact"]:
            badge = '<span class="badge b-ok">&#10003; exact</span>'
        elif r["ok"]:
            badge = f'<span class="badge b-amber">{r["ah"]}-{r["aa"]}</span>'
        else:
            badge = f'<span class="badge b-no">{r["ah"]}-{r["aa"]}</span>'
        cards.append(f'<div class="hc"><span>{match}</span>{badge}</div>')
    summ = f'{exact}/{n} exact scores &middot; {ok}/{n} outcomes right ({ok / n:.0%})'
    if dec_n:
        summ += (f'<br>Forced winner (draw ignored): <b>{dec_ok}/{dec_n} right '
                 f'({dec_ok / dec_n:.0%})</b> on games that had a winner')
    return f'<div class="hcsum">{summ}</div><div class="hcgrid">' + "".join(cards) + "</div>"

CSS = """<style>
/* Editorial-dark forecast almanac. ONE accent (amber = trophy) over warm-graphite
   neutrals; green/red reserved for semantic right/wrong. Radius lock: cards 14px,
   controls 8px, pills full. */
:root{--bg:#0b0c10;--card:#13161c;--ink:#eceae4;--ink2:#c9c6bd;--muted:#8f8b82;
--accent:#f0b23e;--accent-dim:#c8912e;--pos:#61c07d;--neg:#e0736b;
--line:#22262d;--line2:#2e333b;--r-card:14px;--r-ctl:8px}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);background-image:radial-gradient(680px 320px at 50% -12%,rgba(240,178,62,.05),transparent 62%);background-attachment:fixed}
.dsp{font-family:'Space Grotesk',-apple-system,Segoe UI,Roboto,sans-serif}
.nav{position:sticky;top:0;z-index:9;background:rgba(11,12,16,.86);backdrop-filter:blur(12px) saturate(1.2);border-bottom:1px solid var(--line)}
.nav .inner{max-width:1040px;margin:0 auto;padding:0 22px;display:flex;gap:2px;align-items:center;overflow-x:auto;-webkit-overflow-scrolling:touch}
.nav .brand{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;margin-right:18px;white-space:nowrap;letter-spacing:-.01em;color:var(--ink)}
.nav .brand b{color:var(--accent);font-weight:700}
.nav a{padding:16px 13px;color:var(--muted);text-decoration:none;font-size:13.5px;font-weight:600;
white-space:nowrap;border-bottom:2px solid transparent;transition:color .15s,border-color .15s}
.nav a.active{color:var(--ink);border-bottom-color:var(--accent)}.nav a:hover{color:var(--ink)}
.wrap{max-width:1040px;margin:0 auto;padding:40px 22px 80px}
h1{font-family:'Space Grotesk',sans-serif;font-size:40px;font-weight:700;margin:0 0 10px;letter-spacing:-.03em;line-height:1.02}
.sub{color:var(--muted);margin:0 0 14px;font-size:15px;max-width:64ch;line-height:1.55}
.phase{display:inline-flex;align-items:center;background:rgba(240,178,62,.10);color:var(--accent);
font-size:12px;font-weight:600;padding:6px 13px;border-radius:999px;margin:2px 0 34px;border:1px solid rgba(240,178,62,.22);letter-spacing:.01em}
h2{font-family:'Space Grotesk',sans-serif;font-size:12px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.09em;margin:0 0 18px;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r-card);
padding:26px;margin-bottom:24px;transition:border-color .18s}
.card:hover{border-color:var(--line2)}
.bars{display:flex;flex-direction:column;gap:9px}
.bar{display:grid;grid-template-columns:24px 120px 1fr 52px;align-items:center;gap:13px;font-size:14px;padding:3px 0}
.bar .rk{font-family:'Space Grotesk',sans-serif;font-size:12.5px;color:var(--muted);text-align:center;font-variant-numeric:tabular-nums}
.bar .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:500;color:var(--ink2)}
.track{background:#0c0e12;border:1px solid var(--line);border-radius:var(--r-ctl);height:24px;overflow:hidden}
.fill{display:block;background:linear-gradient(90deg,#3c414a,#565c66);height:100%;border-radius:var(--r-ctl);transition:width .5s cubic-bezier(.16,1,.3,1)}
.bar:hover .track{border-color:var(--line2)}.bar:hover .nm{color:var(--ink)}
.bar.lead .fill{background:linear-gradient(90deg,var(--accent-dim),var(--accent))}
.bar.lead .nm{font-weight:700;color:var(--ink)}.bar.lead .rk{color:var(--accent)}
.bar.lead .v{color:var(--accent)}.bar.podium .rk{color:var(--ink2)}
.v{text-align:right;font-family:'Space Grotesk',sans-serif;font-variant-numeric:tabular-nums;font-weight:600}
.nm img,td img,.mt img,.ghead img{border-radius:2px;vertical-align:-2px;margin-right:8px;box-shadow:0 0 0 .5px rgba(255,255,255,.12)}
tbody tr{transition:background .12s}tbody tr:hover{background:rgba(240,178,62,.05)}
table{width:100%;border-collapse:collapse;font-size:14px}.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{text-align:right;padding:10px 9px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
td:not(:first-child){font-family:'Space Grotesk',sans-serif}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
tr:last-child td{border-bottom:none}.foot{color:var(--muted);font-size:13px;margin-top:14px}
.hint{font-size:11px;color:var(--muted);font-weight:400;text-transform:none;letter-spacing:0}
.preds{display:flex;flex-direction:column;gap:5px}
.pred{display:flex;justify-content:space-between;align-items:center;font-size:12.5px;gap:8px;flex-wrap:wrap;padding:5px 9px;border-radius:var(--r-ctl)}
.pred:nth-child(even){background:rgba(255,255,255,.02)}
.pred.dim{opacity:.5}.pred .sc{font-family:'Space Grotesk',sans-serif;font-variant-numeric:tabular-nums;font-weight:600;margin:0 2px}
.grp{font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.07em;margin:18px 0 7px;padding-top:14px;border-top:1px solid var(--line)}
.grp.first{border-top:none;padding-top:0;margin-top:2px}
.badge{font-size:11.5px;padding:3px 9px;border-radius:999px;font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}
.b-pend{background:#191d24;color:var(--muted)}.b-ok{background:rgba(97,192,125,.14);color:var(--pos)}.b-no{background:rgba(224,115,107,.13);color:var(--neg)}.b-done{background:#191d24;color:#6b6b63}.b-amber{background:rgba(240,178,62,.14);color:var(--accent)}
.hcsum{font-size:13px;color:var(--muted);margin-bottom:16px}
.hcgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:8px}
.hc{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:12.5px;background:#0f1217;border:1px solid var(--line);border-radius:var(--r-ctl);padding:9px 12px;transition:border-color .15s}
.hc:hover{border-color:var(--line2)}.hc .sc{font-family:'Space Grotesk',sans-serif;font-weight:600;margin:0 3px}
.hc .lean{color:var(--muted);font-size:11px;margin-left:5px;white-space:nowrap}.hc .lean-ok{color:var(--pos)}.hc .lean-no{color:var(--neg)}
.games{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.game{padding:20px 22px;margin:0}
.ghead{font-family:'Space Grotesk',sans-serif;font-size:15.5px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;flex-wrap:wrap;gap:4px}
.ghead .vs{color:var(--muted);font-weight:400;margin:0 5px}
.bet{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 0;border-top:1px solid var(--line);flex-wrap:wrap}
.bet:first-of-type{border-top:none}
.bsel{display:flex;align-items:center;gap:8px;font-size:13.5px;min-width:0}
.bmeta{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);white-space:nowrap}
.edge{color:var(--ink);font-family:'Space Grotesk',sans-serif;font-variant-numeric:tabular-nums;font-weight:600}.edge.pos{color:var(--pos)}
.vb{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;padding:2px 7px;border-radius:5px;white-space:nowrap}
.vb.bet{background:rgba(97,192,125,.15);color:var(--pos)}.vb.lean{background:rgba(240,178,62,.15);color:var(--accent)}.vb.avoid{background:#191d24;color:#6b6b63}
.vb.mkt{background:rgba(240,178,62,.12);color:var(--accent)}
.gfoot{color:var(--muted);font-size:11.5px;margin-top:13px}
.gkick{font-size:11.5px;color:var(--accent);font-weight:600;margin:-5px 0 11px;font-variant-numeric:tabular-nums}
.gadv{font-size:11.5px;color:var(--muted);margin:-5px 0 11px;font-variant-numeric:tabular-nums}.gadv b{color:var(--ink);font-family:'Space Grotesk',sans-serif}
.pod{display:flex;flex-direction:column;gap:8px}
.pod-row{display:grid;grid-template-columns:104px 1fr auto;align-items:center;gap:14px;padding:12px 15px;background:#0f1217;border:1px solid var(--line);border-radius:var(--r-ctl)}
.pod-row .pl{font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.pod-row .tm{font-size:16px;font-weight:600;color:var(--ink2)}
.pod-row .via{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;text-align:right}
.pod-row.gold{border-color:rgba(240,178,62,.38);background:linear-gradient(90deg,rgba(240,178,62,.10),rgba(240,178,62,.015))}
.pod-row.gold .pl{color:var(--accent)}
.pod-row.gold .tm{font-family:'Space Grotesk',sans-serif;font-size:23px;font-weight:700;letter-spacing:-.02em;color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px}
.stat{background:#0f1217;border:1px solid var(--line);border-radius:var(--r-ctl);padding:15px 16px}
.stat .n{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;letter-spacing:-.02em;line-height:1.1;font-variant-numeric:tabular-nums}
.stat .l{font-size:11.5px;color:var(--muted);margin-top:6px;line-height:1.45}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:var(--r-ctl);padding:14px 16px;font-size:12.5px;color:var(--muted);margin-bottom:22px}
.note b{color:var(--ink)}
.sitefoot{max-width:1040px;margin:0 auto;padding:26px 22px;color:#6b6b63;font-size:11px;line-height:1.7;border-top:1px solid var(--line)}.sitefoot b{color:var(--muted)}
@media(max-width:640px){
.wrap{padding:26px 14px 60px}.nav .inner{padding:0 14px}
h1{font-size:29px}.sub{font-size:13.5px}
.card{padding:18px 15px}.card h2{font-size:11.5px}.game{padding:16px}
.bar{grid-template-columns:20px 94px 1fr 44px;gap:9px;font-size:13px}.bar .nm img{margin-right:5px}
table{table-layout:fixed}th:first-child,td:first-child{width:44%;overflow-wrap:break-word}
th,td{padding:9px 5px;font-size:12.5px}.col-opt{display:none}
.hcgrid,.games{grid-template-columns:1fr}.hc{font-size:12px;flex-wrap:wrap}.pred{font-size:12px}
.pod-row{grid-template-columns:1fr auto;gap:3px 10px;padding:11px 13px}
.pod-row .pl{grid-column:1/-1}.pod-row .via{font-size:11px}.pod-row.gold .tm{font-size:20px}
.stat{padding:13px 14px}.stat .n{font-size:23px}
.bmeta{font-size:11px;gap:6px}
}
</style>"""

NAV = [("index.html", "Overview"), ("table.html", "All teams"),
       ("scores.html", "Group scores"), ("knockout.html", "Knockout"), ("bets.html", "Bets")]


DISCLAIMER = (
    '<div class="sitefoot"><b>Disclaimer.</b> WC2026 is an independent statistical model published '
    'for educational and informational purposes only. Nothing on this site is betting, financial, or '
    'investment advice, no result is guaranteed, and modelled performance does not predict future '
    'outcomes. We are not a bookmaker, do not accept or place wagers, and are not affiliated with any '
    'sportsbook. Odds are shown for comparison only and may be inaccurate or out of date. Any '
    'decision you make is your own, and we accept no liability for any loss or damage arising from use '
    'of this information. Betting carries financial risk; only stake what you can afford to lose. '
    '18+ (21+ where required). If gambling is a problem, call 1-800-GAMBLER.</div>')


def _page(active: str, title: str, body: str) -> str:
    links = ""
    for href, lbl in NAV:
        cls = ' class="active"' if href == active else ""
        links += f'<a href="{href}"{cls}>{lbl}</a>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">'
            f'{CSS}</head><body><div class="nav"><div class="inner">'
            f'<span class="brand">WC<b>2026</b></span>{links}</div></div>'
            f'<div class="wrap">{body}</div>{DISCLAIMER}</body></html>')


def _hero(title: str, sub: str, phase: str) -> str:
    return (f'<h1>{title}</h1><p class="sub">{sub}</p>'
            + (f'<div class="phase">{phase}</div>' if phase else ""))


def _pc(c, i, n):
    return f"{c[i] / n:.1%}" if c[i] else "-"


def render_overview(teams, order, R, n_sims, phase) -> str:
    champ = R["champ"]; mx = max(1, champ[order[0]])
    bars = "".join(
        f'<div class="bar{" lead" if rank==1 else " podium" if rank<=3 else ""}">'
        f'<span class="rk">{rank}</span><span class="nm">{flag(teams[i])}{teams[i]}</span>'
        f'<span class="track"><span class="fill" style="width:{max(2,champ[i]/mx*100):.0f}%"></span></span>'
        f'<span class="v">{_pc(champ, i, n_sims)}</span></div>'
        for rank, i in enumerate(order[:32], 1))
    sub = ("Dixon-Coles + connectivity-weighted squad value, blended with pi-ratings, altitude-aware "
           f"&middot; {n_sims:,} Monte-Carlo runs")
    foot = ('<p class="foot">A calibrated distribution, not a single pick. The favourite '
            'tops out ~16%. See <a href="table.html" style="color:var(--accent2)">all teams</a>.</p>')
    return (_hero("2026 World Cup forecast", sub, phase)
            + f'<div class="card"><h2>Title odds <span class="hint">top 32</span></h2>'
            f'<div class="bars">{bars}</div>{foot}</div>')


def render_table(teams, order, R, n_sims, phase) -> str:
    rows = "".join(
        f"<tr><td>{flag(teams[i])}{teams[i]}</td><td>{_pc(R['qualify'],i,n_sims)}</td>"
        f"<td>{_pc(R['champ'],i,n_sims)}</td><td class='col-opt'>{_pc(R['final'],i,n_sims)}</td>"
        f"<td class='col-opt'>{_pc(R['sf'],i,n_sims)}</td><td>{_pc(R['r16'],i,n_sims)}</td></tr>"
        for i in order)
    return (_hero("All teams", "Chance of reaching each stage", phase)
            + '<div class="card"><div class="tbl"><table><thead><tr><th>Team</th><th>Qualify</th>'
            '<th>Champion</th><th class="col-opt">Final</th><th class="col-opt">Semi</th><th>R16</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div></div>')


def render_scores(preds, hc, phase) -> str:
    return (_hero("Group stage scores", "Predicted scorelines and group backtest", phase)
            + '<div class="card"><h2>Score predictions <span class="hint">by group &middot; '
            f'green = exact</span></h2>{render_preds(preds)}</div>'
            + '<div class="card"><h2>Group hindcast <span class="hint">how it would have called '
            f'played games &middot; no hindsight</span></h2>{render_hindcast(hc["group"])}</div>')


def render_knockout_results(grouped):
    if not grouped:
        return ("<p style='color:var(--muted);font-size:13px;margin:0'>Knockout ties appear "
                "here once the bracket is set.</p>")
    out = []
    for gi, (label, rows) in enumerate(grouped):
        out.append(f'<div class="grp{" first" if gi == 0 else ""}">{label}</div>')
        for r in rows:
            if r["status"] == "upcoming":                       # predicted score + backed advancer
                score = f'{r["pa"]}-{r["pb"]}'
                adv = (f'<span class="lean">&rarr; {short(r["winner"])} '
                       f'{r["win_p"]*100:.0f}%</span>')
                badge = '<span class="badge b-pend">upcoming</span>'
            else:                                               # played: actual score + who advanced
                score = f'<b>{r["act_a"]}-{r["act_b"]}</b>'
                pens = ' <span class="lean">(pens)</span>' if r["pens"] else ''
                adv = (f'<span class="lean lean-ok">&rarr; {short(r["winner"])} '
                       f'advances{pens}</span>')
                badge = {"correct": '<span class="badge b-ok">&#10003; exact</span>',
                         "adv_ok": '<span class="badge b-amber">called it</span>',
                         "wrong": '<span class="badge b-no">missed</span>',
                         "result": '<span class="badge b-done">played</span>'}[r["status"]]
            match = (f'<span class="mt">{flag(r["a"])}{short(r["a"])}</span> '
                     f'<span class="sc">{score}</span> '
                     f'<span class="mt">{flag(r["b"])}{short(r["b"])}</span>{adv}')
            out.append(f'<div class="pred"><span>{match}</span>{badge}</div>')
    return '<div class="preds">' + "".join(out) + "</div>"


PLACE_RANK = {"Champions": 0, "Runners-up": 1, "Third place": 2, "Fourth place": 3,
              "Semi-finals": 4, "Quarter-finals": 5, "Round of 16": 6, "Round of 32": 7,
              "Group stage": 8}


def final_standings(ko_grouped, teams):
    """Where every team actually finished, read off the played bracket.

    Single-elimination means a team loses at most once -- except the two semi-final losers,
    who meet again in the third-place play-off. So the final and that play-off are resolved
    first (they name an exact place), and every other team then falls back to the round it
    lost in. Returns ({team: place label}, champion or None)."""
    rounds = dict(ko_grouped)
    played = lambda rows: [r for r in rows if r["status"] != "upcoming"]
    loser = lambda r: r["b"] if r["winner"] == r["a"] else r["a"]
    place = {}
    for label, won, lost in (("Final", "Champions", "Runners-up"),
                             ("Third-place play-off", "Third place", "Fourth place")):
        for r in played(rounds.get(label, [])):
            place[r["winner"]] = won; place[loser(r)] = lost
    for label, rows in ko_grouped:                    # everyone else exits where they lost
        for r in played(rows):
            place.setdefault(loser(r), label)
    for t in teams:                                   # never reached the knockouts
        place.setdefault(t, "Group stage")
    return place, next((t for t, p in place.items() if p == "Champions"), None)


def _decider(ko_grouped, label):
    """One-line 'how that place was settled' for the final / third-place tie."""
    rows = [r for r in dict(ko_grouped).get(label, []) if r["status"] != "upcoming"]
    if not rows:
        return ""
    r = rows[0]
    return (f'{short(r["a"])} {r["act_a"]}&ndash;{r["act_b"]} {short(r["b"])}'
            + (" <span class='lean'>(pens)</span>" if r["pens"] else ""))


def render_final_overview(teams, place, champion, hc, ko_grouped) -> str:
    """Tournament-over front page: who actually won, and how the model's pre-tournament
    call held up. The live version (title odds bars) is meaningless once one team is left."""
    fin, third = _decider(ko_grouped, "Final"), _decider(ko_grouped, "Third-place play-off")
    by_place = {p: t for t, p in place.items()}
    pod = "".join(
        f'<div class="pod-row{" gold" if lbl == "Champions" else ""}">'
        f'<span class="pl">{lbl}</span>'
        f'<span class="tm">{flag(by_place.get(lbl, ""))}{by_place.get(lbl, "-")}</span>'
        f'<span class="via">{via}</span></div>'
        for lbl, via in (("Champions", fin), ("Runners-up", fin),
                         ("Third place", third), ("Fourth place", third))
        if lbl in by_place)

    g, k = hc["group"], hc["knockout"]                 # (rows, exact, ok, n, dec_ok, dec_n)
    exact, ok, n = g[1] + k[1], g[2] + k[2], g[3] + k[3]
    dec_ok, dec_n = g[4] + k[4], g[5] + k[5]
    stats = "".join(f'<div class="stat"><div class="n">{v}</div><div class="l">{l}</div></div>'
                    for v, l in (
                        (f"{ok/n:.0%}" if n else "-", f"outcomes called right ({ok}/{n})"),
                        (f"{exact}", f"exact scorelines out of {n}"),
                        (f"{dec_ok/dec_n:.0%}" if dec_n else "-",
                         f"winner picked right ({dec_ok}/{dec_n} decided games)")))

    return (_hero("2026 World Cup", f"{champion} win the 2026 World Cup. Final result and a "
                  "report card for the model, now that every game has been played.",
                  "Tournament complete")
            + f'<div class="card"><h2>Final standings <span class="hint">104 games played'
              f'</span></h2><div class="pod">{pod}</div></div>'
            + '<div class="card"><h2>Model report card <span class="hint">pre-tournament model '
              '&middot; fit only on data before 11 Jun 2026, no hindsight</span></h2>'
              f'<div class="stats">{stats}</div>'
              '<p class="foot">Every 2026 game scored by a model that never saw a single '
              'tournament result. Game-by-game breakdowns are on the '
              '<a href="scores.html" style="color:var(--accent)">group</a> and '
              '<a href="knockout.html" style="color:var(--accent)">knockout</a> pages.</p></div>')


def render_final_table(teams, place) -> str:
    rows = "".join(
        f"<tr><td>{flag(t)}{t}</td><td>{place[t]}</td></tr>"
        for t in sorted(teams, key=lambda x: (PLACE_RANK.get(place[x], 9), x)))
    return (_hero("All teams", "Where each of the 48 teams finished", "Tournament complete")
            + '<div class="card"><div class="tbl"><table><thead><tr><th>Team</th>'
              f'<th>Finished</th></tr></thead><tbody>{rows}</tbody></table></div></div>')


def render_knockout(ko_grouped, hc, phase) -> str:
    ko = hc["knockout"]
    ko_hint = ("someone advances &middot; penalty shootouts resolved" if ko[3]
               else "fills in as ties are played")
    done = not any(r["status"] == "upcoming" for _, rows in ko_grouped for r in rows)
    sub, hint = (("The complete bracket, final first.",
                  "actual score &amp; who advanced &middot; graded against the locked pre-match call")
                 if done else
                 ("Results and who advanced, with predictions for upcoming ties",
                  "played = actual score &amp; who advanced &middot; upcoming = prediction"))
    return (_hero("Knockout scores", sub, phase)
            + f'<div class="card"><h2>Knockout bracket <span class="hint">{hint}</span></h2>'
            f'{render_knockout_results(ko_grouped)}</div>'
            + f'<div class="card"><h2>Knockout hindcast <span class="hint">{ko_hint}</span></h2>'
            f'{render_hindcast(ko)}</div>')


def _kickoff(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%b %d &middot; %H:%M UTC")
    except ValueError:
        return ""


def render_bets(final=False) -> str | None:
    """Bake bets.json (from recommend_bets.py) into a static page. None if no data yet.
    final=True labels the page as a closed archive: the slate is whatever was last fetched
    before the tournament ended, and no further odds or settlements are coming."""
    path = os.path.join(os.path.dirname(__file__), "bets.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    games = [g for g in d.get("games", []) if g.get("topBets")]
    games.sort(key=lambda g: g.get("commence") or "z")        # next kickoff first
    cards = []
    for g in games:
        rows = []
        for b in g["topBets"]:
            v = b["recommendation"]
            mv = ' <span class="vb mkt">value</span>' if b.get("valueType") == "market" else ""
            rows.append(
                f'<div class="bet"><span class="bsel"><span class="vb {v}">{v}</span>{b["selection"]}{mv}</span>'
                f'<span class="bmeta"><b class="edge{" pos" if b["edge"] > 0 else ""}">'
                f'{b["edge"]*100:+.0f}%</b><span>{b["modelProbability"]*100:.0f}% vs '
                f'{b["sportsbookImpliedProbability"]*100:.0f}%</span><span>{b["confidence"]}</span></span></div>')
        kick = _kickoff(g.get("commence"))
        adv = g.get("advance")
        adv_line = (f'<div class="gadv">To advance &middot; {short(g["home"])} '
                    f'<b>{adv["HOME"]*100:.0f}%</b> &middot; {short(g["away"])} '
                    f'<b>{adv["AWAY"]*100:.0f}%</b></div>') if adv else ""
        fc = g.get("forecast")
        fc_line = ""
        if fc and fc.get("market"):
            a = fc["anchored"]
            fc_line = (f'<div class="gadv"><b>Forecast</b> (market-anchored, experimental) &middot; '
                       f'{short(g["home"])} <b>{a[0]*100:.0f}%</b> &middot; draw '
                       f'<b>{a[1]*100:.0f}%</b> &middot; {short(g["away"])} <b>{a[2]*100:.0f}%</b></div>')
        cards.append(
            f'<div class="card game"><div class="ghead">{flag(g["home"])}{short(g["home"])}'
            f'<span class="vs">v</span>{flag(g["away"])}{short(g["away"])}</div>'
            + (f'<div class="gkick">{kick}</div>' if kick else "")
            + adv_line + fc_line
            + f'{"".join(rows)}'
            f'<div class="gfoot">{g.get("avoidsCount", 0)} other markets screened out</div></div>')
    note = ('<div class="note"><b>Edge</b> = model probability &minus; <b>Pinnacle&rsquo;s '
            'de-vigged line</b> (the sharpest market = best estimate of true probability); '
            'odds shown are the best price across books. Top 5 de-correlated picks per game. '
            '<b>Totals &amp; BTTS markets show as leans only</b>. The model&rsquo;s goal totals '
            'aren&rsquo;t calibrated yet, so they aren&rsquo;t staked as bets. No historical odds = '
            'not ROI-backtested. The <b>market-anchored forecast</b> blends the model with '
            'Pinnacle&rsquo;s de-vigged line (the sharpest probability estimate); it&rsquo;s '
            'experimental and judged live by CLV. Informational, not betting advice.</div>')
    head = _hero("Value bets",
                 "Model probabilities against the sportsbook odds of the day"
                 + (" &mdash; archived exactly as the last refresh left it." if final
                    else ""),
                 (f'Closed &middot; last refreshed {d.get("generatedAt", "")}' if final
                  else f'updated {d.get("generatedAt", "")}'))
    cs = clv.summary()
    if cs["settled"]:
        clv_line = (f'<div class="note"><b>Closing Line Value: {cs["avg_clv"]:+.1%}</b> over '
                    f'{cs["settled"]} settled picks: '
                    + ('the model ' + ('beat' if final else 'is beating')
                       + ' the market’s closing price (real edge).'
                       if cs["avg_clv"] > 0 else
                       'behind the closing line' + ('.' if final else ' so far.')) + '</div>')
    elif final:
        clv_line = (f'<div class="note"><b>CLV was never settled.</b> {cs["logged"]} picks were '
                    'logged but no closing prices were captured against them, so this run '
                    'produced <b>no evidence either way</b> on whether the edges were real. '
                    'Treat the slate below as a record of what the model said, not as a '
                    'validated result.</div>')
    else:
        clv_line = (f'<div class="note">CLV validation active: {cs["logged"]} picks logged. '
                    'Closing Line Value (whether our entry price beats the market’s '
                    'final price) populates as games kick off; positive average CLV is the '
                    'evidence the edges are real.</div>')
    body = (head + note + clv_line + (f'<div class="games">{"".join(cards)}</div>' if cards
            else '<div class="card"><p style="color:var(--muted);margin:0">No positive-edge '
                 'bets in the current slate.</p></div>'))
    return body


def _bets_placeholder() -> str:
    return (_hero("Value bets", "Model probabilities vs live sportsbook odds", "")
            + '<div class="card"><p style="color:var(--muted);margin:0">No bets generated yet. '
            'Run <code>python recommend_bets.py</code> (needs ODDS_API_KEY) to populate this page.</p></div>')


def write_site(teams, order, R, n_sims, phase, preds, hc, ko_grouped, docsdir, champion=None):
    """champion set => the tournament is over, so the front page and the team table show what
    actually happened instead of a forecast that has collapsed onto a single certain team."""
    os.makedirs(docsdir, exist_ok=True)
    if champion:
        place, _ = final_standings(ko_grouped, teams)
        overview = render_final_overview(teams, place, champion, hc, ko_grouped)
        table = render_final_table(teams, place)
    else:
        overview = render_overview(teams, order, R, n_sims, phase)
        table = render_table(teams, order, R, n_sims, phase)
    pages = {
        "index.html": ("2026 World Cup Forecast", overview),
        "table.html": ("All teams - WC2026", table),
        "scores.html": ("Group scores - WC2026", render_scores(preds, hc, phase)),
        "knockout.html": ("Knockout scores - WC2026", render_knockout(ko_grouped, hc, phase)),
    }
    for fname, (title, body) in pages.items():
        with open(os.path.join(docsdir, fname), "w", encoding="utf-8") as f:
            f.write(_page(fname, title, body))
    # Bets: bake bets.json if present; else only write a placeholder if no page exists yet
    # (so a CI run without bets.json never clobbers a previously committed bets page).
    bets_body = render_bets(final=bool(champion))
    bets_path = os.path.join(docsdir, "bets.html")
    if bets_body is not None:
        with open(bets_path, "w", encoding="utf-8") as f:
            f.write(_page("bets.html", "Value bets - WC2026", bets_body))
    elif not os.path.exists(bets_path):
        with open(bets_path, "w", encoding="utf-8") as f:
            f.write(_page("bets.html", "Value bets - WC2026", _bets_placeholder()))

if __name__ == "__main__":
    df = wc.load()
    m = get_model(df)
    zmap, confmap = mvmod.setup(m)
    wcdf = wc_games(df)
    gdf, kdf = split_games(wcdf)
    groups = groups_from(gdf)
    teams = sorted({t for g in groups for t in g})
    tidx = {t: i for i, t in enumerate(teams)}
    _, pi_predict = wc.fit_pi_wdl(df)
    A = adv_matrix(m, teams, zmap, confmap, pi_predict)
    rem = remaining_group_games(groups, gdf)

    # Option B: the R32 (Match 79) and R16 (Match 92) for Mexico's group winner are at Estadio
    # Azteca (2240m). Build an altitude-adjusted advantage matrix + locate Mexico's group.
    # WC_NO_ALT_KO disables it (for the A/B accuracy check).
    A_azteca = None if os.environ.get("WC_NO_ALT_KO") else \
        adv_matrix(m, teams, zmap, confmap, pi_predict, city="Mexico City",
                   home_nations=KO_HOME | {"Mexico"})       # Mexico's genuine home slot
    azteca_group = next((i for i, g in enumerate(groups) if "Mexico" in g), None) \
        if A_azteca is not None else None

    champion = None                                      # set once the final has been played
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
        # One team left unbeaten = the final has been played, so there is nothing left to
        # forecast; the site switches to its finished-tournament pages.
        champion = teams[alive[0]] if len(alive) == 1 else None
        phase = (f"Tournament complete - {champion} champions" if champion else
                 f"Knockouts - {len(kdf)} games played, {len(alive)} teams alive")

    order = sorted(range(len(teams)), key=lambda i: R["champ"][i], reverse=True)
    print(phase)
    preds = update_and_grade(m, zmap, confmap, rem, gdf, groups)
    hc = hindcast(df, wcdf)
    ko_grouped = update_and_grade_knockout(m, zmap, confmap, df, wcdf)

    if champion:            # nothing left to forecast -> report where everyone finished
        place, _ = final_standings(ko_grouped, teams)
        ranked = sorted(teams, key=lambda t: (PLACE_RANK.get(place[t], 9), t))
        print(f"\n  {'team':<18}finished")
        for t in ranked[:8]:
            print(f"  {t:<18}{place[t]}")
        lines = ["# 2026 World Cup - final standings", "", f"_{phase}_", "",
                 "| Team | Finished |", "|---|---|"]
        lines += [f"| {t} | {place[t]} |" for t in ranked]
    else:
        pct = lambda c, i: f"{c[i] / N_SIMS:.1%}"
        print(f"\n  {'team':<18}{'qualify':>8}{'champ':>8}{'final':>8}")
        for i in order[:16]:
            q = R["qualify"][i] / N_SIMS; c = R["champ"][i] / N_SIMS; f_ = R["final"][i] / N_SIMS
            print(f"  {teams[i]:<18}{q:>7.0%}{c:>8.1%}{f_:>8.1%}")
        lines = ["# 2026 World Cup forecast", "", f"_{phase}_", "",
                 "| Team | Qualify | Champion | Final | Semi | R16 |", "|---|---|---|---|---|---|"]
        lines += [f"| {teams[i]} | {pct(R['qualify'],i)} | {pct(R['champ'],i)} | "
                  f"{pct(R['final'],i)} | {pct(R['sf'],i)} | {pct(R['r16'],i)} |" for i in order]
    with open(os.path.join(os.path.dirname(__file__), "forecast.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    write_site(teams, order, R, N_SIMS, phase, preds, hc, ko_grouped,
               os.path.join(os.path.dirname(__file__), "docs"), champion=champion)

    assert abs(sum(R["champ"].values()) - N_SIMS) < 1
    print(f"\nwrote forecast.md + docs/ (overview, table, group, knockout, bets) | {phase}")
