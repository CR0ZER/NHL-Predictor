"""Actualité d'avant-match : NHL.com -> extraction qwen3:8b (Ollama, local) -> jugement Jev -> variables du modèle.

    uv run --env-file .env python news.py eval [SAISON]   # chaîne complète sur une saison, log-loss par variante

Utilisée en direct par nhl.predictions() (bouton Prédire). Chaque passage est conservé dans data/news_runs/<gameId>.json
(le dernier avant le coup d'envoi), ce qui permet l'évaluation au fil de l'eau.
Anti-fuite : seuls les articles publiés ET modifiés pour la dernière fois avant le coup d'envoi, hors comptes rendus.
"""
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd
from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafeClient

import nhl
from xg import DATA, WEB, get_json

CONTENT = "https://forge-dapi.d3.nhle.com/v2/content/en-us/stories"
OLLAMA, LLM = "http://127.0.0.1:11434", "qwen3:8b"
NUM_CTX = 10240  # tient entièrement dans les 8 Go de la carte graphique avec qwen3:8b (16k débordait sur le CPU)
WINDOW_H, MAX_ARTICLES, MAX_CHARS = 72, 6, 3500
RUNS = DATA / "news_runs"
UNKNOWN = "unknown"

PLAYER = {"type": "object", "properties": {"player": {"type": "string"}, "detail": {"type": "string"}},
          "required": ["player", "detail"]}
TEAM = {"type": "object", "properties": {
    "starting_goalie": {"type": ["string", "null"]},
    "goalie_status": {"type": "string", "enum": ["confirmed", "projected", "unknown"]},
    "out": {"type": "array", "items": PLAYER},
    "doubtful": {"type": "array", "items": PLAYER},
    "returning": {"type": "array", "items": {"type": "string"}},
    "summary": {"type": "string"},
}, "required": ["starting_goalie", "goalie_status", "out", "doubtful", "returning", "summary"]}
SCHEMA = {"type": "object", "properties": {"away": TEAM, "home": TEAM}, "required": ["away", "home"]}
PROMPT = """You prepare pre-game information for an NHL game, using ONLY the articles provided (all published before
puck drop). For each team (away, home) extract:
- starting_goalie: the goalie expected to start THIS game (in projected lineups, the first goalie listed), or null
- goalie_status: "confirmed" if announced/confirmed, "projected" if only projected, "unknown" if not mentioned
- out: players who will miss THIS game (injured, suspended, scratched, illness, personal, traded), with a short detail
- doubtful: players whose participation is uncertain (game-time decision, day-to-day, questionable)
- returning: players returning to the lineup for this game
- summary: 2-3 sentences on the team's situation (form, lineup changes, context)
Never invent players or facts. If the articles say nothing, use empty lists, null and "unknown"."""
EMPTY = {"starting_goalie": None, "goalie_status": "unknown", "out": [], "doubtful": [], "returning": [], "summary": ""}


def norm(name):
    return unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower().strip()


def clean(md):
    return re.sub(r"\s+\n", "\n", re.sub(r"<[^>]+>|\*|\\", "", md)).strip()


def ready():
    """La chaîne est utilisable : Ollama répond avec qwen3:8b, et une clé Jev est configurée."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            models = {m["name"] for m in json.load(r)["models"]}
    except OSError:
        return False
    return LLM in models and bool(os.environ.get("TYPESAFE_API_KEY"))


# ---------------------------------------------------------------- récupération (NHL.com)
_lists, _rosters = {}, {}


def fresh():
    """Oublie les listes d'articles en mémoire (le serveur tourne longtemps : de nouveaux articles paraissent)."""
    _lists.clear()


def stories(tag):
    if tag not in _lists:
        _lists[tag] = get_json(f"{CONTENT}?tags.slug={tag}&$limit=100")["items"]
    return _lists[tag]


def story_text(slug, updated):
    """Texte d'un article, mis en cache par version (un article modifié est relu)."""
    path = DATA / "news" / f"{slug}_{re.sub(r'[^0-9]', '', updated)[:14]}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        parts = get_json(f"{CONTENT}/{slug}")["parts"]
        text = "\n".join(p["content"] for p in parts if isinstance(p, dict) and isinstance(p.get("content"), str))
        path.write_text(clean(text), encoding="utf-8")  # les blocs non textuels (vidéos, images) sont ignorés
    return path.read_text(encoding="utf-8")


def pregame_articles(game_id, home_id, away_id, kickoff):
    """Articles NHL.com du match et des deux équipes, publiés et modifiés avant le coup d'envoi (fenêtre 72 h).
    Les « projected lineups » du match passent en premier."""
    kickoff = pd.Timestamp(kickoff)
    seen, out = set(), []
    for tag in (f"gameid-{game_id}", f"teamid-{home_id}", f"teamid-{away_id}"):
        for s in stories(tag):
            tags = {t["slug"] for t in s["tags"]}
            published, updated = pd.Timestamp(s["contentDate"]), pd.Timestamp(s["lastUpdatedDate"])
            if (s["slug"] in seen or "game-recap" in tags or updated >= kickoff or published >= kickoff
                    or published < kickoff - timedelta(hours=WINDOW_H)):
                continue
            seen.add(s["slug"])
            out.append({"title": s.get("headline") or s.get("title", ""), "published": s["contentDate"],
                        "lineups": f"gameid-{game_id}" in tags,
                        "text": story_text(s["slug"], s["lastUpdatedDate"])[:MAX_CHARS]})
    return sorted(out, key=lambda a: (not a["lineups"], a["published"]))[:MAX_ARTICLES]


def roster_goalies(abbrev, season=None):
    season = season or nhl.current_season()
    if (abbrev, season) not in _rosters:
        r = get_json(f"{WEB}/roster/{abbrev}/{season}{season + 1}")
        _rosters[abbrev, season] = {f"{p['firstName']['default']} {p['lastName']['default']}": p["id"]
                                    for p in r["goalies"]}
    return _rosters[abbrev, season]


# ---------------------------------------------------------------- qwen (extraction) et Jev (jugement)
def extract(game, articles):
    """qwen3:8b en local : informations d'avant-match structurées pour les deux équipes (JSON imposé)."""
    body = {"model": LLM, "stream": False, "think": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": json.dumps({"game": game, "articles": articles}, ensure_ascii=False)}]}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    if resp.get("prompt_eval_count", 0) >= NUM_CTX - 1024:  # Ollama tronque sans prévenir
        print(f"  ATTENTION : {resp['prompt_eval_count']} tokens en entrée, proche de la limite de contexte", flush=True)
    return json.loads(resp["message"]["content"])


def judge(client, game, side, articles, info, goalies):
    """Jev : gardien titulaire, et probabilité que chaque absent signalé par qwen manque réellement ce match."""
    team, opp = (game["home"], game["away"]) if side == "home" else (game["away"], game["home"])
    cands = [p["player"] for p in info["out"] + info["doubtful"]][:10]
    state = {"game": {"team": team, "opponent": opp, "kickoff_utc": game["kickoff"]},
             "articles": [{"title": a["title"], "text": a["text"]} for a in articles[:3]], "extraction": info}
    qs = {"starter": Choice(
              instructions="According to `articles`, which goalie starts for `game.team` against `game.opponent` in "
                           "this game? In projected lineups the first goalie listed is the projected starter.",
              criteria={g: None for g in goalies} | {UNKNOWN: "The articles do not indicate this game's starter"}),
          "announced": Noul(
              instructions="Do `articles` name the goalie who starts for `game.team` in this game?",
              criteria=NoulCriteria(true="A starter is named or projected for this game", false="No starter named"))}
    for i, name in enumerate(cands):
        qs[f"out{i}"] = Noul(
            instructions=f"According to `articles`, will {name} miss this game for `game.team` "
                         f"(injury, illness, suspension, healthy scratch, personal reasons or trade)?",
            criteria=NoulCriteria(true="The articles say or clearly imply he does not play this game",
                                  false="He is expected to play, or nothing indicates he misses this game"))
    r = client.system_one(state, qs, model="jev-latest")
    c = r.choices["starter"]
    return {"starter": c.choice, "p_starter": c.probabilities[c.choice], "announced": r.nouls["announced"].noul,
            "p_out": {name: r.nouls[f"out{i}"].noul for i, name in enumerate(cands)}}


def run_game(client, g, goalies_by_team, refresh=False):
    """Chaîne complète pour un match. Conservée dans data/news_runs/<gameId>.json ; refresh=True la recalcule
    (en direct : les articles évoluent jusqu'au coup d'envoi, la dernière version avant le match fait foi)."""
    path = RUNS / f"{g['gameId']}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.time()
    arts = pregame_articles(g["gameId"], g["home_id"], g["away_id"], g["kickoff"])
    game = {k: g[k] for k in ("away", "home", "kickoff")}
    info = extract(game, arts) if arts else {"away": EMPTY, "home": EMPTY}
    t1 = time.time()
    jev = {s: judge(client, game, s, arts, info[s], goalies_by_team[g[f"{s}_abbr"]]) for s in ("away", "home")}
    out = {"game": g, "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
           "articles": [{k: a[k] for k in ("title", "published", "lineups")} for a in arts],
           "qwen": info, "jev": jev, "seconds": {"qwen": round(t1 - t0, 1), "jev": round(time.time() - t1, 1)}}
    RUNS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def expected_missing(p_out, team_abbr, sk, ppg_now):
    """Points par match attendus des absents : somme de P(absent) selon Jev x points par match du joueur.
    Retourne (total, détail trié par impact) ; un nom introuvable dans les données NHL compte pour 0."""
    names = sk.drop_duplicates("playerId", keep="last")
    team = dict(zip(names.loc[names["teamAbbrev"] == team_abbr, "skaterFullName"].map(norm),
                    names.loc[names["teamAbbrev"] == team_abbr, "playerId"]))
    anyone = dict(zip(names["skaterFullName"].map(norm), names["playerId"]))
    detail = []
    for name, p in p_out.items():
        pid = team.get(norm(name), anyone.get(norm(name)))
        detail.append({"name": name, "p": round(p, 2), "ppg": round(float(ppg_now.get(pid, 0.0)), 2) if pid else None})
    detail.sort(key=lambda d: -(d["p"] * (d["ppg"] or 0)))
    return sum(d["p"] * (d["ppg"] or 0) for d in detail), detail


# ---------------------------------------------------------------- évaluation sur une saison passée ou en cours
def evaluate(season=None):
    season = season or nhl.current_season()
    team, goalies = nhl.load()
    w, _ = nhl.history(team, goalies)
    w = w.dropna(subset=nhl.FEATURES + ["home_win"])
    test = w[w["season"] == season].copy()
    train = w[(w["season"] < season) & (w["season"] > nhl.FIRST_SEASON)]
    base_cols = [f for f in nhl.FEATURES if f != "d_miss"]
    base = nhl.LogisticRegression().fit(train[base_cols], train["home_win"])
    full = nhl.LogisticRegression().fit(train[nhl.FEATURES], train["home_win"])
    sk = nhl.players()

    sched = {}
    for day in sorted(test["date"].dt.date.astype(str).unique()):
        for x in get_json(f"{WEB}/score/{day}")["games"]:
            sched[x["id"]] = x
    g_rat, _ = nhl.goalie_ratings(goalies)
    g_rat["post"] = (g_rat.groupby("playerId")["gsax"].cumsum()
                     / (g_rat.groupby("playerId")["fenwick"].cumsum() + nhl.GOALIE_PRIOR_SHOTS))

    def rating_before(pid, d):
        r = g_rat[(g_rat["playerId"] == pid) & (g_rat["gameDate"] < d)]
        return r["post"].iloc[-1] if len(r) else 0.0

    client, rows = TypeSafeClient(), []
    for i, (gid, row) in enumerate(test.iterrows()):
        x = sched[gid]
        g = {"gameId": int(gid), "kickoff": x["startTimeUTC"], "away": row["away"], "home": row["home"],
             **{f"{s}_id": x[f"{s}Team"]["id"] for s in ("away", "home")},
             **{f"{s}_abbr": x[f"{s}Team"]["abbrev"] for s in ("away", "home")}}
        res = run_game(client, g, {g[f"{s}_abbr"]: roster_goalies(g[f"{s}_abbr"], season) for s in ("away", "home")})
        print(f"{i + 1}/{len(test)} {g['away_abbr']}@{g['home_abbr']} : {len(res['articles'])} articles, "
              f"qwen {res['seconds']['qwen']} s, Jev {res['seconds']['jev']} s", flush=True)
        day = pd.Timestamp(row["date"])
        ppg_now = sk[sk["gameDate"] < day].groupby("playerId")["ppg"].last()
        feat = {}
        for side, s in (("away", "a"), ("home", "h")):
            ab, j = g[f"{side}_abbr"], res["jev"][side]
            heur, _ = nhl.probable_starter(goalies, ab, day)
            jev_pid = roster_goalies(ab, season).get(j["starter"]) if j["announced"] >= 0.5 else None
            actual = goalies[(goalies["gameId"] == gid) & (goalies["gamesStarted"] == 1)
                             & (goalies["teamAbbrev"] == ab)]["playerId"].iloc[0]
            feat[f"{s}_g_heur"], feat[f"{s}_g_jev"] = rating_before(heur, day), rating_before(jev_pid or heur, day)
            feat[f"{s}_goalie_ok_heur"], feat[f"{s}_goalie_ok_jev"] = heur == actual, (jev_pid or heur) == actual
            qwen_out = ({p["player"]: 1.0 for p in res["qwen"][side]["out"]}
                        | {p["player"]: .5 for p in res["qwen"][side]["doubtful"]})
            feat[f"{s}_miss_qwen"] = expected_missing(qwen_out, ab, sk, ppg_now)[0]
            feat[f"{s}_miss_jev"] = expected_missing(j["p_out"], ab, sk, ppg_now)[0]
        rows.append({"gameId": gid, **feat})
    t = test.join(pd.DataFrame(rows).set_index("gameId"))
    k = nhl.SHOTS_PER_GAME
    X = lambda goalie, cols, miss=None: t[cols].assign(d_goalie=k * (t[f"h_g_{goalie}"] - t[f"a_g_{goalie}"]),
                                                       **({"d_miss": miss} if miss is not None else {}))
    p = {
        "A. modèle seul (gardien deviné)": base.predict_proba(X("heur", base_cols))[:, 1],
        "B. + Jev gardien": base.predict_proba(X("jev", base_cols))[:, 1],
        "C. + qwen absences (sans Jev)": full.predict_proba(X("jev", nhl.FEATURES, t["h_miss_qwen"] - t["a_miss_qwen"]))[:, 1],
        "D. chaîne complète (qwen + Jev)": full.predict_proba(X("jev", nhl.FEATURES, t["h_miss_jev"] - t["a_miss_jev"]))[:, 1],
        "E. réel (vrai gardien, vrais absents)": full.predict_proba(t[nhl.FEATURES])[:, 1],
    }
    import odds
    co = odds.closing_odds()
    pin = co[(co["book"] == "pinnacle") & (co["market"] == "151")].pivot_table(index=["home", "date"], columns="side",
                                                                                values="price").reset_index()
    pin["q"] = (1 / pin["H"]) / (1 / pin["H"] + 1 / pin["A"])
    q = t.assign(hn=t["home"].map(odds.norm), dd=t["date"].dt.date).merge(
        pin.rename(columns={"home": "hn", "date": "dd"})[["hn", "dd", "q"]], how="left", on=["hn", "dd"])["q"].to_numpy()
    y = t["home_win"].to_numpy()
    ll = lambda pr: -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    print(f"\n{len(t)} matchs de la saison {season}-{season + 1 - 2000} (du {t['date'].min().date()} au {t['date'].max().date()})")
    print(f"Gardien titulaire trouvé : heuristique {np.mean([t.h_goalie_ok_heur.mean(), t.a_goalie_ok_heur.mean()]):.0%}"
          f" | Jev {np.mean([t.h_goalie_ok_jev.mean(), t.a_goalie_ok_jev.mean()]):.0%}\n")
    ref = ll(p["A. modèle seul (gardien deviné)"])
    for name, pr in p.items():
        d = ll(pr) - ref
        print(f"{name:42s} log-loss {ll(pr).mean():.4f}  écart vs A {d.mean():+.4f} ± {1.96 * d.std() / np.sqrt(len(d)):.4f}")
    has = ~np.isnan(q)
    print(f"{'F. Pinnacle clôture':42s} log-loss {ll(np.where(has, q, .5))[has].mean():.4f}  ({has.sum()} matchs avec cote ;"
          f" modèle A sur ces matchs {ref[has].mean():.4f}, chaîne D {ll(p['D. chaîne complète (qwen + Jev)'])[has].mean():.4f})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
