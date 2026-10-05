"""xG maison à partir des play-by-play NHL : qualité de chaque tir non bloqué -> xG par équipe et GSAx par gardien.

Le modèle de tir est entraîné en walk-forward : les tirs de la saison S sont notés par un modèle appris sur les saisons < S.
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path(__file__).parent / "data"
WEB = "https://api-web.nhle.com/v1"


def get_json(url, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-predictor"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


SHOT_EVENTS = {"shot-on-goal", "goal", "missed-shot"}  # tirs non bloqués (Fenwick)
ANY_SHOT = SHOT_EVENTS | {"blocked-shot"}
XG_FEATURES = ["dist", "angle", "shot_type", "rebound", "rush", "skater_diff", "empty_net", "dt_prev"]


def clock(p):
    m, s = p["timeInPeriod"].split(":")
    return (p["periodDescriptor"]["number"] - 1) * 1200 + int(m) * 60 + int(s)


def game_shots(game_id):
    d = get_json(f"{WEB}/gamecenter/{game_id}/play-by-play")
    home, rows, prev = d["homeTeam"]["id"], [], None
    for p in d["plays"]:
        if p["periodDescriptor"].get("periodType") == "SO":
            break
        det, t = p.get("details", {}), clock(p)
        if p["typeDescKey"] in SHOT_EVENTS and "xCoord" in det and "yCoord" in det and prev:
            rows.append({
                "gameId": game_id, "teamId": det["eventOwnerTeamId"], "is_home": det["eventOwnerTeamId"] == home,
                "goalieId": det.get("goalieInNetId"), "goal": p["typeDescKey"] == "goal",
                "x": det["xCoord"], "y": det["yCoord"], "zone": det.get("zoneCode"),
                "shot_type": det.get("shotType"), "situation": p.get("situationCode"), "t": t,
                "prev_shot_same_team": prev[0] in ANY_SHOT and prev[1] == det["eventOwnerTeamId"],
                "prev_zone": prev[2], "dt_prev": t - prev[3],
            })
        prev = (p["typeDescKey"], det.get("eventOwnerTeamId"), det.get("zoneCode"), t)
    return rows


def season_shots(season, game_ids):
    """Tirs d'une saison, cache disque incrémental (seuls les matchs manquants sont téléchargés)."""
    path = DATA / f"shots_{season}.csv.gz"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["gameId"])
    todo = sorted(set(game_ids) - set(old["gameId"]))
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            new = pd.DataFrame([r for rows in pool.map(game_shots, todo) for r in rows])
        old = pd.concat([old, new], ignore_index=True) if len(old) else new
        old.to_csv(path, index=False)
    return old


def features(s):
    s = s.copy()
    ax = s["x"].abs()
    # ponytail: filet supposé du côté du tir pour les zones O/N ; homeTeamDefendingSide absent avant ~2019
    s["dist"] = np.where(s["zone"] == "D", np.hypot(89 + ax, s["y"]), np.hypot(89 - ax, s["y"]))
    s["angle"] = np.degrees(np.arctan2(s["y"].abs(), (89 - ax).clip(lower=1)))
    s["shot_type"] = s["shot_type"].astype("category").cat.codes
    s["rebound"] = (s["prev_shot_same_team"].astype(bool) & (s["dt_prev"] <= 3)).astype(int)
    s["rush"] = (s["prev_zone"].isin(["N", "D"]) & (s["dt_prev"] <= 4)).astype(int)
    sit = s["situation"].fillna(1551).astype(int).astype(str).str.zfill(4)
    away_g, away_sk, home_sk, home_g = (sit.str[i].astype(int) for i in range(4))
    s["skater_diff"] = np.where(s["is_home"], home_sk - away_sk, away_sk - home_sk)
    s["empty_net"] = np.where(s["is_home"], away_g == 0, home_g == 0).astype(int)
    return s


def score_shots(shots):
    """Ajoute la colonne xg, saison par saison, avec un modèle appris sur les saisons précédentes."""
    s = features(shots)
    s["season"] = s["gameId"] // 1_000_000
    s["xg"] = np.nan
    seasons = sorted(s["season"].unique())
    for i, season in enumerate(seasons):
        train = s[s["season"] < season] if i else s[s["season"] == season]  # 1re saison : rodage, notée sur elle-même
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features=[2])
        m.fit(train[XG_FEATURES], train["goal"].astype(int))
        cur = s["season"] == season
        s.loc[cur, "xg"] = m.predict_proba(s.loc[cur, XG_FEATURES])[:, 1]
    return s


def aggregates(team):
    """(xG par équipe-match, xG subis par gardien-match), à partir de la table équipe-match de nhl.load()."""
    shots = pd.concat([season_shots(season, grp["gameId"].unique()) for season, grp in team.groupby("season")],
                      ignore_index=True)
    s = score_shots(shots)
    xgf = s.groupby(["gameId", "teamId"])["xg"].sum().rename("xgf").reset_index()
    opp = team[["gameId", "teamId"]].merge(team[["gameId", "teamId"]], on="gameId", suffixes=("", "_opp"))
    opp = opp[opp["teamId"] != opp["teamId_opp"]]
    tm = opp.merge(xgf, on=["gameId", "teamId"], how="left").merge(
        xgf.rename(columns={"teamId": "teamId_opp", "xgf": "xga"}), on=["gameId", "teamId_opp"], how="left")
    faced = s[s["empty_net"] == 0].groupby(["gameId", "goalieId"]).agg(
        xg_faced=("xg", "sum"), goals_allowed=("goal", "sum"), fenwick=("xg", "size")).reset_index()
    return tm[["gameId", "teamId", "xgf", "xga"]].fillna(0), faced.rename(columns={"goalieId": "playerId"})
