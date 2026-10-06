"""NHL predictor v2 : forces d'équipe dynamiques (xG, tirs, buts), gardien titulaire (GSAx), fatigue.

    uv run python nhl.py backtest          # walk-forward saison par saison
    uv run python nhl.py predict [AAAA-MM-JJ]

Cible : victoire domicile, prolongation et tirs au but inclus (marché « vainqueur du match »).
"""
import json
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from typesafe_sdk import TypeSafeClient

import starters
import xg
from xg import DATA, WEB, get_json

STATS = "https://api.nhle.com/stats/rest/en"
FIRST_SEASON = 2015         # 2015-16 sert de rodage aux ratings, jamais évaluée
HALFLIFE = 20               # demi-vie (en matchs) des moyennes d'équipe, à calibrer
GOALIE_PRIOR_SHOTS = 1500   # tirs non bloqués « fictifs » à GSAx nul : rétrécit la note des gardiens peu vus
SHOTS_PER_GAME = 40         # tirs non bloqués par match, pour exprimer la note gardien en buts par match
FEATURES = ["d_xg", "d_sat", "d_gd", "d_goalie", "home_b2b", "away_b2b"]


def current_season():
    t = date.today()
    return t.year if t.month >= 9 else t.year - 1


def report(kind, name, season):
    """Une ligne par (match, équipe|gardien) en saison régulière. Cache disque, sauf saison en cours."""
    path = DATA / f"{kind}_{name}_{season}.json"
    if path.exists() and season != current_season():
        return json.loads(path.read_text())
    rows = get_json(f"{STATS}/{kind}/{name}?isAggregate=false&isGame=true&start=0&limit=-1"
                    f"&cayenneExp=seasonId={season}{season + 1}%20and%20gameTypeId=2")["data"]
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def load():
    seasons = range(FIRST_SEASON, current_season() + 1)
    team = pd.concat([
        pd.DataFrame(report("team", "summary", s)).merge(
            pd.DataFrame(report("team", "summaryshooting", s))[["gameId", "teamId", "satFor", "satAgainst"]],
            on=["gameId", "teamId"])
        for s in seasons], ignore_index=True)
    goalies = pd.concat([pd.DataFrame(report("goalie", "summary", s)) for s in seasons], ignore_index=True)
    for df in (team, goalies):
        df["gameDate"] = pd.to_datetime(df["gameDate"])
        df["season"] = df["gameId"] // 1_000_000
    team_xg, goalie_xg = xg.aggregates(team)
    team = team.merge(team_xg, on=["gameId", "teamId"], how="left")
    goalies = goalies.merge(goalie_xg, on=["gameId", "playerId"], how="left")
    goalies[["xg_faced", "goals_allowed", "fenwick"]] = goalies[["xg_faced", "goals_allowed", "fenwick"]].fillna(0)
    return team, goalies


def team_features(team):
    """Features pré-match par (match, équipe), calculées uniquement sur les matchs précédents.
    Les lignes sans stats (matchs à venir) reçoivent l'état après le dernier match joué."""
    # ponytail: moyennes continues d'une saison à l'autre, ajouter une régression vers la moyenne à l'intersaison si besoin
    t = team.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    t["sat_share"] = t["satFor"] / (t["satFor"] + t["satAgainst"])
    t["xg_share"] = t["xgf"] / (t["xgf"] + t["xga"])
    t["gd"] = t["goalsFor"] - t["goalsAgainst"]
    by = t.groupby("teamId")
    for col in ["sat_share", "xg_share", "gd"]:
        t[f"ew_{col}"] = by[col].transform(lambda s: s.ewm(halflife=HALFLIFE).mean().shift())
    t["rest"] = by["gameDate"].diff().dt.days.clip(upper=4).fillna(4)
    return t


def goalie_ratings(goalies):
    """GSAx (xG subis - buts encaissés) par tir non bloqué, cumul carrière rétréci vers 0.
    Retourne (lignes avec note pré-match, note courante par gardien). Lissage testé de 300 à 6000 : sans effet."""
    # ponytail: cumul carrière non pondéré dans le temps, passer à une moyenne exponentielle si la forme récente compte
    g = goalies.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    g["gsax"] = g["xg_faced"] - g["goals_allowed"]
    by = g.groupby("playerId")
    cv, cn = by["gsax"].cumsum(), by["fenwick"].cumsum()
    g["goalie_rating"] = (cv - g["gsax"]) / (cn - g["fenwick"] + GOALIE_PRIOR_SHOTS)
    current = (cv / (cn + GOALIE_PRIOR_SHOTS)).groupby(g["playerId"]).last()
    return g, current


def wide(t):
    """Long (match, équipe) -> une ligne par match, features en différence domicile - extérieur."""
    h, a = (t[t["homeRoad"] == side].set_index("gameId") for side in "HR")
    return pd.DataFrame({
        "date": h["gameDate"], "season": h["season"],
        "home": h["teamFullName"], "away": a["teamFullName"],
        "home_goalie": h["goalieFullName"], "away_goalie": a["goalieFullName"],
        "d_sat": h["ew_sat_share"] - a["ew_sat_share"],
        "d_xg": h["ew_xg_share"] - a["ew_xg_share"],
        "d_gd": h["ew_gd"] - a["ew_gd"],
        "d_goalie": SHOTS_PER_GAME * (h["goalie_rating"] - a["goalie_rating"]),
        "home_b2b": (h["rest"] <= 1).astype(int),
        "away_b2b": (a["rest"] <= 1).astype(int),
        "home_win": h["wins"],
        "reg_tie": 1 - h["winsInRegulation"] - a["winsInRegulation"],  # match décidé en prolongation / tirs au but
    })


def history(team, goalies):
    g, current = goalie_ratings(goalies)
    starters = g[g["gamesStarted"] == 1][["gameId", "homeRoad", "playerId", "goalieFullName", "goalie_rating"]]
    t = team_features(team).merge(starters, on=["gameId", "homeRoad"], how="left")
    return wide(t), current


def fit_predict(train, test, cols):
    if not cols:
        return np.full(len(test), train["home_win"].mean())
    m = LogisticRegression().fit(train[cols], train["home_win"])
    return m.predict_proba(test[cols])[:, 1]


def backtest():
    w, _ = history(*load())
    w = w[(w["season"] > FIRST_SEASON) & (w["season"] < current_season())].dropna(subset=FEATURES + ["home_win"])
    models = {
        "taux_domicile": [],
        "buts_seuls": ["d_gd"],
        "sans_xg": ["d_sat", "d_gd", "d_goalie", "home_b2b", "away_b2b"],
        "sans_gardien": [f for f in FEATURES if f != "d_goalie"],
        "complet": FEATURES,
    }
    rows, per_game = [], {k: [] for k in models}
    for s in sorted(w["season"].unique())[2:]:  # au moins 2 saisons d'entraînement
        train, test = w[w["season"] < s], w[w["season"] == s]
        for name, cols in models.items():
            p = fit_predict(train, test, cols)
            y = test["home_win"].to_numpy()
            per_game[name].append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
            rows.append({"saison": f"{s}-{s + 1 - 2000}", "modèle": name, "n": len(test),
                         "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
                         "accuracy": ((p > 0.5) == y).mean()})
    res = pd.DataFrame(rows)
    print(res.pivot(index="saison", columns="modèle", values="log_loss")[list(models)].round(4).to_string())
    print("\nToutes saisons :")
    print(res.groupby("modèle")[["log_loss", "brier", "accuracy"]].mean().loc[list(models)].round(4).to_string())
    print()
    for new, old in [("complet", "sans_xg"), ("complet", "sans_gardien")]:
        d = np.concatenate(per_game[new]) - np.concatenate(per_game[old])
        print(f"log-loss {new} - {old} : {d.mean():+.4f} ± {1.96 * d.std() / np.sqrt(len(d)):.4f} (IC95)")
    m = LogisticRegression().fit(w[FEATURES], w["home_win"])
    print("\nCoefficients (modèle complet, toutes saisons) :",
          dict(zip(["intercept", *FEATURES], np.round([m.intercept_[0], *m.coef_[0]], 3))))


def probable_starter(goalies, abbrev, before):
    """Heuristique de repli : le gardien le plus titularisé sur les 10 derniers matchs de l'équipe."""
    starts = goalies[(goalies["teamAbbrev"] == abbrev) & (goalies["gamesStarted"] == 1)
                     & (goalies["gameDate"] < pd.Timestamp(before))]
    pid = starts.sort_values("gameDate").tail(10)["playerId"].value_counts().idxmax()
    return pid, starts.loc[starts["playerId"] == pid, "goalieFullName"].iloc[-1]


def predict(day):
    team, goalies = load()
    w, current = history(team, goalies)
    w = w.dropna(subset=FEATURES + ["home_win"])
    model = LogisticRegression().fit(w[FEATURES], w["home_win"])

    week = get_json(f"{WEB}/schedule/{day}")["gameWeek"]
    games = [x for d in week if d["date"] == day for x in d["games"] if x["gameType"] == 2]
    if not games:
        print(f"Aucun match de saison régulière le {day}.")
        return
    client = TypeSafeClient() if os.environ.get("TYPESAFE_API_KEY") else None
    full = lambda tm: f"{tm['placeName']['default']} {tm['commonName']['default']}"
    rows = []
    for x in games:
        kickoff = datetime.fromisoformat(x["startTimeUTC"].replace("Z", "+00:00"))
        for side, key, other in (("H", "homeTeam", "awayTeam"), ("R", "awayTeam", "homeTeam")):
            pid, gname = probable_starter(goalies, x[key]["abbrev"], day)
            if client and kickoff > datetime.now(timezone.utc):  # jamais sur un match commencé : le log sert à l'évaluation
                pid, gname = starters.jev_starter(client, x["id"], x[key]["abbrev"], full(x[key]), full(x[other]),
                                                  kickoff, (pid, gname))
            rows.append({"gameId": x["id"], "gameDate": pd.Timestamp(day), "season": current_season(),
                         "teamId": x[key]["id"], "teamFullName": full(x[key]), "homeRoad": side,
                         "goalieFullName": gname, "goalie_rating": current.get(pid, 0.0)})
    up = pd.DataFrame(rows)
    goalie_cols = ["goalieFullName", "goalie_rating"]
    t = team_features(pd.concat([team, up.drop(columns=goalie_cols)], ignore_index=True))
    t = t[t["gameId"].isin(up["gameId"])].merge(up[["gameId", "homeRoad", *goalie_cols]], on=["gameId", "homeRoad"])
    u = wide(t)
    u["p_home"] = model.predict_proba(u[FEATURES])[:, 1]
    u["cote_juste_dom"] = 1 / u["p_home"]
    u["cote_juste_ext"] = 1 / (1 - u["p_home"])
    print(f"Matchs du {day} — gardien suivi de * = annoncé dans la presse (Jev), sinon heuristique :\n")
    print(u[["away", "away_goalie", "home", "home_goalie", "p_home", "cote_juste_dom", "cote_juste_ext"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        backtest()
    elif cmd == "predict":
        predict(sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat())
    else:
        sys.exit(__doc__)
