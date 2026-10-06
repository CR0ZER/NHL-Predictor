"""NHL predictor v3 : forces d'équipe dynamiques (xG, tirs, buts), gardien titulaire (GSAx), fatigue, absences.

    uv run python nhl.py backtest          # walk-forward saison par saison
    uv run python nhl.py predict [AAAA-MM-JJ]

Cible : victoire domicile, prolongation et tirs au but inclus (marché « vainqueur du match »).
"""
import json
import os
import sys
import urllib.parse
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from typesafe_sdk import TypeSafeClient

import xg
from xg import DATA, WEB, get_json

STATS = "https://api.nhle.com/stats/rest/en"
FIRST_SEASON = 2015         # 2015-16 sert de rodage aux ratings, jamais évaluée
HALFLIFE = 20               # demi-vie (en matchs) des moyennes d'équipe, à calibrer
OFFSEASON_GAMES = 5         # matchs fictifs « moyens » à chaque intersaison ; testé 0-80 : 5-10 optimal, gain minime
GOALIE_PRIOR_SHOTS = 1500   # tirs non bloqués « fictifs » à GSAx nul : rétrécit la note des gardiens peu vus
SHOTS_PER_GAME = 40         # tirs non bloqués par match, pour exprimer la note gardien en buts par match
FEATURES = ["d_xg", "d_sat", "d_gd", "d_goalie", "home_b2b", "away_b2b", "d_miss"]


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


def skater_games(season):
    """Une ligne par (match, joueur de champ) : qui a joué et son temps de glace. Par quinzaine : l'API plafonne à
    10 000 lignes par réponse. Cache disque, sauf saison en cours."""
    path = DATA / f"skater_games_{season}.json"
    if path.exists() and season != current_season():
        return pd.read_json(path)
    rows, starts = [], pd.date_range(f"{season}-09-01", f"{season + 1}-07-01", freq="SMS")
    for a, b in zip(starts[:-1], starts[1:]):
        exp = (f'seasonId={season}{season + 1} and gameTypeId=2 and gameDate>="{a.date()}" and gameDate<"{b.date()}"')
        rows += get_json(f"{STATS}/skater/summary?isAggregate=false&isGame=true&start=0&limit=-1&cayenneExp="
                         + urllib.parse.quote(exp))["data"]
    df = pd.DataFrame(rows)[["gameId", "gameDate", "playerId", "skaterFullName", "teamAbbrev", "homeRoad",
                             "positionCode", "timeOnIcePerGame", "points"]]
    DATA.mkdir(exist_ok=True)
    df.to_json(path)
    return df


_players = {}


def players():
    """Une ligne par (match, joueur de champ) avec ses points par match sur ses 60 matchs précédents (toutes équipes),
    rétrécis vers 0. Recalculé une fois par jour."""
    key = date.today().isoformat()
    if key not in _players:
        sk = pd.concat([skater_games(s) for s in range(FIRST_SEASON, current_season() + 1)], ignore_index=True)
        sk["gameDate"] = pd.to_datetime(sk["gameDate"])
        sk = sk.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
        by = sk.groupby("playerId")["points"]
        pts = by.transform(lambda x: x.rolling(60, min_periods=1).sum().shift().fillna(0))
        n = by.transform(lambda x: x.rolling(60, min_periods=1).count().shift().fillna(0))
        _players.clear()
        _players[key] = sk.assign(ppg=pts / (n + 5))
    return _players[key]


def actual_missing(sk):
    """Absences réelles (entraînement) : points par match habituels des joueurs ayant joué pour l'équipe lors de ses
    10 matchs précédents de la saison, mais absents de ce match. En direct, la même grandeur vient de qwen + Jev."""
    rows, idx = [], ["gameDate", "gameId", "homeRoad"]
    for _, g in sk.assign(season=sk["gameId"] // 1_000_000).groupby(["teamAbbrev", "season"]):
        played = g.pivot_table(index=idx, columns="playerId", values="points", aggfunc="size", fill_value=0).sort_index() > 0
        regular = played.astype(int).rolling(10, min_periods=1).max().shift().fillna(0) > 0
        ppg = g.pivot_table(index=idx, columns="playerId", values="ppg").reindex_like(played).ffill()
        rows.append(ppg.where(regular & ~played, 0).sum(axis=1).rename("miss").reset_index())
    return pd.concat(rows)[["gameId", "homeRoad", "miss"]]


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
    team_xg, goalie_xg = xg.aggregates(team, current_season())
    team = team.merge(team_xg, on=["gameId", "teamId"], how="left")
    goalies = goalies.merge(goalie_xg, on=["gameId", "playerId"], how="left")
    goalies[["xg_faced", "goals_allowed", "fenwick"]] = goalies[["xg_faced", "goals_allowed", "fenwick"]].fillna(0)
    return team, goalies


def team_features(team):
    """Features pré-match par (match, équipe), calculées uniquement sur les matchs précédents.
    Les lignes sans stats (matchs à venir) reçoivent l'état après le dernier match joué."""
    t = team.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    t["sat_share"] = t["satFor"] / (t["satFor"] + t["satAgainst"])
    t["xg_share"] = t["xgf"] / (t["xgf"] + t["xga"])
    t["gd"] = t["goalsFor"] - t["goalsAgainst"]
    t["rest"] = t.groupby("teamId")["gameDate"].diff().dt.days.clip(upper=4).fillna(4)
    # Intersaison : OFFSEASON_GAMES matchs fictifs « moyens » avant le 1er match de chaque nouvelle saison,
    # pour ramener les forces vers la moyenne (transferts, vieillissement... que le modèle ne voit pas).
    first = t[t.groupby("teamId")["season"].diff() > 0]
    pseudo = first.loc[first.index.repeat(OFFSEASON_GAMES), ["teamId", "gameDate", "season"]].assign(
        sat_share=0.5, xg_share=0.5, gd=0.0, pseudo=True)
    u = pd.concat([pseudo, t.assign(pseudo=False)]).sort_values(["gameDate", "pseudo", "gameId"],
                                                                 ascending=[True, False, True], kind="stable")
    by = u.groupby("teamId")
    for col in ["sat_share", "xg_share", "gd"]:
        u[f"ew_{col}"] = by[col].transform(lambda s: s.ewm(halflife=HALFLIFE).mean().shift())
    u = u[~u["pseudo"]].drop(columns="pseudo").astype({"gameId": "int64"})  # les lignes fictives n'ont pas de gameId
    return u.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


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
        "d_miss": h["miss"] - a["miss"],
        "home_b2b": (h["rest"] <= 1).astype(int),
        "away_b2b": (a["rest"] <= 1).astype(int),
        "home_win": h["wins"],
        "reg_tie": 1 - h["winsInRegulation"] - a["winsInRegulation"],  # match décidé en prolongation / tirs au but
    })


def history(team, goalies):
    g, current = goalie_ratings(goalies)
    starters = g[g["gamesStarted"] == 1][["gameId", "homeRoad", "playerId", "goalieFullName", "goalie_rating"]]
    t = team_features(team).merge(starters, on=["gameId", "homeRoad"], how="left")
    t = t.merge(actual_missing(players()), on=["gameId", "homeRoad"], how="left").fillna({"miss": 0.0})
    return wide(t), current


def fit_predict(train, test, cols):
    if not cols:
        return np.full(len(test), train["home_win"].mean())
    m = LogisticRegression().fit(train[cols], train["home_win"])
    return m.predict_proba(test[cols])[:, 1]


def walk_forward(cols=FEATURES):
    """Prédiction hors échantillon de chaque match (saison S prédite par un modèle appris sur les saisons < S)."""
    w, _ = history(*load())
    w = w[w["season"] > FIRST_SEASON].dropna(subset=cols + ["home_win"])
    return pd.concat([w[w["season"] == s].assign(p=fit_predict(w[w["season"] < s], w[w["season"] == s], cols))
                      for s in sorted(w["season"].unique())[2:]]).reset_index()


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


def predictions(day, use_news=True, progress=print):
    """Matchs du jour non encore joués : probabilités (vainqueur et 1N2 temps réglementaire), gardiens et absences.
    Chaîne d'actualité (NHL.com -> qwen3 -> Jev) pour chaque match pas encore commencé ; repli sur l'heuristique du
    gardien et aucune absence si Ollama ou Jev sont indisponibles."""
    import news  # import local : news dépend de nhl

    team, goalies = load()
    w, current = history(team, goalies)
    w = w.dropna(subset=FEATURES + ["home_win"])
    model = LogisticRegression().fit(w[FEATURES], w["home_win"])

    week = get_json(f"{WEB}/schedule/{day}")["gameWeek"]
    played = set(team["gameId"])
    games = [x for d in week if d["date"] == day for x in d["games"] if x["gameType"] == 2 and x["id"] not in played]
    if not games:
        return pd.DataFrame()
    chain = use_news and news.ready()
    if use_news and not chain:
        progress("Ollama (qwen3:8b) ou Jev indisponible : gardien deviné, sans absences.")
    client, news_ok = (TypeSafeClient() if chain else None), {}
    sk = players()
    ppg_now = sk[sk["gameDate"] < pd.Timestamp(day)].groupby("playerId")["ppg"].last()
    news.fresh()  # listes d'articles relues à chaque prédiction
    full = lambda tm: f"{tm['placeName']['default']} {tm['commonName']['default']}"
    rows = []
    for i, x in enumerate(games):
        kickoff = datetime.fromisoformat(x["startTimeUTC"].replace("Z", "+00:00"))
        abbr = {s: x[f"{s}Team"]["abbrev"] for s in ("away", "home")}
        res = None
        if chain and kickoff > datetime.now(timezone.utc):  # jamais sur un match commencé
            progress(f"Actualité {i + 1}/{len(games)} : qwen lit les articles, Jev tranche ({abbr['away']} @ {abbr['home']})")
            g = {"gameId": x["id"], "kickoff": x["startTimeUTC"], "away": full(x["awayTeam"]), "home": full(x["homeTeam"]),
                 **{f"{s}_id": x[f"{s}Team"]["id"] for s in ("away", "home")},
                 **{f"{s}_abbr": abbr[s] for s in ("away", "home")}}
            res = news.run_game(client, g, {ab: news.roster_goalies(ab) for ab in abbr.values()}, refresh=True)
        news_ok[x["id"]] = res is not None
        for side, s in (("H", "home"), ("R", "away")):
            pid, gname = probable_starter(goalies, abbr[s], day)
            miss, absents, summary = 0.0, [], ""
            if res:
                j = res["jev"][s]
                jev_pid = news.roster_goalies(abbr[s]).get(j["starter"])
                if jev_pid and j["announced"] >= 0.5:
                    pid, gname = jev_pid, f"{j['starter']} *"
                miss, absents = news.expected_missing(j["p_out"], abbr[s], sk, ppg_now)
                summary = res["qwen"][s]["summary"]
            rows.append({"gameId": x["id"], "gameDate": pd.Timestamp(day), "season": current_season(),
                         "teamId": x[f"{s}Team"]["id"], "teamFullName": full(x[f"{s}Team"]), "homeRoad": side,
                         "goalieFullName": gname, "goalie_rating": current.get(pid, 0.0), "miss": miss,
                         "absents": json.dumps(absents, ensure_ascii=False), "summary": summary})
    up = pd.DataFrame(rows)
    goalie_cols = ["goalieFullName", "goalie_rating", "miss", "absents", "summary"]
    t = team_features(pd.concat([team, up.drop(columns=goalie_cols)], ignore_index=True))
    t = t[t["gameId"].isin(up["gameId"])].merge(up[["gameId", "homeRoad", *goalie_cols]], on=["gameId", "homeRoad"])
    u = wide(t)
    u["p_home"] = model.predict_proba(u[FEATURES])[:, 1]
    # ponytail: P(prolongation) constante, prolongation à pile ou face (même hypothèse que odds.model_probs)
    tie = w["reg_tie"].mean()
    u["pH"], u["pX"], u["pA"] = (u["p_home"] - tie / 2).clip(0.01), tie, (1 - u["p_home"] - tie / 2).clip(0.01)
    u["kickoff"] = u.index.map({x["id"]: x["startTimeUTC"] for x in games})
    u["news"] = u.index.map(news_ok)
    for side, s in (("H", "home"), ("R", "away")):
        tt = t[t["homeRoad"] == side].set_index("gameId")
        u[f"{s}_absents"], u[f"{s}_summary"], u[f"{s}_miss"] = tt["absents"], tt["summary"], tt["miss"]
    return u.reset_index()


def predict(day):
    u = predictions(day)
    if u.empty:
        return print(f"Aucun match de saison régulière à venir le {day}.")
    u["cote_juste_dom"] = 1 / u["p_home"]
    u["cote_juste_ext"] = 1 / (1 - u["p_home"])
    print(f"Matchs du {day} — gardien suivi de * = identifié par Jev dans la presse, sinon heuristique :\n")
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
