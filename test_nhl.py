"""Garde-fou anti-fuite : une feature pré-match ne doit dépendre que des matchs précédents.  uv run python test_nhl.py"""
import pandas as pd

import nhl
from nhl import goalie_ratings, team_features

team = pd.DataFrame({
    "gameId": [1, 1, 2, 2, 3, 3],
    "gameDate": pd.to_datetime(["2024-10-01"] * 2 + ["2024-10-02"] * 2 + ["2024-10-05"] * 2),
    "teamId": [10, 20] * 3,
    "goalsFor": [5, 1, 0, 2, 3, 3],
    "goalsAgainst": [1, 5, 2, 0, 3, 3],
    "satFor": [60, 40, 50, 50, 70, 30],
    "satAgainst": [40, 60, 50, 50, 30, 70],
    "xgf": [3.0, 1.0, 2.0, 2.0, 2.5, 1.5],
    "xga": [1.0, 3.0, 2.0, 2.0, 1.5, 2.5],
    "season": 2024,
})
t = team_features(team).set_index(["gameId", "teamId"])
assert t["ew_gd"].loc[(1, 10)] != t["ew_gd"].loc[(1, 10)], "1er match : aucune info -> NaN"
assert t.loc[(2, 10), "ew_gd"] == 4, "2e match : uniquement le 1er (+4)"
assert t.loc[(2, 10), "ew_sat_share"] == 0.6
assert t.loc[(2, 10), "ew_xg_share"] == 0.75
assert -2 < t.loc[(3, 10), "ew_gd"] < 1, "3e match : moyenne de +4 (ancien) et -2 (récent), récent plus lourd"
assert t.loc[(2, 10), "rest"] == 1 and t.loc[(3, 10), "rest"] == 3 and t.loc[(1, 10), "rest"] == 4

nxt = team[team["gameId"] == 1].assign(gameId=4, gameDate=pd.Timestamp("2025-10-01"), season=2025)
two_seasons = pd.concat([team, nxt])
before = team_features(two_seasons).set_index(["gameId", "teamId"]).loc[(4, 10), "ew_gd"]
nhl.OFFSEASON_GAMES = 10
after = team_features(two_seasons).set_index(["gameId", "teamId"]).loc[(4, 10), "ew_gd"]
nhl.OFFSEASON_GAMES = 0
assert 0 < after < before, "intersaison : la forme de la saison passée est ramenée vers la moyenne (0)"
goalies = pd.DataFrame({
    "gameId": [1, 2, 3], "gameDate": pd.to_datetime(["2024-10-01", "2024-10-02", "2024-10-05"]),
    "season": 2024, "playerId": 7,
    "xg_faced": [3.0, 3.0, 3.0], "goals_allowed": [1, 0, 10], "fenwick": [40, 40, 40],
})
g, current = goalie_ratings(goalies)
r = g["goalie_rating"]
assert abs(r.iloc[0]) < 1e-12, "1re apparition : note neutre"
assert r.iloc[1] > 0, "après un bon match (1 but pour 3 xG) : note positive"
assert r.iloc[2] > r.iloc[1], "encore un bon match : la note monte"
assert current[7] < r.iloc[2], "le mauvais dernier match (10 buts) n'entre que dans la note courante"

# Qualité des prédictions réelles : CLV = cote prise / cote de clôture du même bookmaker, marché et issue − 1
import tempfile
from pathlib import Path

import app
import odds

odds.CLOSING = Path(tempfile.mkdtemp()) / "closing.csv"
odds.CLOSING.write_text("fixtureId,home,date,book,market,side,price\nx,tampa bay lightning,2026-10-05,unibet.fr,151,H,2.0\n")
q = app.quality(pd.DataFrame([{"date": "2026-10-05", "p_home": 0.6, "pin_fair": 0.5, "result": "H"}]),
                pd.DataFrame([{"date": "2026-10-05", "home": "Tampa Bay Lightning", "kickoff": "2026-10-06T00:30:00Z",
                               "book": "Unibet FR", "market": "Vainqueur", "pick": "H", "price": 2.2, "status": "gagné"}]))["total"]
assert abs(q["clv"] - 0.1) < 1e-9 and q["n_clv"] == 1, "pari à 2,20 contre une clôture à 2,00 : +10 %"
assert q["ll_model_pin"] < q["ll_pin"], "domicile gagnant : 60 % (modèle) fait mieux que 50 % (Pinnacle)"
print("ok")
