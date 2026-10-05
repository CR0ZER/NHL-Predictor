"""Garde-fou anti-fuite : une feature pré-match ne doit dépendre que des matchs précédents.  uv run python test_nhl.py"""
import pandas as pd

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
})
t = team_features(team).set_index(["gameId", "teamId"])
assert t["ew_gd"].loc[(1, 10)] != t["ew_gd"].loc[(1, 10)], "1er match : aucune info -> NaN"
assert t.loc[(2, 10), "ew_gd"] == 4, "2e match : uniquement le 1er (+4)"
assert t.loc[(2, 10), "ew_sat_share"] == 0.6
assert t.loc[(2, 10), "ew_xg_share"] == 0.75
assert -2 < t.loc[(3, 10), "ew_gd"] < 1, "3e match : moyenne de +4 (ancien) et -2 (récent), récent plus lourd"
assert t.loc[(2, 10), "rest"] == 1 and t.loc[(3, 10), "rest"] == 3 and t.loc[(1, 10), "rest"] == 4

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
print("ok")
