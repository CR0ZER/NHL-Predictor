"""Gardien titulaire lu par Jev (TypeSafe) dans les titres de presse d'avant-match.

    uv run python starters.py score    # précision de Jev vs heuristique sur les prédictions loguées, matchs joués

Évaluation prospective uniquement : dans les archives Google News l'heure de publication est arrondie au jour,
un compte rendu d'après-match passerait le filtre « avant le coup d'envoi ». En direct, aucune fuite possible.
"""
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta
from email.utils import parsedate_to_datetime

from typesafe_sdk import Choice, Noul, NoulCriteria

from xg import DATA, WEB, get_json

UNKNOWN = "inconnu"
ANNOUNCED_MIN = 0.5  # seuil « titulaire annoncé » pour préférer Jev à l'heuristique, à recalibrer avec score
LOG = DATA / "jev_starters.jsonl"


def roster_goalies(abbrev):
    r = get_json(f"{WEB}/roster/{abbrev}/current")
    return {f"{g['firstName']['default']} {g['lastName']['default']}": g["id"] for g in r["goalies"]}


def headlines(team_full, kickoff):
    """Titres Google News des 36 h précédant le coup d'envoi (kickoff : datetime UTC)."""
    # ponytail: titres seuls (les liens Google News sont des redirections JS), lire les articles si les titres ne suffisent pas
    q = f'"{team_full}" (goalie OR goaltender OR start OR starter) when:2d'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    items = [(parsedate_to_datetime(i.findtext("pubDate")), i.findtext("title")) for i in root.iter("item")]
    return sorted(((d, t) for d, t in items if kickoff - timedelta(hours=36) <= d < kickoff), reverse=True)[:25]


def jev_starter(client, game_id, abbrev, team, opponent, kickoff, heuristic):
    """heuristic = (playerId, nom) de repli -> (playerId, nom) retenu : celui annoncé dans la presse selon Jev
    (nom suffixé de « * »), sinon l'heuristique. Chaque appel est logué pour l'évaluation prospective."""
    heuristic_pid = heuristic[0]
    goalies = roster_goalies(abbrev)
    items = headlines(team, kickoff)
    choice, p, announced = UNKNOWN, 1.0, 0.0
    if items:
        state = {"game": {"team": team, "opponent": opponent, "kickoff_utc": kickoff.isoformat()},
                 "headlines": [{"published_utc": d.isoformat(), "title": t} for d, t in items]}
        r = client.system_one(state, {
            "starter": Choice(
                instructions="According to `headlines`, who will start in goal for `game.team` against "
                             "`game.opponent` in the game at `game.kickoff_utc`? Headlines about earlier games "
                             "or about the opponent's goalie do not count.",
                criteria={name: None for name in goalies}
                         | {UNKNOWN: "No headline indicates who starts in goal for `game.team` in this game"}),
            "announced": Noul(
                instructions="Do `headlines` report who will start in goal for `game.team` in this game against "
                             "`game.opponent` (named, confirmed, announced or expected)?",
                criteria=NoulCriteria(true="A headline names this game's starting goalie for `game.team`",
                                      false="No headline names it")),
        }, model="jev-latest")
        c = r.choices["starter"]
        choice, p, announced = c.choice, c.probabilities[c.choice], r.nouls["announced"].noul
    jev_pid = goalies.get(choice)
    use_jev = jev_pid is not None and announced >= ANNOUNCED_MIN
    DATA.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"gameId": game_id, "team": abbrev, "kickoff": kickoff.isoformat(), "choice": choice,
                            "jev_pid": jev_pid, "p": p, "announced": announced, "heur_pid": int(heuristic_pid),
                            "headlines": [t for _, t in items]}) + "\n")
    return (jev_pid, f"{choice} *") if use_jev else heuristic


def score():
    import nhl
    import pandas as pd

    log = pd.read_json(LOG, lines=True).drop_duplicates(["gameId", "team"], keep="last")
    _, goalies = nhl.load()
    actual = goalies[goalies["gamesStarted"] == 1][["gameId", "teamAbbrev", "playerId"]]
    r = log.merge(actual, left_on=["gameId", "team"], right_on=["gameId", "teamAbbrev"])
    if r.empty:
        sys.exit("Aucun match logué n'est encore joué.")
    used = (r["jev_pid"].notna()) & (r["announced"] >= ANNOUNCED_MIN)
    final = r["jev_pid"].where(used, r["heur_pid"])
    print(f"{len(r)} équipes-matchs joués")
    print(f"heuristique seule         : {(r['heur_pid'] == r['playerId']).mean():.1%}")
    print(f"Jev utilisé               : {used.mean():.1%} des cas, juste à {(r.loc[used, 'jev_pid'] == r.loc[used, 'playerId']).mean():.1%}")
    print(f"combiné (Jev sinon heur.) : {(final == r['playerId']).mean():.1%}")
    miss = r[final != r["playerId"]]
    if len(miss):
        print("\nErreurs :")
        print(miss[["gameId", "team", "choice", "announced"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)
