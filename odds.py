"""Cotes NHL via OddsPapi : historique Pinnacle / Winamax / Unibet FR, et évaluation du modèle face au marché.

    uv run --env-file .env python odds.py update                        # saison en cours, à lancer ~1 fois par semaine
    uv run --env-file .env python odds.py fetch AAAA-MM-JJ AAAA-MM-JJ   # une période passée précise
    uv run python odds.py eval [SAISON]                                 # défaut 2025 (2025-26) ; 2026 = saison en cours

Chaque update/fetch : 1 requête décomptée (liste des matchs) + historiques gratuits, ~6 s par nouveau match terminé
(une semaine de NHL ≈ 50 matchs ≈ 5 min).

Quota gratuit : 250 requêtes/mois ; /historical-odds n'est pas décompté (mais 5 s d'attente entre appels).
"""
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from xg import DATA

API = "https://api.oddspapi.io/v4"
DIR = DATA / "oddspapi"
CLOSING = DIR / "closing.csv"  # cotes de clôture résumées (versionné) ; les historiques bruts restent en local
BOOKS = "pinnacle,winamax.fr,unibet.fr"  # 3 max par appel ; fdj et zebet.fr sont des clones d'unibet.fr
MONEYLINE, REGULATION = "151", "153"     # vainqueur prolongation incluse (1, 2) ; 1N2 temps réglementaire (1, X, 2)
OUTCOMES = {MONEYLINE: {"151": "H", "152": "A"}, REGULATION: {"153": "H", "154": "X", "155": "A"}}


def api(endpoint, **params):
    url = f"{API}/{endpoint}?" + urllib.parse.urlencode({**params, "apiKey": os.environ["ODDSPAPI_API_KEY"]})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-predictor"}),
                                    timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:  # « No historical odds found »
            return {}
        raise


def fixtures(start, end, refresh=False):
    path = DIR / f"fixtures_{start}_{end}.json"
    if refresh or not path.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(api("fixtures", tournamentId=234, **{"from": f"{start}T00:00:00Z",
                                                                        "to": f"{end}T00:00:00Z"})))
    return json.loads(path.read_text())


def fetch(start, end, refresh=False):
    (DIR / "hist").mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(CLOSING, usecols=["fixtureId"])["fixtureId"]) if CLOSING.exists() else set()  # déjà résumés
    todo = [f for f in fixtures(start, end, refresh) if f["statusId"] == 2 and f["fixtureId"] not in done
            and not (DIR / "hist" / f"{f['fixtureId']}.json").exists()]
    for i, f in enumerate(todo):
        (DIR / "hist" / f"{f['fixtureId']}.json").write_text(json.dumps(api("historical-odds", fixtureId=f["fixtureId"],
                                                                            bookmakers=BOOKS)))
        if i % 50 == 0:
            print(f"{i}/{len(todo)}", flush=True)
        time.sleep(5.5)  # cooldown de l'endpoint : 5000 ms
    print(f"{len(todo)} nouveaux matchs terminés récupérés")


def update():
    """Saison en cours : liste des matchs rafraîchie (fichier unique par saison, statuts à jour) + nouveaux historiques."""
    t = date.today()
    season = t.year if t.month >= 9 else t.year - 1
    fetch(f"{season}-09-01", f"{season + 1}-07-01", refresh=True)
    closing_odds()  # résume les nouveaux matchs dans closing.csv


def norm(name):
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()


def last_prices(hist, before):
    """{(bookmaker, marché, issue): dernière cote publiée avant `before`} à partir d'une réponse historical-odds."""
    out = {}
    for book, b in hist.get("bookmakers", {}).items():
        for market, outcomes in OUTCOMES.items():
            for oid, side in outcomes.items():
                prices = b.get("markets", {}).get(market, {}).get("outcomes", {}).get(oid, {}).get("players", {})
                pre = [o for o in next(iter(prices.values()), []) if pd.Timestamp(o["createdAt"]) < before]
                if pre:
                    out[(book, market, side)] = max(pre, key=lambda o: o["createdAt"])["price"]
    return out


def local_date(ts):
    return (pd.Timestamp(ts) - pd.Timedelta(hours=8)).date()  # date locale nord-américaine


def closing_odds():
    """Dernière cote publiée avant le coup d'envoi réel, par (match, bookmaker, marché, issue).
    Les historiques bruts (~2,5 Mo par match, data/oddspapi/hist) sont résumés dans closing.csv, seul fichier conservé
    dans le dépôt : un match déjà résumé n'a plus besoin de son historique brut."""
    cols = ["fixtureId", "home", "date", "book", "market", "side", "price"]
    old = pd.read_csv(CLOSING, dtype={"market": str}) if CLOSING.exists() else pd.DataFrame(columns=cols)
    done, rows = set(old["fixtureId"]), []
    for path in DIR.glob("fixtures_*.json"):
        for f in json.loads(path.read_text()):
            hist = DIR / "hist" / f"{f['fixtureId']}.json"
            if f["fixtureId"] in done or not hist.exists():
                continue
            kickoff = pd.Timestamp(f["trueStartTime"] or f["startTime"])
            for (book, market, side), price in last_prices(json.loads(hist.read_text()), kickoff).items():
                rows.append({"fixtureId": f["fixtureId"], "home": norm(f["participant1Name"]),
                             "date": local_date(kickoff), "book": book, "market": market, "side": side, "price": price})
    out = pd.concat([old, pd.DataFrame(rows, columns=cols)], ignore_index=True) if rows else old
    if rows:
        out.to_csv(CLOSING, index=False)
    return out.assign(date=pd.to_datetime(out["date"]).dt.date)


def current_odds(home, kickoff):
    """Cotes actuelles d'un match à venir (endpoint historique : gratuit, ~5 s d'attente entre deux appels)."""
    t = date.today()
    season = t.year if t.month >= 9 else t.year - 1
    ids = {(norm(f["participant1Name"]), local_date(f["startTime"])): f["fixtureId"]
           for f in fixtures(f"{season}-09-01", f"{season + 1}-07-01")}
    fid = ids.get((norm(home), local_date(kickoff)))
    if fid is None:
        return {}
    time.sleep(5.5)  # cooldown de l'endpoint
    return last_prices(api("historical-odds", fixtureId=fid, bookmakers=BOOKS), pd.Timestamp(kickoff))


def model_probs(season):
    """Probabilités walk-forward (modèle appris sur les saisons < season) : vainqueur et 1N2 temps réglementaire."""
    import nhl

    w, _ = nhl.history(*nhl.load())
    w = w.dropna(subset=nhl.FEATURES + ["home_win"]).reset_index()
    train, test = w[w["season"] < season], w[w["season"] == season].copy()
    test["p"] = nhl.fit_predict(train, test, nhl.FEATURES)
    # ponytail: P(prolongation) constante et prolongation à pile ou face ; modéliser si le 1N2 devient le marché joué
    tie = train["reg_tie"].mean()
    test["pH"], test["pX"], test["pA"] = (test["p"] - tie / 2).clip(0.01), tie, (1 - test["p"] - tie / 2).clip(0.01)
    test["result3"] = np.where(test["reg_tie"] == 1, "X", np.where(test["home_win"] == 1, "H", "A"))
    test["home"], test["date"] = test["home"].map(norm), test["date"].dt.date
    return test


def ci(x):
    return f"{x.mean():+.4f} ± {1.96 * x.std() / np.sqrt(len(x)):.4f}"


def evaluate(season=2025):
    games = model_probs(season)
    odds = closing_odds().pivot_table(index=["home", "date", "book", "market"], columns="side",
                                      values="price").reset_index()
    m = odds.merge(games, on=["home", "date"])
    print(f"{games['gameId'].nunique()} matchs modélisés, {m['gameId'].nunique()} avec des cotes. Couverture :")
    print(m.groupby(["book", "market"])["gameId"].nunique().to_string(), "\n")

    # 1. Vainqueur (prolongation incluse) : modèle vs Pinnacle, et le modèle apporte-t-il quelque chose au marché ?
    pin = m[(m["book"] == "pinnacle") & (m["market"] == MONEYLINE)].dropna(subset=["H", "A"]).copy()
    pin["q"] = (1 / pin["H"]) / (1 / pin["H"] + 1 / pin["A"])
    y = pin["home_win"]
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"Vainqueur, {len(pin)} matchs — log-loss modèle {ll(pin['p']).mean():.4f} | Pinnacle clôture {ll(pin['q']).mean():.4f}"
          f" | écart modèle - Pinnacle {ci(ll(pin['p']) - ll(pin['q']))}")
    if len(pin) < 100:
        return print("Moins de 100 matchs avec cotes Pinnacle : relancer update plus tard.")
    logit = lambda p: np.log(p / (1 - p))
    X = np.column_stack([logit(pin["q"]), logit(pin["p"])])
    combo = cross_val_predict(LogisticRegression(), X, y, cv=5, method="predict_proba")[:, 1]
    coef = LogisticRegression().fit(X, y).coef_[0]
    print(f"Combinaison Benter (CV 5 plis) : log-loss {ll(combo).mean():.4f}, gain vs Pinnacle {ci(ll(combo) - ll(pin['q']))} ;"
          f" poids Pinnacle {coef[0]:.2f}, poids modèle {coef[1]:.2f}\n")

    # 2. 1N2 temps réglementaire chez les opérateurs français : paris à valeur selon le modèle, et selon Pinnacle
    pin3 = m[(m["book"] == "pinnacle") & (m["market"] == REGULATION)].set_index("gameId")
    for book in ["winamax.fr", "unibet.fr"]:
        b = m[(m["book"] == book) & (m["market"] == REGULATION)].dropna(subset=["H", "X", "A"]).set_index("gameId")
        if b.empty:
            continue
        margin = (1 / b["H"] + 1 / b["X"] + 1 / b["A"]).mean() - 1
        print(f"{book} 1N2, {len(b)} matchs, marge moyenne {margin:.1%}")
        bets = pd.concat([pd.DataFrame({"price": b[s], "p": b[f"p{s}"], "won": b["result3"] == s,
                                        "fair_pin": (1 / pin3[s]) / (1 / pin3["H"] + 1 / pin3["X"] + 1 / pin3["A"])
                                        if not pin3.empty else np.nan}) for s in "HXA"])
        bets["profit"] = np.where(bets["won"], bets["price"] - 1, -1.0)
        for name, sel in [("modèle, valeur ≥ 0 %", bets["p"] * bets["price"] > 1),
                          ("modèle, valeur ≥ 5 %", bets["p"] * bets["price"] > 1.05),
                          ("Pinnacle, valeur ≥ 2 %", bets["fair_pin"] * bets["price"] > 1.02)]:
            s = bets[sel]
            if len(s):
                print(f"  {name:24s}: {len(s):4d} paris, ROI {ci(s['profit'])} par mise")
        print()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "update":
        update()
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 2025)
    else:
        sys.exit(__doc__)
