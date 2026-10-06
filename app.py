"""Interface locale : prédictions du soir, simulation de paris à mise fixe (fictifs), suivi du modèle.

    uv run --env-file .env python app.py        # puis http://127.0.0.1:8765

Paris fictifs enregistrés dans data/paper_bets.csv à la cote disponible au moment de la prédiction,
réglés ensuite avec les résultats officiels NHL. Rien n'est jamais misé réellement.
"""
import json
import threading
import traceback
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import nhl
import odds
from xg import DATA, WEB, get_json

ROOT = Path(__file__).parent
BETS, PREDS = DATA / "paper_bets.csv", DATA / "predictions.csv"
PORT = 8765
VALUE_MIN = 0.0  # avantage minimal (p_modèle × cote - 1) pour les stratégies « valeur »
STRATEGIES = {
    "favori": "Favori du modèle · Unibet vainqueur",
    "valeur_unibet": "Valeur · Unibet vainqueur",
    "valeur_winamax": "Valeur · Winamax 1N2",
}
JOB = {"busy": False, "msg": "", "error": None, "done_at": None}
MODEL_CACHE = {}


def et_today():
    return (datetime.now(timezone.utc) - timedelta(hours=5)).date().isoformat()  # date du jour côté NHL (heure de l'Est)


def read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def records(df):
    return json.loads(df.to_json(orient="records")) if len(df) else []


def clean(x):
    """NaN -> null, types numpy -> types Python (JSON valide pour le navigateur)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return int(x) if isinstance(x, np.integer) else x


def run_predict(day, stake, use_news):
    JOB["msg"] = "Chargement du modèle…"
    u = nhl.predictions(day, use_news=use_news, progress=lambda m: JOB.update(msg=m))
    if u.empty:
        JOB["msg"] = f"Aucun match à venir le {day}."
        return
    now = pd.Timestamp.now(tz="UTC").isoformat()
    preds, bets = [], []
    for i, g in u.reset_index(drop=True).iterrows():
        JOB["msg"] = f"Cotes {i + 1}/{len(u)} : {g.away} @ {g.home}"
        o = odds.current_odds(g.home, g.kickoff)
        c = lambda book, market, side: o.get((book, market, side))
        pin_h, pin_a = c("pinnacle", "151", "H"), c("pinnacle", "151", "A")
        base = {"gameId": int(g.gameId), "date": day, "kickoff": g.kickoff, "away": g.away, "home": g.home}
        preds.append({**base, "placed_at": now, "away_goalie": g.away_goalie, "home_goalie": g.home_goalie,
                      "p_home": g.p_home, "pH": g.pH, "pX": g.pX, "pA": g.pA, "news": bool(g.news),
                      "away_absents": g.away_absents, "home_absents": g.home_absents,
                      "away_summary": g.away_summary, "home_summary": g.home_summary,
                      "away_miss": g.away_miss, "home_miss": g.home_miss,
                      "pin_fair": (1 / pin_h) / (1 / pin_h + 1 / pin_a) if pin_h and pin_a else None,
                      "pin_H": pin_h, "pin_A": pin_a,
                      "uni_H": c("unibet.fr", "151", "H"), "uni_A": c("unibet.fr", "151", "A"),
                      "wmx_H": c("winamax.fr", "153", "H"), "wmx_X": c("winamax.fr", "153", "X"),
                      "wmx_A": c("winamax.fr", "153", "A")})
        label = {"H": g.home, "A": g.away, "X": "Nul après 60 min"}

        def bet(strategy, market, book, side, price, p):
            bets.append({**base, "placed_at": now, "strategy": strategy, "market": market, "book": book, "pick": side,
                         "pick_label": label[side], "price": price, "p_model": p, "ev": p * price - 1,
                         "stake": stake, "status": "en attente", "profit": None, "score": None})

        uni = {"H": (c("unibet.fr", "151", "H"), g.p_home), "A": (c("unibet.fr", "151", "A"), 1 - g.p_home)}
        if all(price for price, _ in uni.values()):
            fav = "H" if g.p_home >= 0.5 else "A"
            bet("favori", "Vainqueur", "Unibet FR", fav, *uni[fav])
            side = max(uni, key=lambda s: uni[s][1] * uni[s][0])
            if uni[side][1] * uni[side][0] - 1 > VALUE_MIN:
                bet("valeur_unibet", "Vainqueur", "Unibet FR", side, *uni[side])
        wmx = {s: (c("winamax.fr", "153", s), g[f"p{s}"]) for s in "HXA"}
        if all(price for price, _ in wmx.values()):
            side = max(wmx, key=lambda s: wmx[s][1] * wmx[s][0])
            if wmx[side][1] * wmx[side][0] - 1 > VALUE_MIN:
                bet("valeur_winamax", "1N2", "Winamax", side, *wmx[side])

    # On remplace les prédictions / paris encore modifiables (match pas commencé) des mêmes matchs
    started = lambda df: pd.to_datetime(df["kickoff"], utc=True) <= pd.Timestamp.now(tz="UTC")
    ids = {p["gameId"] for p in preds}
    old_b, old_p = read(BETS), read(PREDS)
    if len(old_b):
        old_b = old_b[~old_b["gameId"].isin(ids) | (old_b["status"] != "en attente") | started(old_b)]
    if len(old_p):
        old_p = old_p[~old_p["gameId"].isin(ids) | started(old_p)]
    new_b = pd.DataFrame(bets)
    if len(old_b) and len(new_b):  # paris verrouillés (match commencé ou réglé) : on ne les rejoue pas
        locked = set(zip(old_b["gameId"], old_b["strategy"]))
        new_b = new_b[[k not in locked for k in zip(new_b["gameId"], new_b["strategy"])]]
    pd.concat([old_b, new_b], ignore_index=True).to_csv(BETS, index=False)
    pd.concat([old_p, pd.DataFrame(preds)], ignore_index=True).to_csv(PREDS, index=False)
    JOB["msg"] = f"{len(preds)} matchs prédits, {len(bets)} paris fictifs enregistrés."


def run_settle():
    bets, preds = read(BETS), read(PREDS)
    if bets.empty:
        JOB["msg"] = "Aucun pari à régler."
        return
    results = {}
    for day in sorted(set(bets.loc[bets["status"] == "en attente", "date"]) | set(preds.get("date", []))):
        JOB["msg"] = f"Résultats du {day}…"
        for gm in get_json(f"{WEB}/score/{day}")["games"]:
            if gm["gameState"] in ("OFF", "FINAL"):
                h, a = gm["homeTeam"]["score"], gm["awayTeam"]["score"]
                period = gm.get("gameOutcome", {}).get("lastPeriodType", "REG")
                win = "H" if h > a else "A"
                results[gm["id"]] = {"Vainqueur": win, "1N2": win if period == "REG" else "X",
                                     "score": f"{a} - {h}" + ("" if period == "REG" else f" ({period})")}
    n = 0
    for idx, b in bets[bets["status"] == "en attente"].iterrows():
        r = results.get(int(b["gameId"]))
        if r:
            won = b["pick"] == r[b["market"]]
            bets.loc[idx, ["status", "profit", "score"]] = ["gagné" if won else "perdu",
                                                           b["stake"] * (b["price"] - 1) if won else -b["stake"],
                                                           r["score"]]
            n += 1
    bets.to_csv(BETS, index=False)
    if len(preds):
        preds["result"] = preds["gameId"].map(lambda i: results.get(int(i), {}).get("Vainqueur"))
        preds["result3"] = preds["gameId"].map(lambda i: results.get(int(i), {}).get("1N2"))
        preds["score"] = preds["gameId"].map(lambda i: results.get(int(i), {}).get("score"))
        preds.to_csv(PREDS, index=False)
    JOB["msg"] = f"{n} paris réglés."


def run_odds_update():
    JOB["msg"] = "Historique des cotes de la saison (≈ 6 s par nouveau match terminé)…"
    odds.update()
    MODEL_CACHE.clear()
    JOB["msg"] = "Cotes à jour."


def state():
    bets, preds = read(BETS), read(PREDS)
    summary, curve = [], []
    for key, name in STRATEGIES.items():
        b = bets[bets["strategy"] == key] if len(bets) else pd.DataFrame()
        done = b[b["status"] != "en attente"].sort_values("kickoff") if len(b) else b
        staked = done["stake"].sum() if len(done) else 0
        profit = done["profit"].sum() if len(done) else 0
        summary.append({"key": key, "name": name, "settled": len(done),
                        "pending": int((b["status"] == "en attente").sum()) if len(b) else 0,
                        "won": int((done["status"] == "gagné").sum()) if len(done) else 0, "staked": staked,
                        "profit": profit, "roi": profit / staked if staked else None})
        if len(done):
            curve += [{"strategy": key, "i": i + 1, "kickoff": k, "cum": c, "match": f"{a} @ {h}", "profit": p}
                      for i, (k, c, a, h, p) in enumerate(zip(done["kickoff"], done["profit"].cumsum(), done["away"],
                                                              done["home"], done["profit"]))]
    if len(preds):
        preds = preds.sort_values("kickoff", ascending=False)
    return {"bets": records(bets.sort_values("kickoff", ascending=False) if len(bets) else bets),
            "preds": records(preds), "summary": summary, "curve": curve,
            "strategies": STRATEGIES, "today": et_today(), "job": JOB}


def model_stats():
    if MODEL_CACHE:
        return MODEL_CACHE
    wf = nhl.walk_forward()
    ll = lambda y, p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    wf["ll"] = ll(wf["home_win"], wf["p"])
    wf["ok"] = (wf["p"] > 0.5) == (wf["home_win"] == 1)
    co = odds.closing_odds()
    if len(co):
        pin = co[(co["book"] == "pinnacle") & (co["market"] == "151")].pivot_table(
            index=["home", "date"], columns="side", values="price").reset_index()
        pin["q"] = (1 / pin["H"]) / (1 / pin["H"] + 1 / pin["A"])
        wf = wf.assign(home_n=wf["home"].map(odds.norm), date_d=wf["date"].dt.date).merge(
            pin[["home", "date", "q"]].rename(columns={"home": "home_n", "date": "date_d"}), how="left")
        wf["ll_pin"] = ll(wf["home_win"], wf["q"])
    seasons = []
    for s, g in wf.groupby("season"):
        m = g.dropna(subset=["q"]) if "q" in g else g.iloc[:0]
        seasons.append({"season": f"{s}-{s + 1 - 2000}", "n": len(g), "logloss": g["ll"].mean(), "accuracy": g["ok"].mean(),
                        "n_pin": len(m), "ll_model_pin": m["ll"].mean() if len(m) else None,
                        "ll_pin": m["ll_pin"].mean() if len(m) else None})
    wf["bin"] = pd.cut(wf["p"], np.linspace(0, 1, 11))
    calib = [{"p": g["p"].mean(), "obs": g["home_win"].mean(), "n": len(g)}
             for _, g in wf.groupby("bin", observed=True) if len(g) >= 30]
    w, _ = nhl.history(*nhl.load())
    w = w.dropna(subset=nhl.FEATURES + ["home_win"])
    m = nhl.LogisticRegression().fit(w[nhl.FEATURES], w["home_win"])
    MODEL_CACHE.update(clean({
        "seasons": seasons, "calibration": calib, "n_games": len(wf), "logloss": wf["ll"].mean(),
        "accuracy": wf["ok"].mean(), "coefs": dict(zip(nhl.FEATURES, m.coef_[0])),
    }))
    return MODEL_CACHE


def start(job, *args):
    if JOB["busy"]:
        return False

    def run():
        JOB.update(busy=True, error=None, msg="Démarrage…")
        try:
            job(*args)
        except Exception as e:  # remonté tel quel à l'interface
            JOB["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            JOB.update(busy=False, done_at=datetime.now().isoformat(timespec="seconds"))

    threading.Thread(target=run, daemon=True).start()
    return True


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(clean(body), default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, (ROOT / "ui.html").read_bytes(), "text/html")
        routes = {"/api/state": state, "/api/job": lambda: JOB, "/api/model": model_stats}
        if self.path in routes:
            try:
                return self.send(200, routes[self.path]())
            except Exception as e:
                traceback.print_exc()
                return self.send(500, {"error": f"{type(e).__name__}: {e}"})
        self.send(404, {"error": "introuvable"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        jobs = {
            "/api/predict": (run_predict, body.get("date") or et_today(), float(body.get("stake") or 10),
                             bool(body.get("jev", True))),
            "/api/settle": (run_settle,),
            "/api/odds-update": (run_odds_update,),
        }
        if self.path not in jobs:
            return self.send(404, {"error": "introuvable"})
        ok = start(*jobs[self.path])
        self.send(202 if ok else 409, {"started": ok, "job": JOB})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"NHL Predictor : http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
