# NHL Predictor

Prédiction des matchs de NHL et **simulation de paris fictifs à mise fixe**, mise à jour automatiquement chaque jour.
Aucun pari réel n'est engagé : le projet mesure si un modèle peut faire mieux que le marché des cotes.

**Page publique :** https://cr0zer.github.io/NHL-Predictor/

## Fonctionnement

```
API NHL (résultats, play-by-play) ──► modèle xG (qualité de chaque tir) ──► variables d'avant-match ──► régression logistique
NHL.com (articles d'avant-match) ──► modèle de langage (extraction) ──► Jev (gardien titulaire, absences) ──┘        │
OddsPapi (Pinnacle, Unibet FR, Winamax) ──────────────────────────────────────────────────────► paris fictifs ◄──┘
```

| Fichier | Rôle |
|---|---|
| `xg.py` | Modèle xG (gradient boosting) appris saison par saison sur les tirs des saisons précédentes |
| `nhl.py` | Variables d'avant-match (xG, Corsi, buts, gardien GSAx, fatigue, absences), modèle, backtest, prédictions |
| `news.py` | Actualité d'avant-match : articles NHL.com → extraction structurée par un modèle de langage → jugement Jev |
| `odds.py` | Cotes OddsPapi (cotes actuelles et de clôture), évaluation face au marché |
| `app.py` | Paris fictifs, règlement, export de la page ; serveur local avec boutons |
| `ui.html` | Interface (onglets Soirée, Simulation, Modèle) |
| `test_nhl.py` | Garde-fou anti-fuite : une variable d'avant-match n'utilise que les matchs précédents |

**Variable cible :** victoire de l'équipe à domicile, prolongation et tirs au but inclus. Le 1N2 en temps réglementaire
(Winamax) en est déduit avec une probabilité de prolongation constante.

**Règles anti-fuite :** chaque saison est prédite par un modèle appris sur les saisons précédentes ; les articles ne
sont retenus que s'ils ont été publiés *et* modifiés avant le coup d'envoi ; la cote de clôture est la dernière publiée
avant le coup d'envoi réel.

## Résultats (octobre 2026)

| Mesure | Valeur |
|---|---|
| Backtest 2018-19 → 2025-26 (~9 800 matchs hors échantillon) | log-loss 0.665, favori gagnant 58,9 % |
| Face à Pinnacle à la clôture (janvier-avril 2026, 511 matchs) | log-loss identique (0.6719 contre 0.6719) |
| Apport de l'xG | significatif (-0.0012 ± 0.0005 de log-loss) |
| Apport du gardien ou des absences | faible, non significatif à ce stade |

Le modèle est au niveau du marché le plus efficace, sans le battre de façon démontrée : la simulation de paris
fictifs sert à le vérifier dans la durée.

## Stratégies simulées (mise fixe de 10 €)

| Stratégie | Règle |
|---|---|
| Favori du modèle (Unibet FR, vainqueur) | Un pari par match sur l'équipe que le modèle donne gagnante |
| Valeur Unibet (vainqueur) | Côté où probabilité du modèle × cote − 1 est le plus élevé, s'il est positif |
| Valeur Winamax (1N2) | Même règle sur les trois issues |

## Automatisation (GitHub Actions, `.github/workflows/pipeline.yml`)

| Heure (UTC) | Heure de Paris (été) | Tâche |
|---|---|---|
| 16 h 30 | 18 h 30 | Prédictions de la soirée et paris fictifs aux cotes du moment |
| 22 h 00 | minuit | Mise à jour pour les matchs de la côte Ouest |
| 7 h 30 | 9 h 30 | Résultats officiels et règlement des paris |
| lundi 8 h 00 | lundi 10 h | Cotes de clôture de la semaine |

Après chaque tâche, le registre de paris et les prédictions sont enregistrés dans le dépôt (`data/`) et la page est
republiée sur GitHub Pages. Les données NHL brutes sont retéléchargées ou mises en cache.

Secrets du dépôt nécessaires : `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, `ODDSPAPI_API_KEY`.

## Modèles de langage (extraction de l'actualité)

Le modèle se choisit avec la variable d'environnement `NHL_LLM` : un nom de modèle Ollama (local), ou une liste de
modèles OpenRouter essayée dans l'ordre (le suivant prend le relais si l'un est saturé ou indisponible).

| Modèle | Où | Résultat sur les 43 premiers matchs de 2026-27 |
|---|---|---|
| **Nemotron 3 Super 120B** (`nvidia/nemotron-3-super-120b-a12b:free`) | OpenRouter, gratuit | **Production.** 6,7 s par match, gardien trouvé 91 %, absents « certains » exacts à 95 % |
| Gemma 4 26B / 31B (`:free`) | OpenRouter, gratuit | Secours (souvent saturés) |
| qwen3:8b | Ollama, local | Défaut en local. 22 s par match, gardien 87 %, absents exacts à 94 %, moins d'absents repérés |
| Llama 3.1 8B | Ollama, local | Testé en local : fonctionne, mais abandonné (rappel plus faible, réponses JSON qui bouclent) |

Quota gratuit OpenRouter : 50 requêtes par jour, une par match.

## Utilisation en local

```bash
uv sync
uv run --env-file .env python app.py                 # interface avec boutons : http://127.0.0.1:8765
uv run --env-file .env python app.py predict          # ou settle, odds, export
uv run python nhl.py backtest                         # backtest walk-forward
uv run --env-file .env python news.py eval            # évaluation de la chaîne d'actualité sur la saison
uv run --env-file .env python odds.py eval [SAISON]   # modèle face au marché
uv run python test_nhl.py
```

Fichier `.env` (jamais versionné) : `TYPESAFE_API_KEY`, `ODDSPAPI_API_KEY`, `OPENROUTER_API_KEY` et, au besoin,
`NHL_LLM`. Sans `NHL_LLM`, l'extraction utilise `qwen3:8b` via Ollama.

## Sources

API NHL (`api.nhle.com`, `api-web.nhle.com`), contenus NHL.com, OddsPapi, OpenRouter, TypeSafe (Jev).
