# ⚡ WattCast — Prévision de la consommation électrique française (J+1)

> Prévision à 24 h de la consommation électrique nationale, **mise à jour automatiquement chaque jour** et comparée en direct à la réalité.
>
> 🔗 **Démo live : [à remplir — lien Cloudflare](https://wattcast.votre-domaine.workers.dev)**

<!-- ⬜ Renomme librement le projet. "WattCast" est une suggestion. -->

![Aperçu du dashboard](docs/preview.gif)
<!-- ⬜ TODO : remplace par un vrai GIF/screenshot du dashboard. C'est le premier truc que le recruteur regarde. -->

---

## 🎯 En une phrase

Un modèle **XGBoost** prédit la consommation électrique française heure par heure pour les 24 prochaines heures, à partir de l'historique de consommation et de la **température prévue**. Le pipeline se met à jour seul via une GitHub Action, et un dashboard **Astro statique** affiche **prédit vs réel**.

## 📊 Résultats

| Modèle | MAE (MW) | MAPE | vs baseline |
|---|---|---|---|
| Baseline naïve (conso H-168) | 3 210 | 6.20 % | — |
| Prophet (saisonnalités + régresseurs) | 1 449 | 2.94 % | −55 % |
| **WattCast (XGBoost)** | **885** | **1.78 %** | **−72 % d'erreur** |
| Prévision officielle RTE (J-1) | 464 | 0.94 % | référence |

> Chiffres du **bake-off walk-forward** (5 folds, `python -m ml.evaluate`) sur le **jeu synthétique de démonstration** livré avec le repo (`make demo`, données générées hors-ligne, reproductible sans clé ni réseau). Le déploiement quotidien recalcule exactement ces métriques sur les **données réelles** RTE / Open-Meteo.

> **Ce qu'un recruteur doit remarquer :** la performance est mesurée en **validation temporelle stricte** (entraînement sur le passé, test sur la période la plus récente, **jamais de mélange aléatoire**), comparée à une baseline **et** à un second modèle (Prophet). Pas de fuite de données, pas de métrique gonflée.

## 🧠 Approche

- **Données** — consommation réalisée éCO2mix (RTE / ODRÉ), ~3 ans au pas horaire.
- **Feature clé** — la consommation est pilotée par la **température** (chauffage l'hiver, clim l'été). La courbe conso-vs-température en U est le cœur du modèle.
- **Features** — calendaires (heure, jour de semaine, mois), jours fériés (`holidays`), lags (`H-24`, `H-168`), moyennes glissantes, température présente et prévue.
- **Modèles** — **XGBoost** (gradient boosting) en production, comparé à **Prophet** (saisonnalités additives + température/lags en régresseurs). Une interface commune (`ml/models.py`) permet le bake-off à armes égales.
- **Évaluation** — walk-forward (`TimeSeriesSplit`), MAE + MAPE, comparaison baseline naïve + Prophet + prévision officielle RTE.

## 🏗️ Architecture

Aucun serveur à maintenir, aucun calcul au runtime. Une GitHub Action programmée fait tout le travail (le ML tourne là, pas dans le navigateur), puis déploie un site **statique** sur Cloudflare.

```
┌──────────────────────────────────────────────────────────────┐
│  GitHub Action — cron quotidien                               │
│                                                               │
│  ODRÉ éCO2mix (conso récente) ─┐                              │
│                                ├─► model.predict() ─► JSON    │
│  Open-Meteo (température J+1) ──┘            │                │
│                                              ▼                │
│                          build Astro  ──►  wrangler deploy    │
└───────────────────────────────────────────┬──────────────────┘
                                             │ assets statiques (dist/)
                                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Cloudflare Workers Static Assets — sert dist/ (zéro script)  │
│  Dashboard « prédit vs réel » · MAE/MAPE · feature importance │
└──────────────────────────────────────────────────────────────┘
```

> Le front est 100 % statique : il lit `predictions.json` et `actuals.json` côté client. Pas de SSR → pas besoin de l'adaptateur `@astrojs/cloudflare`, pas de framework serveur.

## 📂 Structure du repo

```
wattcast/
├── ml/
│   ├── config.py          # chemins, villes (température pondérée), constantes
│   ├── ingest.py          # éCO2mix + Open-Meteo → data/raw (+ mode --synthetic)
│   ├── features.py        # calendrier, jours fériés, lags, météo → data/processed
│   ├── models.py          # interface forecaster commune : XGBoost & Prophet
│   ├── train.py           # entraîne le modèle → model/model.pkl + metrics.json
│   ├── evaluate.py        # bake-off walk-forward → model/evaluation.json
│   ├── predict.py         # prévision J+1 → web/public/data/*.json
│   ├── utils.py           # métriques (MAE/MAPE) + split temporel
│   └── tests/             # pytest : features, modèles, pipeline bout-en-bout
├── data/                  # parquet (gitignored)
├── model/                 # model.pkl + metrics/evaluation.json (gitignored)
├── web/                   # app Astro (build statique → dist/), géré avec bun
│   ├── src/
│   │   ├── pages/index.astro
│   │   ├── layouts/ · components/   # masthead, stats, graphe Chart.js, table
│   │   └── styles/global.css
│   ├── public/data/        # predictions/actuals/metrics/evaluation.json (snapshot + CI)
│   ├── astro.config.mjs · bun.lock
│   └── wrangler.jsonc      # assets.directory → ./dist
├── .github/workflows/
│   ├── ci.yml              # lint (ruff) + tests (pytest) + build web sur PR
│   └── daily-forecast.yml  # cron : ingest + train + evaluate + predict + deploy
├── requirements.txt · requirements-dev.txt
├── pyproject.toml · Makefile
└── README.md
```

## 🚀 Lancer en local

Les scripts s'exécutent comme des modules (`python -m ml.<étape>`) depuis la racine.

```bash
# 1. Dépendances Python
pip install -r requirements.txt          # (ou requirements-dev.txt pour ruff + pytest)

# 2. Construire le dataset d'entraînement (~3 ans)
python -m ml.ingest --start 2022-01-01 --end 2024-12-31
python -m ml.features
#   ↳ pas de réseau ? ajoute --synthetic pour un jeu de démo réaliste hors-ligne

# 3. Entraîner + évaluer
python -m ml.train                       # XGBoost → model/model.pkl
python -m ml.evaluate                    # bake-off walk-forward XGBoost vs Prophet
python -m ml.train --model prophet       # (optionnel) déployer Prophet à la place

# 4. Générer la prévision J+1 (écrit web/public/data/*.json)
python -m ml.predict

# 5. Dashboard (bun)
cd web && bun install && bun run dev      # dev local
bun run build                             # build statique → dist/
bunx wrangler deploy                      # déploie sur Cloudflare Workers
```

> Raccourci : `make demo` enchaîne tout le pipeline hors-ligne (données synthétiques), `make test` lance la suite pytest, `make web` construit le dashboard.

## ☁️ Déploiement (Cloudflare Workers Static Assets)

Site purement statique → aucun script Worker. Le `wrangler.jsonc` se résume à :

```jsonc
{
  "name": "wattcast",
  "compatibility_date": "2026-06-23",
  "assets": { "directory": "./dist" }
}
```

Pas de champ `main` ni de binding `ASSETS` (ils ne servent que s'il y a du code Worker). `bunx wrangler deploy` suffit.

*Alternative :* Cloudflare Pages (Git-first, previews par branche automatiques) fonctionne aussi — connecter le repo, build `npm run build`, output `dist/`. À noter : Cloudflare fait converger Pages vers Workers à terme.

## 🔄 Mise à jour automatique

`.github/workflows/daily-forecast.yml` s'exécute chaque jour : récupère les dernières valeurs réelles (éCO2mix) + la prévision météo (Open-Meteo), réentraîne, relance le bake-off, recalcule la prévision J+1, écrit les JSON, rebuild Astro et `bunx wrangler deploy`. L'ingestion tourne en mode `--strict` (échec bruyant plutôt que repli synthétique silencieux en production). Secrets requis : `CLOUDFLARE_API_TOKEN` et `CLOUDFLARE_ACCOUNT_ID`.

En parallèle, `.github/workflows/ci.yml` garde le repo vert à chaque PR : lint `ruff`, tests `pytest`, et build du dashboard avec `bun`.

## 🗃️ Sources de données

Aucune clé d'API requise.

| Source | Usage | Lien |
|---|---|---|
| éCO2mix conso consolidée/définitive (ODRÉ) | Entraînement (historique 2012→) | https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-national-cons-def/ |
| éCO2mix temps réel (ODRÉ) | Valeurs réelles récentes (live) | https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-national-tr/ |
| Open-Meteo Archive | Température historique | https://open-meteo.com/en/docs/historical-weather-api |
| Open-Meteo Forecast | Température prévue J+1 | https://open-meteo.com/en/docs |

**Exemple d'export ODRÉ** (API Opendatasoft v2.1, renvoie tout le dataset filtré en CSV) :

```
https://opendata.reseaux-energies.fr/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv?timezone=Europe/Paris
```

> ⚠️ Vérifie les noms exacts des colonnes dans la console API du dataset (a priori `date_heure`, `consommation`, `prevision_j1`, `prevision_j`). La consommation est au pas demi-heure, les prévisions au pas quart d'heure → rééchantillonne à l'heure et filtre les lignes où `consommation` est nulle avant d'entraîner.

## 🛣️ Pistes d'amélioration

- **Découpler la donnée du déploiement** : écrire les JSON dans **Cloudflare R2** ou **KV** plutôt que dans le repo → la donnée se met à jour sans rebuild du site (et démontre la maîtrise des primitives Cloudflare).
- Température pondérée par la population sur plusieurs villes (Paris, Lyon, Marseille, Lille, Toulouse) plutôt qu'un seul point.
- Intervalles de prédiction (quantile regression) plutôt qu'une valeur ponctuelle.
- Variante prix spot en plus de la consommation.
- Étendre le bake-off (Prophet est déjà branché) : LightGBM, ou un petit modèle séquentiel (LSTM / TFT).

## 📝 Licence

Code sous licence **MIT**.

Données de consommation © **RTE** (éCO2mix), via **ODRÉ**, réutilisées à titre informatif avec mention de la source, hors usage commercial. Données météo © **Open-Meteo** (CC-BY 4.0).

## 👤 Auteur

`Rossetto-Giaccherino Corentin` — [LinkedIn]([https://linkedin.com/in/...](https://www.linkedin.com/in/corentin-rossetto-giaccherino/)) ·
<!-- ⬜ TODO : mets tes liens. Un recruteur qui arrive ici doit pouvoir te contacter en un clic. -->
