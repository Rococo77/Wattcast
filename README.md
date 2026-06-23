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
| Baseline naïve (conso H-168) | `XX` | `XX %` | — |
| Prévision officielle RTE (J-1) | `XX` | `XX %` | référence |
| **WattCast (XGBoost)** | **`XX`** | **`XX %`** | **`-XX %` d'erreur** |

<!-- ⬜ TODO : remplis ce tableau après `python ml/evaluate.py`.
     L'argument fort = battre la baseline ET se rapprocher (voire battre) la prévision officielle de RTE. -->

> **Ce qu'un recruteur doit remarquer :** la performance est mesurée en **validation temporelle stricte** (entraînement sur le passé, test sur la période la plus récente, **jamais de mélange aléatoire**) et systématiquement comparée à une baseline. Pas de fuite de données, pas de métrique gonflée.

## 🧠 Approche

- **Données** — consommation réalisée éCO2mix (RTE / ODRÉ), ~3 ans au pas horaire.
- **Feature clé** — la consommation est pilotée par la **température** (chauffage l'hiver, clim l'été). La courbe conso-vs-température en U est le cœur du modèle.
- **Features** — calendaires (heure, jour de semaine, mois), jours fériés (`holidays`), lags (`H-24`, `H-168`), moyennes glissantes, température présente et prévue.
- **Modèle** — XGBoost (gradient boosting), robuste et rapide à entraîner sur ce volume.
- **Évaluation** — walk-forward (`TimeSeriesSplit`), MAE + MAPE, comparaison baseline + prévision officielle RTE.

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
│   ├── ingest.py          # télécharge éCO2mix + Open-Meteo → data/raw
│   ├── features.py        # calendrier, lags, météo → data/processed
│   ├── train.py           # entraîne XGBoost → model/model.pkl + metrics.json
│   ├── predict.py         # charge le modèle → web/public/data/predictions.json
│   └── evaluate.py        # walk-forward, MAE/MAPE vs baseline
├── data/                  # parquet (gitignored)
├── model/                 # model.pkl + metrics.json
├── web/                   # app Astro (build statique → dist/)
│   ├── src/
│   │   ├── pages/index.astro
│   │   └── components/     # île interactive (graphe)
│   ├── public/data/        # predictions.json + actuals.json (écrits par la CI)
│   ├── astro.config.mjs
│   └── wrangler.jsonc      # assets.directory → ./dist
├── .github/workflows/
│   └── daily-forecast.yml  # cron : ingest récent + predict + build + deploy
├── requirements.txt
└── README.md
```

## 🚀 Lancer en local

```bash
# 1. Dépendances Python
pip install -r requirements.txt

# 2. Construire le dataset d'entraînement (~3 ans)
python ml/ingest.py --start 2022-01-01 --end 2024-12-31
python ml/features.py

# 3. Entraîner + évaluer
python ml/train.py
python ml/evaluate.py        # affiche MAE / MAPE vs baseline

# 4. Générer la prévision J+1 (écrit web/public/data/predictions.json)
python ml/predict.py

# 5. Dashboard
cd web && npm install && npm run dev    # dev local
npm run build                            # build statique → dist/
npx wrangler deploy                      # déploie sur Cloudflare Workers
```

## ☁️ Déploiement (Cloudflare Workers Static Assets)

Site purement statique → aucun script Worker. Le `wrangler.jsonc` se résume à :

```jsonc
{
  "name": "wattcast",
  "compatibility_date": "2026-06-23",
  "assets": { "directory": "./dist" }
}
```

Pas de champ `main` ni de binding `ASSETS` (ils ne servent que s'il y a du code Worker). `npx wrangler deploy` suffit.

*Alternative :* Cloudflare Pages (Git-first, previews par branche automatiques) fonctionne aussi — connecter le repo, build `npm run build`, output `dist/`. À noter : Cloudflare fait converger Pages vers Workers à terme.

## 🔄 Mise à jour automatique

`.github/workflows/daily-forecast.yml` s'exécute chaque jour : récupère les dernières valeurs réelles (éCO2mix temps réel) + la prévision météo (Open-Meteo), recalcule la prévision J+1, écrit les JSON, rebuild Astro et `wrangler deploy`. Secrets requis : `CLOUDFLARE_API_TOKEN` et `CLOUDFLARE_ACCOUNT_ID`.

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
- Comparaison LightGBM / Prophet / un petit modèle séquentiel.

## 📝 Licence

Code sous licence **MIT**.

Données de consommation © **RTE** (éCO2mix), via **ODRÉ**, réutilisées à titre informatif avec mention de la source, hors usage commercial. Données météo © **Open-Meteo** (CC-BY 4.0).

## 👤 Auteur

`Rossetto-Giaccherino Corentin` — [LinkedIn]([https://linkedin.com/in/...](https://www.linkedin.com/in/corentin-rossetto-giaccherino/)) ·
<!-- ⬜ TODO : mets tes liens. Un recruteur qui arrive ici doit pouvoir te contacter en un clic. -->
