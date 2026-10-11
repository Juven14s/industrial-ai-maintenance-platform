# Industrial AI Maintenance Platform

[![CI](https://github.com/Juven14s/industrial-ai-maintenance-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Juven14s/industrial-ai-maintenance-platform/actions/workflows/ci.yml)

> Une interface Streamlit de démonstration permet de tester la prédiction, d'interroger la documentation et de construire un planning de flotte sans écrire de JSON. Elle appelle l’API FastAPI par HTTP et ne duplique ni le prétraitement, ni les modèles, ni la logique métier.

## Interface de démonstration Streamlit

Le dashboard guide l’utilisateur dans un parcours simple : choisir un moteur FD001 de démonstration ou importer un CSV, sélectionner le cycle d’observation, examiner les courbes des capteurs, puis demander une prédiction. Il affiche la RUL conservatrice comme valeur de décision, la RUL centrale comme information complémentaire, une échéance prudente exprimée en cycles et les détails des modèles. Une deuxième zone accepte une question technique et développe chaque passage cité. Une troisième permet d'éditer une flotte simulée, d'utiliser la RUL conservatrice courante et de comparer une heuristique au planning CP-SAT. Les deux RUL proviennent de modèles indépendants et ne constituent pas un intervalle de confiance.

En local, lance les deux services dans deux terminaux depuis la racine du projet.

Terminal 1 — API FastAPI :

```powershell
.\.venv\Scripts\serve-maintenance-api.exe --config configs/api.toml
```

Terminal 2 — interface Streamlit :

```powershell
.\.venv\Scripts\serve-maintenance-dashboard.exe --api-url http://127.0.0.1:8000 --port 8501
```

Ouvre ensuite <http://127.0.0.1:8501>. Le dashboard communique uniquement avec FastAPI ; l’URL peut aussi être fournie par la variable d’environnement `MAINTENANCE_API_URL`.

Le mode Docker Compose expose l’API sur `http://127.0.0.1:8000` et le dashboard sur `http://127.0.0.1:8501`. Les fichiers de conteneurisation sont fournis, mais leur exécution reste à valider sur une machine équipée de Docker : Docker n’est pas installé dans l’environnement local actuel.

Cette interface sert à démontrer les pipelines ML, RAG et d'optimisation. Les coûts, criticités, durées et ressources du planning sont simulés ; l’« échéance prudente » reste exprimée en cycles et ne constitue pas une date d’intervention réelle.

Projet de portfolio après un master 2 en data science : prédire la durée de vie restante d'équipements à partir de séries temporelles, interroger une documentation technique avec des réponses sourcées et proposer un planning de maintenance sous contraintes.

**État actuel : chaîne RUL, RAG documentaire, optimisation sous contraintes, API FastAPI et dashboard Streamlit réalisés sur FD001 et des scénarios métier simulés.** La V1 atteint une **RMSE de 20,986 cycles** et une **MAE de 14,233 cycles** sur le test officiel, mais 73 % de surestimations. La V2 réduit le taux de surestimation hors pli à **5,1 %**, avec le compromis d'un biais prudent de −19,573 cycles. Le RAG local fournit des réponses citées avec abstention. Le planificateur compare une heuristique à OR-Tools CP-SAT et réduit le coût simulé de **33,5 %** sur le scénario contraint, sans présenter ce chiffre comme une économie industrielle observée.

Le développement est prévu sur PC local, avec Google Colab pour les entraînements si nécessaire.

- [Cahier des charges](docs/CAHIER_DES_CHARGES.md) : périmètre, architecture, protocole scientifique et critères de validation.
- [Feuille de route](docs/FEUILLE_DE_ROUTE.md) : étapes de réalisation, livrables et préparation de la présentation du projet.
- [Architecture](docs/ARCHITECTURE.md) : diagramme et circulation des données entre ML, RAG, planification, API et interface.
- [Rapport technique synthétique](reports/PROJECT_REPORT.md) : protocole, résultats, limites et bilan de la plateforme.
- [Validation finale locale](reports/FINAL_VALIDATION.md) : tests, lint, dépendances, liens et configurations contrôlés.
- [Scénario de démonstration](docs/DEMONSTRATION.md) et [présentation portfolio](docs/PORTFOLIO.md) : parcours de quelques minutes, formulation CV et points d'entretien.
- [Audit des données](reports/DATASET_AUDIT.md) : effectifs mesurés, schéma des colonnes, contrôles et écart documentaire sur FD004.
- [Manifeste des fichiers](reports/dataset_manifest.json) : comptages, résultats de contrôle et empreintes SHA-256 de la copie locale.
- [Notebook d'exploration exécuté](notebooks/01_fd001_exploration.ipynb) : explications, trajectoires, capteurs et vérification du protocole.
- [Dictionnaire des données](docs/DICTIONNAIRE_DONNEES.md) : colonnes, significations physiques, unités et réserves documentaires.
- [Rapport de préparation FD001](reports/fd001/PREPARATION_FD001.md) : groupes, prétraitement, fenêtres et graphiques.
- [Configuration FD001](configs/fd001.toml) : chemins, graines et longueur des fenêtres.
- [Rapport de comparaison des modèles](reports/fd001/models/MODEL_COMPARISON.md) : protocole, résultats, figures et limites.
- [Métriques par modèle](reports/fd001/models/comparison.csv), [prédictions de validation](reports/fd001/models/validation_predictions.csv) et [figures](reports/fd001/models/figures/) : sorties vérifiables du jalon 2.
- [Configuration des modèles](configs/fd001_models.toml) : filtrage des fenêtres, candidats Random Forest, LSTM et suivi MLflow.
- [Évaluation officielle FD001](reports/fd001/test/OFFICIAL_TEST_EVALUATION.md) : protocole final, métriques, analyse des erreurs et limites.
- [Configuration de l'évaluation finale](configs/fd001_final.toml), [prédictions officielles](reports/fd001/test/official_test_predictions.csv) et [résumé reproductible](reports/fd001/test/official_test_summary.json).
- [Sélection sécurité V2](reports/fd001/safety_v2/SAFETY_SELECTION.md) et [calibration croisée](reports/fd001/safety_v2/SAFETY_CALIBRATION.md) : validation groupée, coût asymétrique et analyse du compromis précision/prudence.
- [Configurations de sélection](configs/fd001_safety_v2.toml) et de [calibration](configs/fd001_safety_calibration.toml), [comparaison calibrée](reports/fd001/safety_v2/calibrated_comparison.csv) et [bundle conservateur](artifacts/models/FD001/safety_v2/model_manifest.json).
- [API FastAPI](src/industrial_maintenance/api.py), [pipeline d'inférence](src/industrial_maintenance/inference.py) et [configuration du service](configs/api.toml) : validation des historiques, vérification des artefacts et double prédiction RUL.
- [Dashboard Streamlit](src/industrial_maintenance/dashboard_app.py), [client et validation UI](src/industrial_maintenance/dashboard.py) et [trajectoire de démonstration](examples/fd001_demo_engine_3.csv) : prédiction, question-réponse et édition d'une flotte à planifier sans manipulation de JSON.
- [Image de service](Dockerfile) et [configuration Compose](compose.yaml) : construction de l'environnement d'inférence, de RAG et de planification avec contrôle de santé HTTP.
- [Validation de bout en bout de l'API](reports/fd001/api/API_VALIDATION.md) : routes réellement appelées, prédiction issue d'une trajectoire FD001 et résultat des contrôles automatisés.
- [Validation du dashboard](reports/fd001/ui/DASHBOARD_VALIDATION.md) : lancement réel, interaction Streamlit, prédiction de démonstration et limites actuelles.
- [Corpus RAG gelé](configs/rag_corpus.toml), [configuration](configs/rag.toml), [jeu de 30 questions](data/evaluation/rag/questions_v1.jsonl), [évaluation de la recherche](reports/rag/RETRIEVAL_EVALUATION.md) et [évaluation de la génération](reports/rag/GENERATION_EVALUATION.md) : extraction traçable, recherche hybride, réponses citées et abstention mesurée.
- [Service de recherche](src/industrial_maintenance/document_search.py), [génération locale](src/industrial_maintenance/rag_generation.py) et [validation API RAG](reports/rag/API_VALIDATION.md) : chargement unique des modèles, citations contrôlées et refus des demandes sans preuve documentaire.
- [Planificateur](src/industrial_maintenance/planning.py), [configuration](configs/maintenance_planning.toml), [scénarios](data/scenarios/maintenance/) et [rapport d'évaluation](reports/planning/PLANNING_EVALUATION.md) : heuristique, CP-SAT, contraintes de ressources, coût simulé et détection des scénarios irréalisables.

## Résultats de la préparation

Seul `CMAPSSData/train_FD001.txt` est lu par cette étape. Ses 100 moteurs sont séparés en **80 moteurs d'apprentissage et 20 de validation**, avant la sélection des capteurs et la normalisation.

| Groupe | Observations utilisées | Forme des entrées `X` |
| --- | ---: | --- |
| Apprentissage : 80 moteurs complets | 16 779 | `(16779, 30, 15)` |
| Validation : 20 moteurs tronqués avant la fin de vie | 2 037 | `(20, 30, 15)` |

Une entrée correspond à une fenêtre de **30 cycles × 15 capteurs**. L'apprentissage contient une fenêtre par observation ; la validation contient une seule fenêtre au dernier cycle observable de chaque moteur. Les 20 trajectoires de validation comptaient 3 852 observations avant troncature. La RUL est calculée avant cette troncature, puis conservée comme cible.

Six capteurs exactement constants sur l'apprentissage sont écartés : `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18` et `sensor_19`. Les capteurs de faible variance restent présents. Les moyennes et écarts-types de normalisation proviennent uniquement des moteurs d'apprentissage. Les historiques de moins de 30 cycles sont complétés à gauche en répétant leur première observation ; leur longueur réelle est enregistrée.

À ce stade de préparation, le test officiel et ses cibles RUL étaient réservés à l'évaluation finale. Le pipeline de préparation et le notebook ne les chargent toujours pas ; seule la commande finale dédiée les utilise.

## Résultats de validation du jalon 2

L'entraînement utilise 15 179 fenêtres admissibles : leur cycle final est au moins égal à 20 et précède strictement la panne. Chaque moteur possède le même poids total, afin que les trajectoires longues ne dominent pas l'objectif. Les statistiques du Random Forest et le LSTM utilisent `observed_lengths` pour ignorer le remplissage des historiques courts.

| Modèle | MAE de validation | RMSE de validation |
| --- | ---: | ---: |
| Référence constante pondérée | 36,387 | 46,684 |
| Random Forest `depth12_leaf5` | 22,836 | 30,970 |
| LSTM compact sélectionné | **18,378** | **23,196** |

Le LSTM sélectionné atteint sa meilleure époque à l'époque 6 et s'arrête après 13 époques. Sur les 20 prédictions de validation, son biais moyen `prédiction - RUL réelle` est de +6,801 cycles, son erreur absolue au 90e percentile de 35,392 cycles, son taux de surestimation de 65 % et son score NASA de 342,950.

Par rapport au meilleur Random Forest, le LSTM réduit la RMSE de 7,774 cycles, soit 25,1 %. Le gain est de 23,488 cycles, soit 50,3 %, face à la référence constante.

Ces mesures reposent sur 20 moteurs et une seule coupure fixée par moteur. Trois moteurs seulement ont une RUL inférieure ou égale à 30 cycles. Elles ont servi exclusivement à sélectionner le modèle et l'époque d'arrêt avant l'évaluation officielle.

## Résultats sur le test officiel FD001

Après sélection, le prétraitement et le LSTM ont été réentraînés sur les **100 moteurs complets** de `train_FD001.txt`, soit 18 631 fenêtres admissibles. L'architecture, les règles de filtrage et la durée de six époques étaient figées. Une seule prédiction a ensuite été produite au dernier cycle observé de chacun des 100 moteurs de `test_FD001.txt`, puis comparée à `RUL_FD001.txt`.

| Mesure finale | Résultat |
| --- | ---: |
| MAE | **14,233 cycles** |
| RMSE | **20,986 cycles** |
| Biais moyen, prédiction − réalité | +7,850 cycles |
| Erreur absolue au 90e percentile | 36,439 cycles |
| Taux de surestimation | 73 % |
| Score NASA, somme sur 100 moteurs | 14 432,738 |

La RMSE officielle est inférieure de 2,210 cycles à la RMSE de validation, mais les deux groupes n'ont pas la même distribution et cette différence ne mesure donc pas un gain de modèle. Le biais positif et le taux de surestimation montrent que le modèle annonce souvent une durée restante trop longue. L'erreur la plus forte concerne le moteur de test 78 : 201,675 cycles prédits contre 107 réels, soit +94,675 cycles. Toute amélioration future devra créer un nouveau protocole de validation et conserver ce test comme bilan déjà consulté, sans régler le modèle sur ces 100 cibles.

## Version 2 orientée sécurité

La V2 utilise seulement les 100 moteurs de `train_FD001.txt`. Une validation croisée à cinq plis garde les moteurs entièrement disjoints, réajuste le prétraitement dans chaque pli et évalue neuf horizons déterministes de 5 à 150 cycles. Une surestimation reçoit un coût cinq fois supérieur à une sous-estimation de même amplitude. Le Random Forest MSE est comparé à des Gradient Boosting quantiles, puis une marge de sécurité est calibrée pour chaque pli avec les erreurs des quatre autres plis.

Le modèle retenu est un **HistGradientBoosting au quantile 0,05**, suivi d'une marge finale de **3,969 cycles** et d'un plancher à zéro. Sur les prédictions hors pli :

| Mesure V2 | Modèle conservateur | Random Forest MSE de référence |
| --- | ---: | ---: |
| MAE | 20,029 | 12,317 |
| RMSE | 29,220 | 18,718 |
| Biais | −19,573 | +5,568 |
| Taux de surestimation | **5,1 %** | 66,4 % |
| Surestimation supérieure à 10 cycles | **0,5 %** | 27,1 % |
| Plus forte surestimation | **18,374 cycles** | 76,248 cycles |
| Pire taux parmi les cinq plis | **10,4 %** | non calibré |
| Pire taux parmi les neuf horizons | **15,9 %** | non calibré |

Cette sortie est une **RUL conservatrice destinée à la décision de maintenance**, et non une estimation centrale. Sa prudence entraîne davantage d'interventions anticipées, surtout loin de la panne. Les taux mesurés sont des résultats empiriques de validation croisée ; ils ne garantissent pas qu'un futur moteur respectera exactement ces fréquences. La RMSE V2 n'est pas directement comparable à la RMSE officielle V1, car les horizons et le protocole d'évaluation diffèrent. Le test officiel déjà consulté reste exclu de toute sélection V2.

## API d'inférence, documentaire et de planification

L'API recharge au démarrage le LSTM V1, le Gradient Boosting quantile V2, leurs prétraitements et la calibration. Elle vérifie les empreintes SHA-256 et la cohérence des manifestes avant d'accepter une prédiction. Aucun entraînement n'est lancé par le service.

| Route | Rôle |
| --- | --- |
| `GET /health` | Confirmer que les deux bundles sont chargés et retourner leurs versions |
| `GET /models` | Décrire les capteurs, la fenêtre, le rôle et les métriques de chaque modèle |
| `POST /predict/rul` | Produire l'estimation centrale et la RUL conservatrice d'un moteur |
| `GET /documents/status` | Décrire la version, le modèle et la taille de l'index documentaire |
| `POST /documents/search` | Retourner les passages les plus pertinents avec pages, sources et limites |
| `POST /documents/ask` | Générer une réponse française sourcée ou signaler une abstention |
| `POST /maintenance/plan` | Comparer l'heuristique et CP-SAT sur une flotte simulée |
| `GET /docs` | Ouvrir la documentation interactive générée par FastAPI |

Une requête `POST /predict/rul` contient un `unit_id` entier positif et une liste `observations`. Chaque observation contient un cycle positif et les 15 mesures brutes suivantes, sans normalisation préalable : `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20` et `sensor_21`. Les six capteurs constants écartés à l'entraînement sont facultatifs ; les réglages opératoires, la RUL réelle et tout autre champ sont refusés.

Les cycles doivent être strictement consécutifs, ordonnés et sans doublon, et le dernier cycle doit être au moins égal à 20 comme pendant l'apprentissage. Le service accepte au plus 1 000 observations. Un historique court accepté contient donc de 20 à 29 cycles et doit commencer au cycle 1 ; il est complété à gauche comme pendant l'entraînement. Avec 30 observations ou plus, seuls les 30 derniers cycles alimentent les modèles. Une requête terminée avant le cycle 20 est rejetée au lieu de produire une extrapolation insuffisamment étayée.

La réponse distingue les champs suivants :

| Champ | Signification |
| --- | --- |
| `rul_estimate` | Estimation ponctuelle du LSTM V1 |
| `safety_base_prediction` | Sortie brute du Gradient Boosting quantile V2 |
| `safety_margin` | Marge V2 sérialisée, actuellement 3,969 cycles |
| `rul_conservative` | `max(safety_base_prediction - safety_margin, 0)` |
| `decision_value` | Copie explicite de `rul_conservative` destinée au planificateur |
| `observed_cycles` | Nombre d'observations reçues |
| `window_cycles_used` | Nombre de cycles réellement utilisés, au plus 30 |
| `central_model_version`, `safety_model_version` | Versions dérivées des empreintes des artefacts |

Les deux sorties proviennent de **modèles indépendants**. La marge V2 n'est jamais soustraite à `rul_estimate`, et l'écart entre les deux valeurs n'est ni un intervalle de confiance ni une probabilité de panne. La V1 donne une estimation centrale plus précise sur son test officiel, tandis que la V2 sous-estime volontairement la RUL pour réduire les maintenances tardives. Ses 5,1 % de surestimations sont une mesure de validation croisée sur FD001, pas une garantie pour un nouveau moteur.

Après installation, lancer le service depuis la racine du projet :

```powershell
.\.venv\Scripts\serve-maintenance-api.exe --config configs/api.toml
```

Le service écoute par défaut sur `http://127.0.0.1:8000`. Dans un second terminal PowerShell, contrôler son état puis construire une requête reproductible avec les 30 dernières observations brutes du moteur 3 préparé au jalon 1 :

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/health'

$unitId = 3
$sensorIds = 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
$rows = Import-Csv 'data/processed/FD001/validation_observed.csv' |
    Where-Object { [int]$_.unit_id -eq $unitId } |
    Sort-Object { [int]$_.cycle } |
    Select-Object -Last 30
$observations = foreach ($row in $rows) {
    $observation = [ordered]@{ cycle = [int]$row.cycle }
    foreach ($sensorId in $sensorIds) {
        $name = "sensor_$sensorId"
        $observation[$name] = [double]$row.$name
    }
    $observation
}
$body = @{ unit_id = $unitId; observations = @($observations) } |
    ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8000/predict/rul' `
    -ContentType 'application/json' `
    -Body $body | ConvertTo-Json -Depth 4
```

La forme de la réponse est la suivante ; les deux valeurs RUL ci-dessous sont illustratives :

```json
{
  "unit_id": 3,
  "history_start_cycle": 71,
  "last_cycle": 100,
  "observed_cycles": 30,
  "window_cycles_used": 30,
  "rul_estimate": 48.2,
  "rul_conservative": 27.6,
  "safety_base_prediction": 31.5687460699,
  "safety_margin": 3.9687460699,
  "decision_value": 27.6,
  "unit": "cycles",
  "central_model_version": "lstm-1e6e9aa46336",
  "safety_model_version": "hgb_quantile_005-6543e7b21788"
}
```

Une réponse `422` signale un historique invalide ; une réponse `503` indique que les bundles sont absents, modifiés ou incohérents. Les valeurs sont exprimées en cycles C-MAPSS, pas directement en heures ou en jours, et constituent un pronostic de durée restante plutôt qu'un diagnostic de la panne.

La recherche documentaire accepte une question en français ou en anglais et un `top_k` compris entre 1 et 10 :

```powershell
$body = @{
    query = "Que signifie la RUL dans les fichiers de test C-MAPSS ?"
    top_k = 5
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8000/documents/search' `
    -ContentType 'application/json' `
    -Body $body | ConvertTo-Json -Depth 5
```

Chaque résultat fournit le titre, la section, la page PDF éventuelle, le passage, l'URL publique, la réserve d'usage de la source et une citation directement affichable.

Pour obtenir directement une réponse sourcée :

```powershell
$body = @{
    question = "Quel critère arrête une simulation run-to-failure dans C-MAPSS ?"
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8000/documents/ask' `
    -ContentType 'application/json' `
    -Body $body | ConvertTo-Json -Depth 5
```

Le générateur local Qwen3-1.7B ne reçoit que les cinq passages récupérés. Il doit produire un JSON strict, citer au moins un de ces passages lorsqu'il répond et s'abstenir lorsque le corpus ne fournit pas de preuve. Une règle supplémentaire empêche de transformer une RUL seule en prescription de réparation. Sur le jeu de mise au point de 30 questions, les décisions répondre/s'abstenir et les références attendues atteignent 100 % après correction d'une annotation incomplète et ajout de cette règle. La similarité sémantique moyenne avec les réponses attendues vaut 0,915. Ce jeu a servi au développement : ces nombres vérifient le fonctionnement actuel, mais ne mesurent pas la généralisation sur de nouvelles questions. Sur le CPU local, une réponse générée prend environ 40 secondes en moyenne ; les abstentions détectées par règle sont quasi immédiates.

La planification accepte une flotte avec RUL conservatrice, durée, besoin en techniciens, criticité et coût préventif. L'échéance est la partie entière de la RUL et porte sur la fin de l'intervention. Les équipements dont l'échéance dépasse l'horizon sont différés. Pour les autres, l'API respecte simultanément la capacité de l'atelier et le nombre de techniciens, puis compare une heuristique de priorité à OR-Tools CP-SAT :

```powershell
$body = Get-Content 'data/scenarios/maintenance/constrained_load.json' -Raw
Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8000/maintenance/plan' `
    -ContentType 'application/json' `
    -Body $body | ConvertTo-Json -Depth 8
```

Sur ce scénario simulé, CP-SAT prouve un coût optimal de 5 110 contre 7 680 pour l'heuristique, soit −33,5 %. Il réduit le retard total de cinq à deux cycles. Un troisième scénario volontairement surchargé est correctement déclaré `infeasible`. Les unités de coût, durées, criticités et ressources sont des hypothèses de démonstration ; le résultat ne représente pas une économie mesurée.

Le [rapport de validation de l'API](reports/fd001/api/API_VALIDATION.md) conserve la preuve d'un lancement local avec les vrais artefacts, ainsi que la réponse obtenue pour une trajectoire de 30 cycles.

Pour construire et lancer l'API et le dashboard dans Docker Desktop, avec les bundles déjà présents dans `artifacts/models/FD001/` :

```powershell
docker compose up --build
```

Compose publie l'API sur `http://127.0.0.1:8000`, le dashboard sur `http://127.0.0.1:8501` et vérifie périodiquement la santé des deux services. Arrêter ensuite les services et supprimer les conteneurs avec :

```powershell
docker compose down
```

L'image contient les artefacts d'inférence figés et la trajectoire de démonstration sans cible RUL. Pendant sa construction, elle vérifie les cinq sources documentaires, régénère les 192 passages, télécharge les révisions figées du modèle d'embeddings et de Qwen3-1.7B, puis construit l'index. Le modèle génératif ajoute environ 4 Go au cache de construction. L'image ne relance aucun entraînement RUL. Toute modification d'un bundle, d'une source ou du manifeste impose de reconstruire l'image.

## Reproduire les jalons 1 à 5 sous Windows

Depuis la racine du projet, avec Python 3.12 installé et les fichiers bruts présents dans `CMAPSSData/`, utiliser PowerShell. La création de `.venv` ne se fait qu'à la première installation :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,models,api,ui,rag,planning]'
```

Le [snapshot Python 3.12](requirements-lock-win-py312.txt) conserve les versions exactes utilisées pour les jalons 1 à 3. Pour les réinstaller sans recalculer les dépendances :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-win-py312.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

L'installation portable ajoute les outils de test, PyTorch, Transformers, MLflow, FastAPI, Uvicorn, Streamlit, pypdf, Sentence Transformers et OR-Tools.

Reconstruire les données, exécuter le notebook, entraîner les candidats, puis reproduire le réentraînement et l'évaluation finale :

```powershell
.\.venv\Scripts\python.exe -m industrial_maintenance.prepare --config configs/fd001.toml
.\.venv\Scripts\python.exe scripts/execute_notebook.py
.\.venv\Scripts\train-fd001.exe --config configs/fd001_models.toml
.\.venv\Scripts\evaluate-fd001.exe --config configs/fd001_final.toml
.\.venv\Scripts\python.exe -m industrial_maintenance.train_safety --config configs/fd001_safety_v2.toml
.\.venv\Scripts\python.exe -m industrial_maintenance.calibrate_safety --config configs/fd001_safety_calibration.toml
.\.venv\Scripts\rag-maintenance.exe --config configs/rag.toml prepare
.\.venv\Scripts\rag-maintenance.exe --config configs/rag.toml index
.\.venv\Scripts\rag-maintenance.exe --config configs/rag.toml evaluate
.\.venv\Scripts\rag-maintenance.exe --config configs/rag.toml search "Comment la RUL est-elle définie ?"
.\.venv\Scripts\answer-maintenance.exe --config configs/rag.toml download-model
.\.venv\Scripts\answer-maintenance.exe --config configs/rag.toml ask "Quel critère arrête une simulation run-to-failure ?"
.\.venv\Scripts\python.exe -m industrial_maintenance.rag_generation_evaluation --config configs/rag.toml
.\.venv\Scripts\plan-maintenance.exe --config configs/maintenance_planning.toml data/scenarios/maintenance/constrained_load.json
.\.venv\Scripts\evaluate-maintenance-planning.exe --config configs/maintenance_planning.toml
```

La commande `train-fd001` lit uniquement les artefacts préparés à partir de `train_FD001.txt`. Elle écrit les modèles de sélection dans `artifacts/models/FD001/`, les résultats de validation dans `reports/fd001/models/` et les expériences MLflow locales dans `mlruns/`. La commande `evaluate-fd001` vérifie l'empreinte du résumé de sélection gelé, réentraîne le LSTM pendant six époques sur les 100 moteurs train, puis écrit le bundle V1 dans `artifacts/models/FD001/final/` et le bilan dans `reports/fd001/test/`. Les deux commandes `safety` n'ouvrent aucun fichier de test officiel ; elles écrivent la V2 dans `artifacts/models/FD001/safety_v2/` et `reports/fd001/safety_v2/`.

MLflow utilise une base SQLite locale `mlruns/mlflow.db` et un stockage d'artefacts séparé dans `mlruns/artifacts/`. Pour consulter les exécutions :

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri "sqlite:///./mlruns/mlflow.db" --port 5000
```

L'interface est alors disponible sur `http://127.0.0.1:5000`. L'appel explicite aux exécutables de `.venv` dispense d'activer l'environnement. Relancer la préparation ou l'entraînement régénère les sorties correspondantes sans modifier les fichiers bruts.

Pour ouvrir le notebook dans JupyterLab, ou directement dans VS Code avec l'interpréteur `.venv` :

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_fd001_exploration.ipynb
```

Exécuter les contrôles automatisés :

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Les tests utilisent de petites trajectoires et flottes synthétiques. Ils couvrent notamment les fuites entre moteurs, la causalité des fenêtres, le rechargement des modèles, les schémas HTTP, les citations, l'abstention, l'unicité des interventions, les capacités de l'atelier et des techniciens, les coûts et les scénarios irréalisables. Ils ne nécessitent ni serveur lancé séparément, ni service distant.

## État de finalisation

La première version proposée utilise C-MAPSS FD001 pour le pronostic, un corpus documentaire distinct pour le RAG et des scénarios explicitement simulés pour la planification.

FastAPI expose la double prédiction RUL, la recherche documentaire, la question-réponse sourcée et la comparaison des plannings. Elle applique la marge sérialisée au modèle V2 approprié et retourne les versions des artefacts. Le Dockerfile et la configuration Compose sont présents ; leur exécution doit encore être validée sur une machine disposant de Docker Desktop.

Le corpus RAG v1 est sélectionné et gelé dans [`configs/rag_corpus.toml`](configs/rag_corpus.toml). Ses cinq sources produisent 192 passages avec citations. La recherche hybride combine `multilingual-e5-small` et TF-IDF ; sur 25 questions répondables annotées, la bonne référence document-page apparaît dans le top 5 dans **100 %** des cas, dans le top 3 dans 76 % des cas et au premier rang dans 48 % des cas. Qwen3-1.7B génère ensuite une réponse contrainte aux passages ou s'abstient. L'évaluation de mise au point obtient 30/30 décisions correctes et 25/25 références attendues citées ; une évaluation indépendante reste nécessaire avant tout usage réel.

Les six jalons fonctionnels sont réalisés et documentés. Le dépôt contient le rapport de portfolio, le diagramme d'architecture, un scénario de démonstration et les formulations utilisables sur un CV. Les contrôles locaux couvrent les tests, le style, les dépendances et les configurations. La seule validation d'exécution restante concerne Docker Compose sur une machine équipée de Docker Desktop.

Les extensions possibles sont l'évaluation sur FD002/FD004, une évaluation RAG indépendante, la calibration d'un intervalle prédictif et le branchement du planificateur sur des calendriers et coûts réels.
