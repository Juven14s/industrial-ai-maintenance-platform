# Feuille de route du projet

Les jalons sont ordonnés par dépendance. Leur durée sera ajustée au temps disponible et aux premiers essais sur le PC local. Chaque étape doit produire une preuve vérifiable pour le portfolio.

## Jalon 1 — Données et protocole

**Avancement : réalisé et exécuté sur PC local.** Les fichiers FD001 à FD004 sont présents et l'[audit initial](../reports/DATASET_AUDIT.md) est terminé. L'environnement Python 3.12, le paquet réutilisable, la commande de préparation, les tests et le notebook sont maintenant en place.

Le protocole FD001 utilise 80 moteurs d'apprentissage et 20 moteurs de validation issus du seul fichier officiel `train_FD001.txt`. La validation est tronquée avant la fin de vie en conservant la RUL calculée auparavant. La sélection des capteurs et la normalisation sont ajustées uniquement sur l'apprentissage : 15 capteurs sont conservés après retrait de six constantes exactes.

Les fenêtres produites ont les formes `(16779, 30, 15)` pour l'apprentissage et `(20, 30, 15)` pour la validation. Les historiques courts sont complétés à gauche avec leur première observation ; une longueur observée permet d'identifier ce remplissage. Les graines de séparation et de troncature sont respectivement 42 et 43.

Livrables disponibles :

- [Notebook d'exploration exécuté et commenté](../notebooks/01_fd001_exploration.ipynb).
- [Dictionnaire des données](DICTIONNAIRE_DONNEES.md) et [manifeste des fichiers bruts](../reports/dataset_manifest.json).
- [Configuration de préparation](../configs/fd001.toml) et [rapport FD001](../reports/fd001/PREPARATION_FD001.md).
- Préparation dans `src/industrial_maintenance/`, données dérivées dans `data/processed/FD001/` et tests dans `tests/`.
- [Commandes Windows de reproduction, d'exécution du notebook, des modèles, de l'API et des tests](../README.md#reproduire-les-jalons-1-à-5-sous-windows).

Critère de passage atteint : la préparation est exécutable depuis les fichiers bruts, avec groupes disjoints, prétraitement appris sur les seuls moteurs d'apprentissage et sorties traçables. À l'issue de ce jalon, le test officiel restait réservé au bilan final.

## Jalon 2 — Première comparaison ML/DL

**Avancement : réalisé et exécuté sur PC local.**

La référence constante, trois configurations de Random Forest et un LSTM compact ont été entraînés sur le protocole fixé. L'apprentissage retient 15 179 fenêtres se terminant à partir du cycle 20 et strictement avant la panne. Les moteurs ont le même poids total ; les 15 capteurs utilisent des fenêtres de 30 cycles et le remplissage est ignoré à l'aide de `observed_lengths`. La validation conserve une prédiction finale pour chacun des 20 moteurs tenus à l'écart.

| Modèle | MAE de validation | RMSE de validation |
| --- | ---: | ---: |
| Référence constante pondérée | 36,387 | 46,684 |
| Random Forest `depth12_leaf5` | 22,836 | 30,970 |
| LSTM compact | **18,378** | **23,196** |

Le LSTM est sélectionné sur cette validation. Sa meilleure époque est la 6e sur 13 exécutées ; son biais moyen est de +6,801 cycles, son erreur absolue au 90e percentile de 35,392 cycles, son taux de surestimation de 65 % et son score NASA de 342,950.

Sa RMSE est inférieure de 7,774 cycles, soit 25,1 %, à celle du meilleur Random Forest. Cette amélioration justifie le choix provisoire du modèle séquentiel sur le protocole actuel.

Livrables disponibles :

- [Configuration d'entraînement](../configs/fd001_models.toml) et commande `train-fd001` documentée dans le [README](../README.md#reproduire-les-jalons-1-à-5-sous-windows).
- [Rapport de comparaison](../reports/fd001/models/MODEL_COMPARISON.md), [tableau des métriques](../reports/fd001/models/comparison.csv) et [prédictions par moteur](../reports/fd001/models/validation_predictions.csv).
- [Figures de validation](../reports/fd001/models/figures/) et résumé structuré des expériences.
- Suivi MLflow local avec une base SQLite et artefacts rechargeables sur CPU.

Critère de passage atteint : la comparaison est reproductible hors notebook et le choix provisoire du LSTM est justifié par les mesures. La validation reste limitée à 20 moteurs, dont seulement trois avec une RUL inférieure ou égale à 30 cycles. Ces résultats servent à sélectionner le modèle ; ils ne constituent pas une évaluation finale. `test_FD001.txt` et `RUL_FD001.txt` n'ont pas été lus par l'entraînement.

### Bilan final sur le test officiel

Après gel du choix du LSTM et de sa durée d'entraînement à six époques, le prétraitement et le modèle ont été réajustés sur les 100 moteurs de `train_FD001.txt` et 18 631 fenêtres admissibles. L'évaluation unique sur les 100 moteurs officiels de test donne une **MAE de 14,233 cycles** et une **RMSE de 20,986 cycles**. Le biais moyen est de +7,850 cycles, l'erreur absolue au 90e percentile de 36,439 cycles et le taux de surestimation de 73 %.

Le [rapport d'évaluation officielle](../reports/fd001/test/OFFICIAL_TEST_EVALUATION.md), la [configuration gelée](../configs/fd001_final.toml), les [prédictions par moteur](../reports/fd001/test/official_test_predictions.csv) et le bundle final rendent ce bilan vérifiable. Le score NASA de 14 432,738 est une somme sur 100 moteurs ; il ne doit pas être comparé directement à la somme obtenue sur les 20 moteurs de validation. Le test ayant désormais été consulté, il ne servira pas à ajuster cette version du modèle.

### Version 2 orientée sécurité

Le taux de surestimation officiel de 73 % a conduit à reformuler l'objectif métier, sans réutiliser les cibles de test. Une validation croisée à cinq plis sépare les moteurs, réajuste le prétraitement dans chaque pli et mesure neuf horizons de 5 à 150 cycles. La fonction de sélection facture une surestimation cinq fois plus cher qu'une sous-estimation. Des Gradient Boosting quantiles remplacent l'objectif MSE, puis une marge de sécurité est calibrée de façon croisée : le pli évalué ne participe jamais au calcul de sa propre marge.

Le modèle conservateur retenu utilise le quantile 0,05 et une marge finale de 3,969 cycles. Ses prédictions hors pli ont une MAE de 20,029 cycles, une RMSE de 29,220, un biais de −19,573 et un taux de surestimation de **5,1 %**. Le taux de surestimation supérieur à 10 cycles est de **0,5 %** ; le pire pli atteint 10,4 % et le pire horizon 15,9 %. Cette baisse du risque se paie par une sous-estimation systématique et ne constitue pas une garantie probabiliste sur de futurs moteurs.

Livrables : [rapport de sélection](../reports/fd001/safety_v2/SAFETY_SELECTION.md), [rapport de calibration](../reports/fd001/safety_v2/SAFETY_CALIBRATION.md), [comparaison complète](../reports/fd001/safety_v2/calibrated_comparison.csv), [configuration de sélection](../configs/fd001_safety_v2.toml), [configuration de calibration](../configs/fd001_safety_calibration.toml) et bundle `artifacts/models/FD001/safety_v2/`.

## Jalon 3 — Première tranche d'API

**Avancement : API d'inférence et dashboard Streamlit réalisés et testés localement ; Dockerfile et configuration Compose créés, exécution des conteneurs à valider selon la disponibilité de Docker Desktop.**

Le service FastAPI charge au démarrage les bundles V1 et V2, vérifie leurs manifestes et leurs empreintes, puis conserve les modèles en mémoire. Il valide un historique brut appartenant à un seul moteur, impose des cycles consécutifs et des valeurs finies, reprend le remplissage des fenêtres courtes et applique les prétraitements sérialisés. Aucun réentraînement ni cible RUL n'intervient dans une requête.

`POST /predict/rul` retourne l'estimation ponctuelle du LSTM, la prédiction de base du Gradient Boosting quantile, sa marge de 3,969 cycles, la RUL conservatrice et une `decision_value` égale à cette dernière. Les versions des deux artefacts accompagnent la réponse. La marge est appliquée uniquement au modèle V2 ; elle n'est pas soustraite au LSTM et les deux résultats ne constituent pas un intervalle prédictif.

Livrables disponibles : [`GET /health`, `GET /models`, `POST /predict/rul` et documentation OpenAPI](../src/industrial_maintenance/api.py), [pipeline d'inférence](../src/industrial_maintenance/inference.py), [configuration du service](../configs/api.toml), commandes `serve-maintenance-api` et `serve-maintenance-dashboard`, [interface Streamlit](../src/industrial_maintenance/dashboard_app.py), [trajectoire de démonstration](../examples/fd001_demo_engine_3.csv), [rapport de validation API](../reports/fd001/api/API_VALIDATION.md), [rapport de validation du dashboard](../reports/fd001/ui/DASHBOARD_VALIDATION.md), [Dockerfile](../Dockerfile) et [service Compose](../compose.yaml). L'image embarque le code, l'exemple et les artefacts d'inférence figés ; elle n'acquiert pas les données et ne réentraîne pas les modèles.

Critère d'inférence atteint : une même entrée donne les mêmes prédictions via le pipeline local et via l'API, y compris après rechargement des artefacts. Les historiques incohérents produisent une réponse `422` et les bundles indisponibles ou altérés une réponse `503`. Le critère complet du jalon sera atteint après un démarrage Compose et un contrôle de santé réussis.

### Interface de démonstration Streamlit

Le jalon d'intégration comprend désormais un dashboard Streamlit connecté à l'API FastAPI par HTTP. L'utilisateur peut choisir un moteur FD001 de démonstration ou importer un CSV, fixer le cycle d'observation, visualiser les trajectoires des capteurs et déclencher la prédiction. L'interface présente la RUL conservatrice en premier, la RUL centrale en complément, l'échéance prudente en cycles et les métadonnées des modèles.

Le lancement local utilise deux processus : `serve-maintenance-api --config configs/api.toml` sur le port 8000, puis `serve-maintenance-dashboard --api-url http://127.0.0.1:8000 --port 8501` sur le port 8501. Docker Compose décrit également ces deux services et expose les ports uniquement sur l'interface locale ; ce parcours reste à exécuter sur une machine équipée de Docker, absent de l'environnement local actuel.

Le dashboard reste une interface de démonstration. Il permet aussi de poser une question documentaire, d'afficher la réponse, l'abstention et les passages cités, puis d'éditer une flotte et de comparer deux plannings. La définition de seuils métier et la traduction des cycles en dates calendaires nécessitent encore des données industrielles réelles.

## Jalon 4 — RAG évalué

**Avancement : réalisé et évalué localement.**

Les cinq sources publiques du [manifeste](../configs/rag_corpus.toml) sont vérifiées par taille et SHA-256, puis extraites sur les seules plages de pages retenues. Le découpage produit 192 passages qui conservent le document, la section, les pages, l'URL et la limite d'usage. L'index dense utilise la révision figée de `multilingual-e5-small`; un score TF-IDF et une fusion RRF complètent la recherche pour mieux traiter les termes techniques exacts.

Le [jeu d'évaluation](../data/evaluation/rag/questions_v1.jsonl) contient 30 questions françaises, dont 25 répondables annotées au niveau document-page et cinq volontairement sans réponse. Après calibration sur ce jeu, le [rapport](../reports/rag/RETRIEVAL_EVALUATION.md) mesure un hit référence@1 de 48 %, @3 de 76 % et @5 de 100 %, avec une MRR@5 de 0,649. Ces mesures décrivent le jeu de mise au point actuel ; une évaluation indépendante devra confirmer la généralisation si le corpus ou le récupérateur change.

`GET /documents/status` expose la version de l'index, `POST /documents/search` retourne les passages avec citations et réserves d'usage, et `POST /documents/ask` produit une réponse française ou une abstention. Qwen3-1.7B est chargé localement à une révision figée, fonctionne sans service externe et ne peut citer que les passages effectivement récupérés. Une règle métier refuse aussi de déduire une réparation précise à partir d'une RUL lorsque les sources ne donnent aucune procédure.

Le [rapport de génération](../reports/rag/GENERATION_EVALUATION.md) mesure 100 % de sorties valides, 30 décisions répondre/s'abstenir correctes sur 30, 25 références attendues citées sur 25 réponses et une similarité sémantique moyenne de 0,915. La latence CPU moyenne est de 39,5 secondes. Le [test de bout en bout](../reports/rag/API_VALIDATION.md) charge réellement les bundles RUL, l'index, le modèle d'embeddings et Qwen, puis contrôle une réponse citée et une abstention. Ces mesures portent sur le jeu de mise au point et ne constituent pas une estimation indépendante de généralisation.

Livrables disponibles : manifeste du corpus, commandes de préparation/indexation/recherche/génération, jeu de questions, rapports séparés de récupération et de génération, route FastAPI et tests de contrat. Le critère de passage est atteint : les erreurs de recherche, de décision et de citation sont mesurées séparément et les réponses exposent leurs passages sources.

## Jalon 5 — Optimisation sous contraintes

**Avancement : réalisé et évalué localement sur des scénarios simulés.**

Trois flottes versionnées couvrent une charge normale, une capacité contrainte et un horizon volontairement irréalisable. L'échéance correspond à la fin de l'intervention et vaut la partie entière de la RUL conservatrice. Le coût simulé additionne maintenance préventive, immobilisation, anticipation et retard pondéré par la criticité. Les ressources portent sur la capacité simultanée de l'atelier et le nombre de techniciens.

L'heuristique traite les échéances proches et les criticités élevées en priorité. OR-Tools CP-SAT utilise des variables d'intervalle et deux contraintes cumulatives, avec une limite de dix secondes et un seul worker pour la reproductibilité. Sur le scénario contraint, il prouve un optimum de 5 110 unités contre 7 680 pour l'heuristique, soit **33,5 % de coût simulé en moins**, et réduit le retard total de cinq à deux cycles. Sur le scénario surchargé, le statut `infeasible` distingue clairement une requête valide d'un planning impossible.

Livrables disponibles : [configuration](../configs/maintenance_planning.toml), [scénarios](../data/scenarios/maintenance/), [planificateur](../src/industrial_maintenance/planning.py), [évaluation](../src/industrial_maintenance/planning_evaluation.py), [rapport et diagramme de Gantt](../reports/planning/PLANNING_EVALUATION.md), [validation API](../reports/planning/API_VALIDATION.md), route `POST /maintenance/plan` et section de planification dans Streamlit.

Critère de passage atteint : les plans réalisables respectent indépendamment les capacités, les coûts des deux méthodes sont calculés avec les mêmes règles, le statut d'optimalité est exposé et le cas sans solution est traité explicitement. Les gains restent présentés comme des résultats de simulation.

## Jalon 6 — Livraison du portfolio

**Avancement : finalisation locale réalisée ; exécution Docker Compose restante.**

Les versions de dépendances sont verrouillées, les instructions de reproduction sont consolidées et les contrôles locaux réussissent. Une intégration continue pourra reprendre ces commandes une fois le dépôt hébergé. Docker Compose reste à lancer sur une machine équipée de Docker Desktop, absent du poste actuel.

Livrables disponibles : [rapport technique synthétique](../reports/PROJECT_REPORT.md), [diagramme d'architecture](ARCHITECTURE.md), [scénario de démonstration](DEMONSTRATION.md), [guide de présentation portfolio](PORTFOLIO.md), [validation finale locale](../reports/FINAL_VALIDATION.md), résultats mesurés et interface Streamlit.

Le scénario de démonstration charge un historique, obtient une RUL, interroge la documentation avec ses sources, puis compare deux plannings de la flotte simulée. Il rappelle les limites à chaque étape sans confondre pronostic, diagnostic et prescription.

## Mise en valeur sur le CV et en entretien

Le projet peut désormais être présenté comme un **projet personnel réalisé**, en précisant que la validation Docker reste à effectuer.

Formulation utilisable pendant la réalisation :

> Industrial AI Maintenance Platform — Projet personnel sur NASA C-MAPSS : comparaison Random Forest/LSTM, modèle quantile conservateur réduisant les surestimations à 5,1 % en validation croisée, RAG local avec réponses citées et abstention, puis planification sous contraintes par OR-Tools CP-SAT, exposés via FastAPI et Streamlit.

Quatre résultats peuvent être cités avec leur contexte : **RMSE de 20,986 cycles et MAE de 14,233 cycles sur les 100 moteurs du test officiel FD001 pour la V1**, **5,1 % de surestimations en validation croisée par moteur pour la RUL conservatrice V2**, **30/30 décisions et 25/25 références attendues correctes sur le jeu de mise au point RAG**, puis **33,5 % de réduction du coût sur un scénario de planification simulé contraint**. Il faut expliciter le compromis de la V2, la portée non indépendante de l'évaluation RAG et le caractère entièrement simulé du gain de planification.

Questions auxquelles le dépôt et la présentation doivent permettre de répondre :

- Pourquoi ces données et quels équipements représentent-elles ?
- Comment les fuites entre moteurs, fenêtres et phases de prétraitement sont-elles évitées ?
- Le LSTM apporte-t-il un gain par rapport à un modèle plus simple ?
- Comment vérifier qu'une réponse RAG est effectivement soutenue par ses citations ?
- Quelle fonction l'optimiseur minimise-t-il, et quelles hypothèses de simulation utilise-t-il ?
- Quelles commandes permettent de reproduire les résultats et de lancer la démonstration ?
