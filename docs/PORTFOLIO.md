# Présenter le projet dans un portfolio

## Formulation CV en français

Version courte :

> Conception d'une plateforme de maintenance prédictive de bout en bout sur NASA C-MAPSS : prévision de RUL par LSTM et Gradient Boosting quantile, RAG local sourcé, planification CP-SAT, API FastAPI, dashboard Streamlit, configuration Docker et 222 tests automatisés.

Version avec résultats :

> Développé une plateforme de maintenance prédictive sur séries temporelles multivariées : LSTM à 14,23 cycles de MAE sur le test FD001, modèle conservateur ramenant les surestimations à 5,1 % en validation croisée, RAG local avec citations et optimisation CP-SAT réduisant de 33,5 % le coût d'un scénario simulé contraint. Industrialisation avec FastAPI, Streamlit, MLflow, Docker et tests automatisés.

## English résumé version

> Built an end-to-end predictive maintenance platform on NASA C-MAPSS: multivariate RUL forecasting with LSTM and quantile Gradient Boosting, source-grounded local RAG, CP-SAT maintenance scheduling, FastAPI, Streamlit, MLflow, Docker configuration, and 222 automated tests. Achieved 14.23-cycle MAE on the FD001 official test and reduced overestimation to 5.1% under grouped cross-validation with a safety-oriented model.

## Présentation en 30 secondes

> J'ai construit une plateforme qui part de séries de capteurs et produit deux estimations de durée de vie : une estimation centrale et une valeur plus prudente pour la décision. Cette valeur alimente ensuite un planificateur de maintenance sous contraintes. J'ai aussi ajouté un assistant documentaire local qui cite ses sources et sait s'abstenir. L'ensemble est exposé par FastAPI et une interface Streamlit, avec des expériences suivies, des tests et une configuration Docker. Le point central du projet est le compromis explicite entre précision et risque de surestimer la durée restante.

## Points à expliquer en entretien

- La séparation par moteur évite qu'une partie d'une trajectoire fuite vers la validation.
- Une fenêtre glissante de 30 cycles fournit au modèle la dynamique récente des capteurs.
- La V1 et la V2 répondent à deux objectifs différents : précision centrale et décision prudente.
- Le test officiel n'a servi qu'au bilan de la V1 ; la V2 a été calibrée en validation croisée sur l'apprentissage.
- Les citations RAG restent attachées aux passages extraits, et l'abstention protège les questions non couvertes.
- CP-SAT garantit le respect des contraintes et peut prouver l'optimalité, selon son statut de résolution.
- Le chiffre de 33,5 % vient d'un scénario simulé et ne représente pas une économie observée en usine.

## Démonstration recommandée

Suivre le [scénario de démonstration](DEMONSTRATION.md), puis ouvrir le [rapport technique](../reports/PROJECT_REPORT.md) pour commenter les résultats et leurs limites. Le [diagramme d'architecture](ARCHITECTURE.md) aide à expliquer la séparation entre préparation hors ligne, services FastAPI et interface.

