# Présenter le projet dans un portfolio

## Formulation CV en français

Version courte :

> Conception d'une plateforme agentique de maintenance prédictive sur NASA C-MAPSS : RUL par LSTM et Gradient Boosting quantile, RAG local sourcé, planification CP-SAT et orchestration LangGraph supervisée avec checkpoints, outils MCP, FastAPI, Streamlit, configuration Docker et 281 tests automatisés.

Version avec résultats :

> Développé une plateforme agentique de maintenance prédictive sur séries temporelles multivariées : LSTM à 14,23 cycles de MAE sur le test FD001, modèle conservateur ramenant les surestimations à 5,1 % en validation croisée, RAG local avec citations et optimisation CP-SAT réduisant de 33,5 % le coût d'un scénario simulé contraint. Orchestration LangGraph avec validation humaine, checkpoints SQLite et outils MCP ; contrat de routage et de sécurité validé sur 6 scénarios déterministes.

## English résumé version

> Built an agentic predictive-maintenance platform on NASA C-MAPSS: multivariate RUL forecasting with LSTM and quantile Gradient Boosting, source-grounded local RAG, CP-SAT scheduling, and a human-supervised LangGraph workflow with SQLite checkpoints and MCP tools. Achieved 14.23-cycle MAE on the FD001 official test, reduced overestimation to 5.1% under grouped cross-validation, and validated the orchestration contract on 6 deterministic scenarios with 281 automated tests.

## Présentation en 30 secondes

> J'ai construit une plateforme qui part de séries de capteurs et produit deux estimations de durée de vie : une estimation centrale et une valeur plus prudente pour la décision. Un workflow LangGraph peut ensuite résumer les tendances, appeler la RUL, interroger un RAG sourcé et proposer un planning sous contraintes. Il conserve son état et s'interrompt avant toute recommandation finale pour laisser la décision à une personne. FastAPI, Streamlit et MCP exposent ces capacités. Le point central reste le compromis explicite entre précision, sécurité et contrôle humain.

## Points à expliquer en entretien

- La séparation par moteur évite qu'une partie d'une trajectoire fuite vers la validation.
- Une fenêtre glissante de 30 cycles fournit au modèle la dynamique récente des capteurs.
- La V1 et la V2 répondent à deux objectifs différents : précision centrale et décision prudente.
- Le test officiel n'a servi qu'au bilan de la V1 ; la V2 a été calibrée en validation croisée sur l'apprentissage.
- Les citations RAG restent attachées aux passages extraits, et l'abstention protège les questions non couvertes.
- CP-SAT garantit le respect des contraintes et peut prouver l'optimalité, selon son statut de résolution.
- Le chiffre de 33,5 % vient d'un scénario simulé et ne représente pas une économie observée en usine.
- Le routeur agentique actuel est déterministe : Qwen génère les réponses documentaires mais ne choisit pas les outils.
- Une planification déclenche toujours la RUL et reçoit uniquement la valeur conservatrice ; le brouillon doit ensuite être approuvé, modifié ou rejeté.
- Les 6/6 scénarios agentiques mesurent un contrat logiciel avec des services factices, pas un taux de réussite industriel.

## Démonstration recommandée

Suivre le [scénario de démonstration](DEMONSTRATION.md), puis ouvrir le [rapport technique](../reports/PROJECT_REPORT.md) pour commenter les résultats et leurs limites. Le [diagramme d'architecture](ARCHITECTURE.md) aide à expliquer la séparation entre préparation hors ligne, services FastAPI et interface.

