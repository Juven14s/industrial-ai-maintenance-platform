# Rapport technique synthétique

**Projet :** Industrial AI Maintenance Platform  
**Périmètre :** NASA C-MAPSS FD001, documentation technique publique et scénarios de maintenance simulés  
**État :** version portfolio finalisée localement

## Objectif

Le projet relie trois problèmes souvent traités séparément : prévoir la durée de vie restante d'un équipement, retrouver des informations dans une documentation technique et organiser les interventions sous contraintes. Une API FastAPI expose ces fonctions et un dashboard Streamlit permet de les démontrer sans construire de requêtes JSON.

## Données et protocole ML

FD001 contient des trajectoires multivariées run-to-failure de turboréacteurs simulés. Une observation correspond à un moteur et à un cycle de fonctionnement. Après suppression de six capteurs constants, chaque exemple utilise 30 cycles et 15 capteurs. Les moteurs sont séparés par groupe avant le calcul du prétraitement afin d'éviter qu'une même trajectoire soit présente dans l'apprentissage et la validation.

Le protocole distingue trois usages :

1. 80 moteurs d'apprentissage et 20 moteurs de validation pour choisir la V1 ;
2. réentraînement sur les 100 moteurs d'apprentissage, puis une unique évaluation sur le test officiel ;
3. validation croisée groupée sur les données d'apprentissage pour concevoir la V2 orientée sécurité, sans régler cette V2 sur le test officiel déjà consulté.

## Résultats de prédiction

| Modèle et protocole | MAE | RMSE | Biais | Surestimation |
| --- | ---: | ---: | ---: | ---: |
| LSTM V1, test officiel FD001 | 14,233 | 20,986 | +7,850 | 73,0 % |
| HGB quantile V2, prédictions hors pli | 20,029 | 29,220 | −19,573 | 5,1 % |

La V1 est l'estimation centrale la plus précise sur le test officiel. Son biais positif la rend toutefois risquée pour décider seule d'une intervention. La V2 combine un HistGradientBoosting au quantile 0,05, une marge calibrée de 3,969 cycles et un plancher à zéro. Elle réduit nettement les surestimations au prix d'une erreur moyenne plus élevée et d'interventions potentiellement anticipées. Elle sert donc de valeur de décision, sans être présentée comme une garantie statistique pour un futur moteur.

## Recherche documentaire augmentée

Le corpus est extrait en conservant titre, section, page, origine et réserve d'usage. La recherche hybride combine embeddings multilingues E5 et TF-IDF. Un Qwen3-1.7B local génère une réponse en français à partir des passages récupérés. Le système contrôle les références produites et s'abstient lorsque les preuves sont insuffisantes.

Sur le jeu de développement de 30 questions, la recherche récupère les passages attendus dans les cinq premiers résultats pour 25 questions couvertes sur 25. La chaîne générative prend la bonne décision de réponse ou d'abstention pour 30 cas sur 30, conserve les 25 références attendues et atteint une similarité moyenne de 0,915. La latence CPU moyenne mesurée est de 39,5 secondes. Ce jeu a servi au développement et ne constitue pas une évaluation indépendante définitive.

## Optimisation du planning

La flotte, les ressources et les coûts sont décrits par des scénarios simulés. Pour chaque équipement, la partie entière de la RUL conservatrice fixe l'échéance de fin d'intervention. Les contraintes cumulatives limitent les places d'atelier et le nombre de techniciens. Le coût additionne maintenance préventive, immobilisation, intervention anticipée et retard pondéré par la criticité.

Trois scénarios vérifient une charge normale, une charge contrainte et un cas irréalisable. Sur le scénario contraint, l'heuristique produit un coût de 7 680, trois interventions en retard et cinq cycles de retard cumulé. CP-SAT prouve une solution optimale de coût 5 110, avec deux interventions en retard et deux cycles cumulés. Le gain de 33,5 % décrit uniquement cette simulation. Sur le cas impossible, le service signale explicitement l'absence de planning complet.

## Service et interface

FastAPI expose l'état du service, les métadonnées des modèles, la prédiction RUL, la recherche, la réponse documentaire et la planification. Les requêtes sont validées par Pydantic. Les bundles ML et l'index sont chargés une fois, leurs manifestes et empreintes sont vérifiés, et aucune étape d'entraînement n'est exécutée en ligne.

Streamlit consomme exclusivement cette API. L'utilisateur peut importer un historique, choisir un cycle, observer les capteurs, obtenir les deux RUL, interroger les documents et éditer une flotte à planifier. Cette séparation évite de dupliquer le prétraitement et la logique métier dans l'interface.

## Reproductibilité et validation

Les configurations TOML fixent les graines, découpages, chemins, paramètres de modèles et coûts. MLflow suit les expériences d'entraînement. Les évaluations produisent des rapports Markdown, des données tabulaires et des figures vérifiables. La suite automatisée compte 222 tests réussis ; Ruff et `pip check` passent également. Le fichier de verrouillage fige les dépendances utilisées localement.

Le `Dockerfile` et `compose.yaml` décrivent l'API et le dashboard, avec contrôles de santé. Leur exécution réelle reste à valider car Docker Desktop n'est pas installé sur la machine de développement.

## Limites et suite possible

FD001 représente une seule condition opératoire et un seul mode de dégradation simulé. Une généralisation industrielle demanderait des données réelles, une validation temporelle externe, une définition métier du cycle et une surveillance de dérive. Le corpus documentaire demeure restreint et son jeu de questions a participé au développement. Les coûts de maintenance et les capacités sont hypothétiques. Une suite pertinente consisterait à tester FD002/FD004, réserver un nouveau jeu RAG à l'évaluation finale, calibrer une incertitude prédictive et relier le planificateur à des calendriers réels.

## Conclusion

La plateforme constitue un prototype de bout en bout : données multivariées, modèles comparés, objectif de sécurité explicite, RAG sourcé, optimisation sous contraintes, API, interface et tests. Ses résultats sont reproductibles dans le périmètre étudié et ses limites sont rendues visibles afin d'éviter d'interpréter une simulation académique comme une validation industrielle.

