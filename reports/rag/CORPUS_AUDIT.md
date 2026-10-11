# Audit du corpus RAG — version 1

Date de gel : 8 octobre 2026  
Identifiant : `cmapss-maintenance-v1`  
Manifeste machine : `configs/rag_corpus.toml`

## Décision

Le corpus initial contient cinq sources distinctes et atteint le minimum fixé dans le cahier des charges. Il couvre trois périmètres qui devront rester séparés dans les réponses : description du dataset, fonctionnement du simulateur et maintenance aéronautique générale.

| Source | Rôle principal | Autorité | Réserve essentielle |
|---|---|---|---|
| README NASA C-MAPSS | Format FD001-FD004, cycles, train/test et RUL | Primaire | Settings anonymisés, aucune procédure de maintenance |
| Damage Propagation Modeling | Génération de la dégradation, capteurs, fin de vie | Primaire | Décrit surtout le jeu PHM08, distinct de FD001-FD004 |
| Guide C-MAPSS v1 | Architecture, variables, défauts et détérioration | Primaire | Ne définit pas les colonnes FD001 ni les actions de maintenance |
| Review on CMAPSS Dataset | Différences FD001-FD004 et pratiques d'évaluation | Secondaire | Revue scientifique, pas manuel opérationnel |
| FAA Powerplant, chapitre 10 | Inspection et maintenance moteur génériques | Primaire | Ne concerne aucun moteur C-MAPSS particulier |

Les empreintes SHA-256, tailles, origines, dates, nombres de pages, plages retenues et limites d'usage sont enregistrés dans le manifeste. Toute modification d'un fichier devra invalider l'index et imposer sa reconstruction.

Le README officiel contient deux anomalies conservées comme telles dans le fichier source et signalées dans les métadonnées : il inverse les effectifs réels train/test de FD004 (`249/248`) et nomme par erreur la dernière variable « sensor measurement 26 ». Le schéma réel comporte 26 colonnes au total, dont 21 capteurs après l'identifiant, le cycle et les trois settings.

Pour réduire le bruit, l'index v1 retiendra les pages PDF 7–12 et 27–31 du guide C-MAPSS, 1–8 des deux articles, et 44–65 du chapitre FAA consacré aux turbines. Les pages bibliographiques, administratives, les tutoriels GUI datés et la partie FAA sur les moteurs à pistons seront exclues.

## Règles de récupération et de réponse

- Les questions sur les fichiers, cycles, jeux train/test et RUL doivent privilégier le README et la revue de 2014.
- Les questions sur la simulation, les composants et la génération de la dégradation doivent privilégier le guide C-MAPSS et l'article de 2008.
- Les questions de maintenance générale peuvent utiliser le chapitre FAA, avec la mention explicite de son caractère générique.
- Le système doit s'abstenir si une question demande une procédure constructeur, un seuil de réparation propre à FD001 ou une correspondance non documentée des settings.
- Une prédiction de RUL ne doit jamais être transformée en diagnostic de panne ou en prescription de réparation par le RAG.

## Documents exclus du corpus scientifique

Les fichiers de pilotage du projet (`CAHIER_DES_CHARGES`, `FEUILLE_DE_ROUTE`, README du dépôt et rapports produits par le projet) ne sont pas indexés. Leur inclusion créerait des citations circulaires. Le dictionnaire de données interne pourra être ajouté plus tard avec le statut explicite de source synthétique, après vérification de chacune de ses affirmations.

Le guide C-MAPSS version 2 et le jeu PHM08 Challenge ne sont pas retenus dans cette version : le premier est redondant et décrit une version différente du simulateur, tandis que le second constitue un dataset distinct.

## Prochaine étape

L'extraction, le découpage, l'indexation et la génération sont terminés. Les cinq sources produisent 192 passages et la recherche hybride atteint 100 % de hit document-page au top 5 sur les 25 questions répondables du jeu de mise au point. Les routes FastAPI `POST /documents/search` et `POST /documents/ask` exposent les passages, les réponses et leurs citations.

Qwen3-1.7B ne reçoit que les passages récupérés et retourne un objet structuré qui doit contenir une décision, une réponse et des identifiants de citation valides. L'évaluation obtient 25 réponses avec la référence attendue et cinq abstentions sur les cinq questions hors corpus. Une règle explicite interdit de transformer une RUL seule en prescription de réparation. Ces résultats ont été obtenus sur le jeu de mise au point ; la prochaine amélioration scientifique serait une évaluation indépendante, idéalement annotée par une autre personne.
