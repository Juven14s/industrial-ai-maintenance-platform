# Évaluation de la recherche documentaire

Date : 2026-10-08  
Modèle : `intfloat/multilingual-e5-small`  
Révision : `960310a7742a4e0a250dd720466bed84fee8caac`  
Configuration : top-5, au plus 5 passages par document.
Recherche hybride : similarité sémantique multilingue + TF-IDF, fusion RRF (poids 2.00/1.00, k=60), bonus lexical brut 0.08.

## Résultats

- Questions répondables : 25
- Questions volontairement sans réponse : 5
- Hit document@1/@3/@5 : 68.0% / 88.0% / 100.0%
- Hit référence page@1/@3/@5 : 48.0% / 76.0% / 100.0%
- MRR@5 : 0.649

La métrique de référence exige le bon document et, pour les PDF, une page située dans la plage annotée. Les cinq questions sans réponse servent au futur test d'abstention et ne sont pas comptées dans les métriques de récupération.

## Détail par question

| ID | Catégorie | Répondable | Premier passage pertinent |
|---|---|---:|---|
| `dataset_01` | dataset | oui | rang 1 |
| `dataset_02` | dataset | oui | rang 1 |
| `dataset_03` | dataset | oui | rang 1 |
| `dataset_04` | dataset | oui | rang 1 |
| `dataset_05` | dataset | oui | rang 1 |
| `damage_01` | degradation | oui | rang 2 |
| `damage_02` | degradation | oui | rang 4 |
| `damage_03` | degradation | oui | rang 5 |
| `damage_04` | degradation | oui | rang 1 |
| `damage_05` | evaluation | oui | rang 1 |
| `simulator_01` | simulator | oui | rang 4 |
| `simulator_02` | simulator | oui | rang 4 |
| `simulator_03` | simulator | oui | rang 1 |
| `simulator_04` | simulator | oui | rang 5 |
| `simulator_05` | simulator | oui | rang 2 |
| `review_01` | dataset | oui | rang 3 |
| `review_02` | dataset | oui | rang 1 |
| `review_03` | dataset | oui | rang 3 |
| `review_04` | evaluation | oui | rang 4 |
| `review_05` | evaluation | oui | rang 3 |
| `maintenance_01` | maintenance | oui | rang 1 |
| `maintenance_02` | maintenance | oui | rang 1 |
| `maintenance_03` | maintenance | oui | rang 2 |
| `maintenance_04` | maintenance | oui | rang 1 |
| `maintenance_05` | maintenance | oui | rang 3 |
| `unanswerable_01` | abstention | non | abstention à évaluer avec la génération |
| `unanswerable_02` | abstention | non | abstention à évaluer avec la génération |
| `unanswerable_03` | abstention | non | abstention à évaluer avec la génération |
| `unanswerable_04` | abstention | non | abstention à évaluer avec la génération |
| `unanswerable_05` | abstention | non | abstention à évaluer avec la génération |
