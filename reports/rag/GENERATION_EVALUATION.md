# Évaluation de la génération RAG

Date : 2026-10-09  
Récupérateur : `multilingual-e5-small-ccc49cee9290-7bb45c2e36b4`  
Générateur : `Qwen3-1.7B-8d4744f9e130-procedure-guard-v1`

## Résultats

- Sorties valides : 100.0%
- Décisions répondre/s'abstenir correctes : 100.0%
- Questions répondables effectivement répondues : 100.0%
- Questions sans réponse correctement refusées : 100.0%
- Référence attendue parmi les citations des réponses : 100.0%
- Similarité sémantique moyenne avec la réponse attendue : 0.915
- Latence CPU moyenne / médiane / maximale : 39.5 / 45.3 / 60.0 secondes

La similarité sémantique est un indicateur automatique, pas une preuve de fidélité factuelle. Le hit de référence vérifie le document et la page annotés. Le même jeu ayant servi à mettre au point le récupérateur, ces résultats ne sont pas une estimation indépendante de généralisation.

Pendant la mise au point, une demande de réparation fondée sur la seule RUL a révélé un excès de confiance. La politique finale refuse ce type de prescription lorsqu'aucun passage ne contient de procédure. L'annotation de `damage_02` a aussi été complétée avec la page 1, dont le résumé énonce explicitement la réponse. Ces deux corrections expliquent pourquoi le résultat final doit rester présenté comme une validation de développement.

## Détail

| ID | Attendu | Produit | Bonne référence citée | Latence |
|---|---|---|---:|---:|
| `dataset_01` | réponse | réponse | oui | 23.3 s |
| `dataset_02` | réponse | réponse | oui | 19.4 s |
| `dataset_03` | réponse | réponse | oui | 21.6 s |
| `dataset_04` | réponse | réponse | oui | 34.6 s |
| `dataset_05` | réponse | réponse | oui | 44.2 s |
| `damage_01` | réponse | réponse | oui | 46.4 s |
| `damage_02` | réponse | réponse | oui | 50.7 s |
| `damage_03` | réponse | réponse | oui | 34.7 s |
| `damage_04` | réponse | réponse | oui | 60.0 s |
| `damage_05` | réponse | réponse | oui | 48.3 s |
| `simulator_01` | réponse | réponse | oui | 48.2 s |
| `simulator_02` | réponse | réponse | oui | 34.6 s |
| `simulator_03` | réponse | réponse | oui | 53.6 s |
| `simulator_04` | réponse | réponse | oui | 49.5 s |
| `simulator_05` | réponse | réponse | oui | 49.6 s |
| `review_01` | réponse | réponse | oui | 41.0 s |
| `review_02` | réponse | réponse | oui | 27.5 s |
| `review_03` | réponse | réponse | oui | 37.9 s |
| `review_04` | réponse | réponse | oui | 52.3 s |
| `review_05` | réponse | réponse | oui | 55.9 s |
| `maintenance_01` | réponse | réponse | oui | 46.7 s |
| `maintenance_02` | réponse | réponse | oui | 49.6 s |
| `maintenance_03` | réponse | réponse | oui | 52.2 s |
| `maintenance_04` | réponse | réponse | oui | 50.1 s |
| `maintenance_05` | réponse | réponse | oui | 52.0 s |
| `unanswerable_01` | abstention | abstention | non | 22.9 s |
| `unanswerable_02` | abstention | abstention | non | 0.2 s |
| `unanswerable_03` | abstention | abstention | non | 25.3 s |
| `unanswerable_04` | abstention | abstention | non | 26.9 s |
| `unanswerable_05` | abstention | abstention | non | 27.3 s |
