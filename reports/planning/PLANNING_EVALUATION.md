# Évaluation de la planification de maintenance

Date : 2026-10-09  
Version : `maintenance-planning-v1`  
Solveur : OR-Tools CP-SAT, un seul worker, limite de 10 secondes

## Résultat principal

Sur le scénario contraint, l'heuristique obtient un coût simulé de 7680, contre 5110 pour CP-SAT, soit une réduction de 2570 (33.5 %). Le scénario normal est résolu sans retard par les deux méthodes. Le scénario surchargé est déclaré irréalisable par CP-SAT car toutes les interventions obligatoires ne tiennent pas dans l'horizon.

| Scénario | Méthode | Statut | Coût | Interventions tardives | Retard total | Non planifiées | Temps |
|---|---|---|---:|---:|---:|---:|---:|
| `constrained_load` | heuristic | feasible | 7680 | 3 | 5 | 0 | 0.0002 s |
| `constrained_load` | cp_sat | optimal | 5110 | 2 | 2 | 0 | 0.0102 s |
| `infeasible_load` | heuristic | partial | 43500 | 1 | 2 | 1 | 0.0000 s |
| `infeasible_load` | cp_sat | infeasible | — | 0 | 0 | 3 | 0.0009 s |
| `normal_load` | heuristic | feasible | 2570 | 0 | 0 | 0 | 0.0004 s |
| `normal_load` | cp_sat | optimal | 2570 | 0 | 0 | 0 | 0.0037 s |

## Fonction de coût et convention

L'échéance porte sur la **fin** de l'intervention et vaut la partie entière de la RUL conservatrice. Le coût additionne le coût préventif, 50 unités par cycle d'immobilisation, 10 unités par cycle d'anticipation et 250 unités par cycle de retard, multipliées par une criticité de 1, 2 ou 4. Les équipements dont l'échéance dépasse l'horizon sont différés et exclus du coût courant.

## Portée

Les coûts, durées, ressources et criticités sont des hypothèses simulées. La réduction mesurée n'est ni une économie industrielle observée ni une validation du planning sur une flotte réelle. L'intérêt de l'expérience est de vérifier la chaîne RUL conservatrice → échéance → contraintes → décision et de comparer les méthodes avec exactement les mêmes règles.
