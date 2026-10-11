# Validation de l'API de planification

Date : 9 octobre 2026  
Configuration : `configs/api.toml` et `configs/maintenance_planning.toml`

## Appel réel

La route `POST /maintenance/plan` a été appelée dans le cycle de vie FastAPI avec le véritable service de planification et OR-Tools CP-SAT 9.15. Le scénario `constrained_load` contient cinq interventions, un seul emplacement d'atelier et deux techniciens.

| Contrôle | Résultat |
|---|---:|
| Statut HTTP | 200 |
| Statut de l'heuristique | `feasible` |
| Coût simulé de l'heuristique | 7 680 |
| Statut CP-SAT | `optimal` |
| Coût simulé CP-SAT | 5 110 |
| Réduction simulée | 2 570, soit 33,5 % |
| Interventions retournées | 5 |

La réponse expose pour chaque moteur le début, la fin, l'échéance, la durée, le besoin en techniciens, la criticité, l'anticipation, le retard et le coût. Le statut `optimal` signifie que CP-SAT a prouvé l'optimalité pour ce scénario et cette fonction de coût.

## Contrat et erreurs

Pydantic refuse les champs inconnus, les ressources non positives, les RUL négatives, les nombres non finis et les listes vides. La couche métier refuse aussi les identifiants de moteur dupliqués. Un scénario valide mais surchargé retourne un résultat métier `infeasible` avec un statut HTTP 200 ; il ne s'agit pas d'une requête invalide.

## Portée

Les coûts, ressources, durées et criticités sont simulés. Cette validation prouve l'intégration technique du solveur à l'API ; elle ne représente pas un gain économique observé sur une flotte industrielle.
