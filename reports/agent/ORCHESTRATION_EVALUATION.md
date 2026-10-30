# Évaluation du contrat d'orchestration agentique

**Suite :** `maintenance-agent-routing-safety-v1`

**Politique :** `maintenance-agent-v1`

**Routeur :** `KeywordRouter`
**Exécution locale :** 29 octobre 2026

## Objet

Cette évaluation vérifie le comportement du workflow LangGraph indépendamment de la qualité des modèles métier. Les six scénarios utilisent des services factices déterministes, sans réseau, modèle ML, modèle génératif ni solveur réel. Les résultats mesurent donc un **contrat d'orchestration et de sécurité**, pas une performance industrielle de bout en bout.

## Résultats

| Contrôle | Résultat |
| --- | ---: |
| Scénarios entièrement conformes | **6 / 6** |
| Routes d'outils exactement conformes | **6 / 6** |
| Statuts initiaux conformes | **6 / 6** |
| Politique d'interruption humaine conforme | **6 / 6** |
| Résultats finaux attendus | **6 / 6** |
| Tâches réalisables menées jusqu'à l'approbation | **5 / 5** |
| Politique de citations ou d'abstention | **3 / 3** |
| Usage exclusif de la RUL conservatrice pour le planning | **2 / 2** |

La latence d'orchestration seule est de **8,175 ms en moyenne**, avec un **p95 nearest-rank de 13,754 ms** sur cette exécution locale. Cette mesure exclut explicitement l'inférence RUL, la recherche, Qwen et OR-Tools ; elle ne doit pas être présentée comme la latence de la plateforme réelle.

## Scénarios couverts

1. Routage lexical vers tendances capteurs puis RUL.
2. Réponse documentaire répondable avec citation.
3. Abstention documentaire sans citation inventée.
4. Planification précédée de la RUL et alimentée par `decision_value == rul_conservative`.
5. Route complète explicitement demandée, dans l'ordre imposé.
6. Demande de planning incomplète : échec structuré sans interruption humaine inutile.

Chaque scénario vérifie également le statut initial, la présence ou l'absence attendue de l'interruption LangGraph et, pour les tâches réalisables, la reprise après une approbation humaine.

## Reproduire

```powershell
.\.venv\Scripts\python.exe -m industrial_maintenance.agent_eval --scenarios evals\agent_scenarios.json --output reports\agent\orchestration_evaluation.json
```

Le processus retourne un code non nul dès qu'un scénario échoue. Les preuves détaillées, routes observées, statuts, contrôles de sûreté et latences se trouvent dans [`orchestration_evaluation.json`](orchestration_evaluation.json).

## Interprétation correcte

On peut affirmer que le workflow respecte **6 scénarios contractuels versionnés sur 6** dans ce harnais déterministe. On ne peut pas en déduire un taux de réussite sur des incidents industriels, une précision de diagnostic, une qualité RAG indépendante ou une latence opérationnelle. Une future évaluation de bout en bout devra exécuter des incidents réservés avec les vrais artefacts RUL, le vrai corpus, Qwen et OR-Tools.
