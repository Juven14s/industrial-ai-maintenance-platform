# Validation finale locale

**Date :** 29 octobre 2026
**Environnement :** Windows, Python 3.12.10, environnement virtuel `.venv`

## Résultat

| Contrôle | Résultat |
| --- | --- |
| Suite automatisée | **281 tests réussis** en 23,38 s |
| Qualité du code | `ruff check .` réussi |
| Dépendances installées | `pip check` : aucune incompatibilité |
| Fichiers de livraison modifiés | liens Markdown locaux contrôlés, aucune cible absente |
| Données structurées versionnées | 23 JSON et 13 TOML chargés sans erreur |
| Commandes installées | API, dashboard, agent, évaluation agentique, MCP et planification présentes |
| Compose | YAML valide, services `maintenance-api` et `maintenance-dashboard` détectés |
| Exécution Docker | non réalisée : Docker est absent du poste |

Un avertissement de dépréciation externe provenant de SQLAlchemy via MLflow apparaît pendant les tests. Il ne provoque aucun échec.

## Validations fonctionnelles déjà conservées

- [API de prédiction](fd001/api/API_VALIDATION.md) : chargement des deux bundles et prédiction HTTP cohérente avec l'inférence locale.
- [Dashboard](fd001/ui/DASHBOARD_VALIDATION.md) : rendu Streamlit et parcours de prédiction.
- [RAG](rag/API_VALIDATION.md) : réponse citée et abstention avec les modèles réellement chargés.
- [Planification](planning/API_VALIDATION.md) : scénario contraint envoyé à la route HTTP et solution CP-SAT optimale.
- [Évaluation du planning](planning/PLANNING_EVALUATION.md) : comparaison de trois scénarios, contraintes et cas irréalisable.
- [Contrat d'orchestration agentique](agent/ORCHESTRATION_EVALUATION.md) : six scénarios déterministes sur le routage, l'interruption humaine, les citations et la RUL conservatrice.

Un smoke test supplémentaire a chargé les vrais bundles RUL sur la trajectoire FD001 de démonstration, exécuté les tendances puis la prédiction, observé l'interruption obligatoire et repris le workflow après approbation. La valeur conservatrice obtenue était de 74,99 cycles pour ce point d'observation.

Le parcours agentique a également été vérifié dans un navigateur réel avec l'API et le dashboard Streamlit lancés localement. Pour l'incident `dashboard-3-100`, l'interface a exécuté `sensor_trends → predict_rul`, affiché 97,34 cycles de RUL centrale et 74,99 cycles de RUL conservatrice, imposé la validation humaine, puis affiché la recommandation finale avec le statut `approved`. Après l'arrêt du serveur, la commande `run-maintenance-agent status dashboard-3-100` a rouvert le checkpoint SQLite et retrouvé la décision `approve`, la recommandation finale et une liste d'interruptions vide. Ce contrôle confirme la reprise persistante du workflow ; il ne constitue pas une mesure de qualité industrielle du modèle.

## Limites de livraison

La syntaxe de `compose.yaml` est valide, mais le build, les contrôles de santé des conteneurs et le parcours Streamlit à travers le réseau Compose doivent être vérifiés sur une machine équipée de Docker Desktop.

Le dossier est initialisé comme dépôt Git sur la branche `main`. Une intégration continue GitHub reprend les commandes de lint, de test et de contrôle des dépendances sur Windows avec Python 3.12. Le benchmark agentique actuel emploie des services factices déterministes et ne remplace pas une évaluation de bout en bout sur des incidents réservés.

