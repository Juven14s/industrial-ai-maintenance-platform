# Validation finale locale

**Date :** 11 octobre 2026  
**Environnement :** Windows, Python 3.12.10, environnement virtuel `.venv`

## Résultat

| Contrôle | Résultat |
| --- | --- |
| Suite automatisée | **222 tests réussis** en 14,37 s |
| Qualité du code | `ruff check .` réussi |
| Dépendances installées | `pip check` : aucune incompatibilité |
| Liens Markdown locaux | 37 fichiers contrôlés, aucun lien cassé |
| Données structurées du projet | 44 JSON et 12 TOML chargés sans erreur |
| Commandes installées | API, dashboard, planification et évaluation présentes |
| Compose | YAML valide, services `maintenance-api` et `maintenance-dashboard` détectés |
| Exécution Docker | non réalisée : Docker est absent du poste |

Deux avertissements de dépréciation externes apparaissent pendant les tests : l'un vient de l'intégration `TestClient` de FastAPI/Starlette, l'autre de SQLAlchemy via MLflow. Ils ne provoquent aucun échec.

## Validations fonctionnelles déjà conservées

- [API de prédiction](fd001/api/API_VALIDATION.md) : chargement des deux bundles et prédiction HTTP cohérente avec l'inférence locale.
- [Dashboard](fd001/ui/DASHBOARD_VALIDATION.md) : rendu Streamlit et parcours de prédiction.
- [RAG](rag/API_VALIDATION.md) : réponse citée et abstention avec les modèles réellement chargés.
- [Planification](planning/API_VALIDATION.md) : scénario contraint envoyé à la route HTTP et solution CP-SAT optimale.
- [Évaluation du planning](planning/PLANNING_EVALUATION.md) : comparaison de trois scénarios, contraintes et cas irréalisable.

## Limites de livraison

La syntaxe de `compose.yaml` est valide, mais le build, les contrôles de santé des conteneurs et le parcours Streamlit à travers le réseau Compose doivent être vérifiés sur une machine équipée de Docker Desktop.

Le dossier est initialisé comme dépôt Git sur la branche `main`. Une intégration continue GitHub reprend les commandes de lint, de test et de contrôle des dépendances sur Windows avec Python 3.12.

