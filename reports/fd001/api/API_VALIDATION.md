# Validation de l'API d'inférence FD001

Date de l'exécution : **7 octobre 2026**  
Environnement : Windows, Python 3.12.10, exécution locale sur CPU

## Périmètre vérifié

Le service a été lancé avec la commande installée par le projet :

```powershell
.\.venv\Scripts\serve-maintenance-api.exe --config configs/api.toml --port 8011
```

Le démarrage a rechargé les bundles réels du LSTM V1 et du modèle conservateur V2. Le chargeur a vérifié leurs manifestes, leurs contrats d'entrée et leurs empreintes SHA-256 avant que le serveur accepte des requêtes.

| Contrôle HTTP | Résultat |
| --- | --- |
| `GET /health` | `200`, statut `ok` |
| `GET /models` | `200`, 15 capteurs requis, fenêtre de 30 cycles, cycle minimal 20 |
| `GET /docs` | `200` |
| `POST /predict/rul` avec 30 cycles valides | `200`, deux prédictions produites |
| `POST /predict/rul` avec un dernier cycle égal à 19 | `422`, historique refusé |

## Prédiction de contrôle

La requête utilise le moteur 3 de `validation_observed.csv`, des cycles 71 à 100. Seuls le numéro de cycle et les 15 capteurs requis ont été envoyés ; la cible `rul` du fichier n'a pas été incluse dans la requête.

```json
{
  "unit_id": 3,
  "history_start_cycle": 71,
  "last_cycle": 100,
  "observed_cycles": 30,
  "window_cycles_used": 30,
  "rul_estimate": 97.34269714355469,
  "safety_base_prediction": 78.95684865714281,
  "safety_margin": 3.9687460699329264,
  "rul_conservative": 74.98810258720988,
  "decision_value": 74.98810258720988,
  "unit": "cycles",
  "central_model_version": "lstm-1e6e9aa46336",
  "safety_model_version": "hgb_quantile_005-6543e7b21788"
}
```

Cette exécution confirme que la marge de sécurité est appliquée à la sortie du modèle V2 : `78.95684865714281 - 3.9687460699329264 = 74.98810258720988`. Elle n'est pas appliquée à la sortie LSTM.

## Contrôles automatisés

```text
pytest : 185 tests réussis, dont 29 dédiés au dashboard et à sa frontière HTTP
ruff   : tous les contrôles réussis
```

Les tests couvrent notamment les schémas HTTP, les historiques invalides, le rechargement des bundles, les empreintes modifiées, l'ordre des variables, la transformation des fenêtres et l'égalité entre l'inférence locale et l'inférence exposée par l'API.

## Conteneur

Le `Dockerfile`, `.dockerignore` et `compose.yaml` sont présents. Le build du conteneur n'a pas été exécuté dans cet environnement, car la commande `docker` n'y est pas installée. Il reste donc à lancer `docker compose up --build` sur une machine équipée de Docker Desktop avant de déclarer l'image validée.
