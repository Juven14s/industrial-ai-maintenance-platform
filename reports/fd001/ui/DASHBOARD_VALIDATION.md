# Validation du dashboard Streamlit

Date de l'exécution initiale : **7 octobre 2026**  
Extension documentaire vérifiée : **9 octobre 2026**  
Environnement : Windows, Python 3.12.10, Streamlit 1.65.0, exécution locale sur CPU

## Parcours vérifié

L'API et le dashboard ont été lancés avec les commandes installées par le projet :

```powershell
.\.venv\Scripts\serve-maintenance-api.exe --config configs/api.toml
.\.venv\Scripts\serve-maintenance-dashboard.exe --api-url http://127.0.0.1:8000 --port 8501 --headless
```

| Contrôle | Résultat |
| --- | --- |
| Santé de l'API sur `http://127.0.0.1:8000/health` | `200`, statut `ok` |
| Santé de Streamlit sur `http://127.0.0.1:8501/_stcore/health` | `200`, corps `ok` |
| Page du dashboard sur `http://127.0.0.1:8501/` | `200`, contenu HTML |
| Chargement de la trajectoire FD001 de démonstration | 100 cycles, 15 capteurs, aucune cible RUL |
| Connexion du dashboard à FastAPI | Réussie |
| Clic automatisé sur `Estimer la RUL` | Réussi |
| Affichage des cartes de résultat | RUL de décision, estimation centrale et fenêtre analysée présentes |
| Indisponibilité réseau et erreurs `422` | Traduites en messages utilisateur et couvertes par les tests |

Le test de composant Streamlit a exécuté le script réel, vérifié le titre, le statut connecté, actionné le bouton de prédiction et contrôlé la présence des métriques après la réponse. Le client du dashboard communique exclusivement avec FastAPI par HTTP ; il n'importe ni Torch, ni le predictor, ni les prétraitements.

## Prédiction obtenue depuis l'interface

La trajectoire embarquée correspond au moteur 3, cycles 1 à 100. Le fichier contient uniquement `unit_id`, `cycle` et les 15 capteurs requis. L'API reçoit les 100 observations et utilise sa fenêtre finale de 30 cycles.

```json
{
  "unit_id": 3,
  "history_start_cycle": 1,
  "last_cycle": 100,
  "observed_cycles": 100,
  "window_cycles_used": 30,
  "rul_estimate": 97.34269714355469,
  "rul_conservative": 74.98810258720988,
  "safety_base_prediction": 78.95684865714281,
  "safety_margin": 3.9687460699329264,
  "decision_value": 74.98810258720988,
  "unit": "cycles",
  "central_model_version": "lstm-1e6e9aa46336",
  "safety_model_version": "hgb_quantile_005-6543e7b21788"
}
```

Le dashboard affiche `rul_conservative` comme RUL de décision et `rul_estimate` comme estimation centrale distincte. Il ne les présente pas comme un intervalle et n'invente aucune conversion en jours.

## Extension documentaire

Le dashboard comprend désormais une quatrième étape qui appelle `POST /documents/ask` avec un délai adapté à la génération CPU. Elle affiche séparément une réponse ou une abstention, puis chaque citation avec le passage, l'URL publique et sa limite d'usage. Le client HTTP de cette nouvelle action est couvert par un test déterministe. Le parcours réel de l'API a aussi été vérifié avec une réponse citant la page PDF 6 et une abstention sans citation ; le clic Streamlit de cette extension n'a pas été rejoué dans un navigateur lors de cette mise à jour.

## Extension de planification

Une cinquième étape propose une flotte éditable avec RUL conservatrice, durée, besoin en techniciens, criticité et coût préventif. Lorsque la prédiction courante est disponible, sa RUL conservatrice initialise le premier moteur. L'interface permet de régler l'horizon, les techniciens et la capacité de l'atelier, puis affiche les coûts, le statut du solveur, les moteurs différés ou non planifiés, les tableaux d'affectation et deux diagrammes de Gantt interactifs. Le client HTTP `POST /maintenance/plan` est couvert par un test déterministe ; la route réelle a été exécutée avec OR-Tools sur le scénario contraint. Le script Streamlit complet a aussi été rendu avec `AppTest` : les cinq sections sont présentes et aucune exception d'interface n'a été levée.

## Contrôles automatisés

```text
Tests propres au dashboard : 31 réussis
Suite complète du projet    : 222 réussis
Ruff                        : tous les contrôles réussis
pip check                   : aucune dépendance cassée
API RAG réelle              : réponse citée et abstention réussies
Compose                     : non exécuté, Docker absent du poste
```

Les tests couvrent les colonnes manquantes, les valeurs non finies, les cycles dupliqués ou non consécutifs, les coupures autorisées pour les historiques courts et complets, la sélection du moteur et du cycle, le filtrage de la cible, les types JSON, les appels `health`, `models`, `predict` et `documents/ask`, les erreurs réseau, les réponses `422` et les réponses JSON invalides.

## Limites actuelles

L'interface constitue une démonstration des pipelines de pronostic, de question-réponse et de planification. Les criticités, coûts, ressources et durées du planning sont simulés ; les cycles ne sont pas convertis en dates calendaires et aucun ordre de travail réel n'est généré. Les fichiers Docker ont été étendus aux deux services et incluent les modèles RAG et OR-Tools, mais le build reste non exécuté localement car Docker n'est pas installé sur cette machine.
