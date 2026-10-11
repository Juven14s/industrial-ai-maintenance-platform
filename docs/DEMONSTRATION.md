# Scénario de démonstration

Ce parcours présente le projet en six à huit minutes. Il suppose que les dépendances, les artefacts ML et l'index documentaire ont déjà été préparés.

## 1. Démarrer les services

Depuis la racine du projet, ouvrir deux terminaux PowerShell.

```powershell
# Terminal 1
.\.venv\Scripts\serve-maintenance-api.exe --config configs/api.toml
```

```powershell
# Terminal 2
.\.venv\Scripts\serve-maintenance-dashboard.exe --api-url http://127.0.0.1:8000 --port 8501
```

Ouvrir <http://127.0.0.1:8501>. Le premier démarrage de l'API peut prendre plus de temps car les modèles RUL, l'index et le modèle génératif local sont chargés en mémoire.

## 2. Présenter la prédiction RUL

1. Montrer le statut de l'API.
2. Garder la trajectoire de démonstration proposée et choisir un cycle d'observation valide.
3. Montrer que les courbes décrivent l'évolution des capteurs jusqu'à ce cycle.
4. Lancer la prédiction.
5. Expliquer la différence entre l'estimation centrale V1 et la RUL conservatrice V2 utilisée pour la décision.

Message à faire passer : la V1 recherche la précision centrale, tandis que la V2 accepte des interventions plus précoces afin de limiter les surestimations dangereuses. Les résultats restent exprimés en cycles C-MAPSS.

## 3. Interroger la documentation

Utiliser par exemple la question suivante :

> Quel critère arrête une simulation run-to-failure dans C-MAPSS ?

Déplier les passages cités et montrer le document, la section et la page. Poser ensuite une question hors corpus, par exemple sur le couple de serrage d'une pièce absente des documents, afin d'illustrer l'abstention.

Message à faire passer : la réponse est limitée aux preuves récupérées et chaque référence peut être vérifiée.

## 4. Construire un planning

1. Ouvrir la section de planification.
2. Utiliser la RUL conservatrice obtenue pour un équipement de la flotte.
3. Garder plusieurs interventions en concurrence pour des ressources limitées.
4. Lancer la comparaison.
5. Comparer le coût, le nombre de retards et la chronologie de l'heuristique avec CP-SAT.

Message à faire passer : l'optimisation respecte les mêmes contraintes que l'heuristique et améliore le scénario contraint enregistré, mais les coûts et ressources sont simulés.

## 5. Conclure avec les résultats

Présenter quatre résultats mesurés :

- V1 sur le test officiel FD001 : MAE de 14,233 cycles et RMSE de 20,986 cycles ;
- V2 hors pli : 5,1 % de surestimations, avec un biais prudent de −19,573 cycles ;
- RAG sur le jeu de développement : 30 décisions sur 30 et 25 références attendues sur 25 ;
- planification contrainte : coût simulé réduit de 7 680 à 5 110 et retard cumulé réduit de 5 à 2 cycles.

Terminer en indiquant les limites : un seul sous-jeu C-MAPSS, aucune donnée de maintenance réelle, corpus documentaire restreint, coûts de planning simulés et conteneurs non exécutés sur le poste actuel.

## Solution de repli

Si Streamlit n'est pas disponible, ouvrir <http://127.0.0.1:8000/docs> pour appeler les routes depuis Swagger. Les évaluations déjà produites restent consultables dans les rapports RUL, RAG et planification. Pour vérifier seulement l'optimiseur sans charger les modèles de langage :

```powershell
.\.venv\Scripts\plan-maintenance.exe `
  --config configs/maintenance_planning.toml `
  --scenario data/scenarios/maintenance/constrained_load.json `
  --method both
```

