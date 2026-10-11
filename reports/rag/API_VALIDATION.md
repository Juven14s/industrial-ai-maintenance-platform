# Validation de l'API documentaire

Date : 9 octobre 2026  
Configuration : `configs/api.toml` et `configs/rag.toml`

## Vérification réelle

L'application FastAPI complète a été démarrée dans son cycle de vie de test avec les deux bundles RUL, l'index documentaire, le modèle d'embeddings et Qwen3-1.7B chargés depuis le cache local. Les appels suivants ont réellement été exécutés :

| Route | Statut | Résultat contrôlé |
|---|---:|---|
| `GET /health` | 200 | Bundles `lstm-1e6e9aa46336` et `hgb_quantile_005-6543e7b21788` chargés |
| `GET /documents/status` | 200 | Index `multilingual-e5-small-ccc49cee9290-7bb45c2e36b4`, 192 passages |
| `POST /documents/search` | 200 | Trois passages sourcés retournés pour une question française sur la RUL |
| `POST /documents/ask` | 200 | Réponse « H(t) = 0 » avec citation de l'article, page PDF 6 |
| `POST /documents/ask` | 200 | Abstention sans citation pour une demande de réparation déduite de la seule RUL |

Le premier résultat de la recherche provenait de l'article *Damage Propagation Modeling for Aircraft Engine Prognostics*, page PDF 1, section « II. PROGNOSTICS ». Le README officiel C-MAPSS figurait au rang 2 avec le passage qui définit la RUL des données de test. Chaque résultat contenait aussi l'URL publique et la limite d'usage enregistrée dans le manifeste.

Pour la question « Quel critère arrête une simulation run-to-failure dans C-MAPSS ? », la route de réponse a cité le passage « Stop when health H = 0 », page PDF 6. Pour la question demandant la réparation exacte du moteur 17 à partir d'une RUL de 20 cycles, elle a retourné `status="abstained"`, aucune citation et le message d'absence de preuve. La version de génération exposée était `Qwen3-1.7B-8d4744f9e130-procedure-guard-v1`.

## Contrôles automatisés

Les tests vérifient le statut de l'index, les schémas de requête, les bornes de `top_k`, le refus des champs inconnus, la structure des citations, les réponses et abstentions, ainsi que les réponses `503` lorsque la recherche ou la génération n'est pas disponible. Ils utilisent des substituts déterministes pour ne pas charger les modèles pendant les tests de contrat HTTP.

## Limites

La génération sur CPU demande plusieurs dizaines de secondes. Le jeu de 30 questions a servi à développer le récupérateur et les règles d'abstention ; le résultat de 100 % documenté dans le rapport de génération n'est donc pas une mesure indépendante. Docker n'étant pas installé sur le poste, le même parcours reste à confirmer dans Compose.
