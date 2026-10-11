# Audit initial de CMAPSSData

Date : 8 septembre 2026. Périmètre : contrôle des fichiers locaux avant préparation et entraînement.

## Résultat

Les quatre sous-ensembles sont présents avec leurs fichiers d'entraînement, de test et de cibles RUL, soit 12 fichiers de données. Le dossier contient également `readme.txt` et `Damage Propagation Modeling.pdf`.

**Aucune erreur de structure détectée dans les fichiers de données contrôlés.** Une incohérence documentaire concerne les effectifs train/test de FD004 ; elle produit trois avertissements dans le manifeste, pour les fichiers train, test et RUL. Les fichiers bruts n'ont pas été modifiés.

| Sous-ensemble | Lignes train | Moteurs train | Lignes test | Moteurs test | Cibles RUL |
| --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | 20 631 | 100 | 13 096 | 100 | 100 |
| FD002 | 53 759 | 260 | 33 991 | 259 | 259 |
| FD003 | 24 720 | 100 | 16 596 | 100 | 100 |
| FD004 | 61 249 | 249 | 41 214 | 248 | 248 |

Ces valeurs proviennent du comptage des fichiers locaux. Une ligne de train/test représente un cycle d'un moteur ; une ligne du fichier RUL représente une cible pour un moteur test.

## Contrôles exécutés

- Lecture de toutes les lignes des 12 fichiers de données.
- Exactement 26 champs numériques finis par ligne train/test, séparés par des espaces ; aucune ligne vide.
- Identifiants et cycles entiers positifs ; moteurs contigus et ordonnés à partir de 1 dans chaque fichier.
- Cycles consécutifs à partir de 1 pour chaque moteur, sans doublon de couple moteur/cycle.
- Exactement une valeur entière non négative par ligne RUL.
- Nombre de lignes RUL égal au nombre réel de moteurs test pour chacun des quatre sous-ensembles.
- Calcul des empreintes SHA-256 des 12 fichiers de données et des deux fichiers de référence.

Les effectifs de FD004 ont également été recomptés avec une seconde méthode d'extraction des identifiants uniques. Le détail des contrôles, comptages et empreintes est enregistré dans [dataset_manifest.json](dataset_manifest.json).

## Schéma des données

| Colonnes | Noms proposés | Signification |
| --- | --- | --- |
| 1 | `unit_id` | Identifiant du moteur dans le fichier |
| 2 | `cycle` | Cycle de fonctionnement |
| 3 à 5 | `setting_1` à `setting_3` | Réglages opérationnels |
| 6 à 26 | `sensor_1` à `sensor_21` | Mesures des 21 capteurs |

Le readme se termine par « sensor measurement 26 » : il s'agit d'une coquille de numérotation. La colonne 26 correspond au capteur 21.

Les identifiants train et test appartiennent à des flottes distinctes : le moteur 1 du train n'est pas le moteur 1 du test. La ligne i du fichier RUL doit être associée au moteur i du fichier test, dont les identifiants contigus ont été vérifiés.

## Écart documentaire sur FD004

Le readme annonce 248 moteurs train et 249 moteurs test. Les fichiers donnent **249 moteurs train, 248 moteurs test et 248 cibles RUL**. Il ne faut donc pas permuter les fichiers pour suivre les nombres du readme.

Ces effectifs moteurs sont corroborés par Li, Goebel et Wu, *Degradation Modeling and Remaining Useful Life Prediction of Aircraft Engines Using Ensemble Learning*, section 4.1, dans une [publication hébergée par NASA](https://c3.ndc.nasa.gov/dashlink/static/media/publication/2018_DegradationModelingRULEnsemble_Wu.pdf). Cette référence annonce toutefois 57 522 observations d'entraînement, contre 61 249 dans le fichier local : elle étaye les effectifs moteurs, sans authentifier l'intégralité de notre archive. Aucun erratum officiel n'a été établi dans cet audit.

## Premières observations sur le train FD001

Les 100 trajectoires d'entraînement ont entre **128 et 362 cycles**. Six capteurs sont exactement constants sur ce fichier : `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18` et `sensor_19`. Le réglage `setting_3` est également constant.

Ce constat descriptif ne remplace pas la préparation expérimentale : toute sélection automatique de caractéristiques et toute normalisation devront être ajustées sur les seuls moteurs du groupe d'apprentissage, après séparation des moteurs de validation. Les caractéristiques de très faible variance restent à étudier dans cette étape.

Pour les trajectoires d'entraînement complètes, la cible brute sera calculée par `RUL = dernier_cycle_du_moteur - cycle`. Elle vaut zéro à la dernière observation. Pour le test, le dernier cycle observé précède la fin de vie : sa cible est celle du fichier RUL fourni. L'unité est le cycle opérationnel.

## Provenance et limites

Les données ont été fournies dans le dossier local par l'utilisateur ; l'URL exacte de téléchargement n'a pas encore été enregistrée. Les SHA-256 permettent de reconnaître cette copie locale, mais n'ont pas été comparés à une empreinte d'origine authentifiée indépendamment.

L'inspection du test et des fichiers RUL est limitée aux contrôles de structure et de correspondance des effectifs. Aucun entraînement, choix d'hyperparamètres ou calcul de métrique test n'a été effectué. Le PDF de référence a été inventorié et haché ; son contenu local n'a pas été analysé.

> **Mise à jour — bilan FD001 réalisé :** la préparation, la [comparaison ML/DL](fd001/models/MODEL_COMPARISON.md) et l'[évaluation officielle](fd001/test/OFFICIAL_TEST_EVALUATION.md) sont exécutées. Le LSTM a été retenu sur la validation interne avant l'ouverture des cibles officielles, puis réentraîné sur les 100 moteurs train. Il obtient une RMSE de 20,986 cycles et une MAE de 14,233 cycles sur les 100 moteurs test.

> **Mise à jour — modèle conservateur V2 :** le taux de surestimation de la V1 a motivé une nouvelle sélection utilisant uniquement `train_FD001.txt`. La [calibration croisée](fd001/safety_v2/SAFETY_CALIBRATION.md) obtient 5,1 % de surestimations hors pli, sans réutiliser `test_FD001.txt` ni `RUL_FD001.txt`.
