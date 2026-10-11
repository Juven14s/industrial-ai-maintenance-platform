# Dictionnaire des données C-MAPSS

Ce document décrit les colonnes brutes utilisées pour FD001 et les variables créées par notre préparation. Les noms `sensor_1` à `sensor_21` restent ceux du code : la correspondance physique ci-dessous sert à interpréter les observations, sans convertir automatiquement leurs unités.

## Structure et rôle des fichiers

Une ligne contient l'état d'un moteur à un cycle de fonctionnement. Ce cycle est une unité d'utilisation, sans durée en heures fournie. Un moteur peut fonctionner avec une dégradation progressive ; la fin de vie correspond au critère opérationnel de la simulation. Les trajectoires commencent avec une usure initiale variable, pas nécessairement à l'état neuf. [Readme fourni](../CMAPSSData/readme.txt), [article de référence, sections V–VI](../CMAPSSData/Damage%20Propagation%20Modeling.pdf).

| Fichier | Contenu | Usage dans notre protocole |
| --- | --- | --- |
| `train_FD001.txt` | Trajectoires jusqu'à la fin de vie | Séparation par moteur en apprentissage et validation |
| `test_FD001.txt` | Autres moteurs, observés avant leur fin de vie | Réservé au bilan final ; non chargé à l'étape 1 |
| `RUL_FD001.txt` | Une RUL par moteur test, à son dernier cycle observé | Cibles du bilan final ; non chargé à l'étape 1 |

Les identifiants sont propres à chaque fichier : le moteur 1 du train et le moteur 1 du test sont deux équipements distincts. Les 26 colonnes se lisent dans l'ordre du tableau suivant. Le readme contient une coquille à sa dernière ligne de schéma : la colonne 26 est bien le **capteur 21**, puisque les cinq premières colonnes décrivent l'identifiant, le cycle et les trois réglages.

## Les 26 colonnes brutes

La correspondance des 21 capteurs suit le tableau 2 de l'[article fourni avec les données](../CMAPSSData/Damage%20Propagation%20Modeling.pdf). Le [guide du simulateur NASA, tableaux 1.2–1.4](https://ntrs.nasa.gov/api/citations/20070034949/downloads/20070034949.pdf?attachment=true) précise les symboles et met en évidence les réserves décrites après le tableau. Les numéros du guide désignent le vecteur de sortie du simulateur et ne sont pas les numéros de colonne de nos fichiers.

| Colonne | Nom dans le code | Symbole | Sens physique ou rôle | Unité / réserve |
| ---: | --- | --- | --- | --- |
| 1 | `unit_id` | — | Identifiant du moteur dans ce fichier | Entier ; jamais une variable d'entrée |
| 2 | `cycle` | — | Compteur des cycles observés | Cycle ; sert à l'ordre temporel |
| 3 | `setting_1` | Alt | Altitude de fonctionnement | Échelle du fichier à confirmer [A] |
| 4 | `setting_2` | Mach | Nombre de Mach | Sans dimension |
| 5 | `setting_3` | TRA | Position de commande des gaz | Degrés dans le guide [B] |
| 6 | `sensor_1` | T2 | Température totale à l'entrée de la soufflante | °R |
| 7 | `sensor_2` | T24 | Température totale en sortie du compresseur basse pression | °R |
| 8 | `sensor_3` | T30 | Température totale en sortie du compresseur haute pression | °R |
| 9 | `sensor_4` | T50 | Température totale en sortie de turbine basse pression | °R |
| 10 | `sensor_5` | P2 | Pression à l'entrée de la soufflante | psia |
| 11 | `sensor_6` | P15 | Pression totale dans la veine de dérivation | psia |
| 12 | `sensor_7` | P30 | Pression totale en sortie du compresseur haute pression | psia |
| 13 | `sensor_8` | Nf | Vitesse de rotation de la soufflante | tr/min |
| 14 | `sensor_9` | Nc | Vitesse de rotation du corps haute pression | tr/min |
| 15 | `sensor_10` | epr | Rapport de pression moteur : P50 / P2 | Sans dimension |
| 16 | `sensor_11` | Ps30 | Pression statique en sortie du compresseur haute pression | psia |
| 17 | `sensor_12` | phi | Débit de carburant rapporté à Ps30 | (lbm/s) / psi |
| 18 | `sensor_13` | NRf | Vitesse corrigée de la soufflante | tr/min |
| 19 | `sensor_14` | NRc | Vitesse corrigée du corps haute pression | tr/min |
| 20 | `sensor_15` | BPR | Rapport de dilution | Sans dimension |
| 21 | `sensor_16` | farB | Rapport carburant/air dans la chambre de combustion | Sans dimension |
| 22 | `sensor_17` | htBleed | Enthalpie de l'air prélevé | Unité non indiquée [C] |
| 23 | `sensor_18` | Nf_dmd | Consigne de vitesse de la soufflante | tr/min |
| 24 | `sensor_19` | PCNfR_dmd | Consigne de vitesse corrigée de la soufflante | Incohérence tr/min / pourcentage [D] |
| 25 | `sensor_20` | W31 | Débit d'air prélevé pour refroidir la turbine haute pression | lbm/s |
| 26 | `sensor_21` | W32 | Débit d'air de refroidissement de turbine | lbm/s ; composant ambigu [E] |

`°R` désigne le degré Rankine, `psia` la livre-force par pouce carré en pression absolue et `lbm/s` un débit massique en livres par seconde. « Vitesse corrigée » désigne une grandeur ajustée aux conditions thermodynamiques ; ce n'est pas une correction numérique que nous appliquons au dataset. Les colonnes dites capteurs comprennent donc aussi des rapports calculés et des consignes.

## Réserves documentaires

- **[A] Altitude :** le guide exprime l'altitude en pieds. Le readme ne spécifie pas l'échelle numérique de `setting_1` dans l'export. L'interprétation en milliers de pieds parfois associée aux sous-ensembles à plusieurs conditions reste une inférence à vérifier pour notre copie. Nous conservons les valeurs brutes ; les petites valeurs proches de zéro de FD001 ne justifient pas une conversion arbitraire.
- **[B] TRA :** le tableau 1.3 du guide exprime l'angle en degrés. La valeur 100 n'est donc pas assimilée à « 100 % de puissance ».
- **[C] htBleed :** les sources consultées ne donnent pas son unité dans ces tableaux. Nous ne l'étiquetons ni °R ni énergie massique sans confirmation supplémentaire.
- **[D] PCNfR_dmd :** le tableau 2 de l'article indique rpm, tandis que le tableau 1.3 du guide donne pct pour `PCNfRdmd`. Nous signalons ce désaccord et ne convertissons pas cette colonne.
- **[E] W32 :** l'article indique un prélèvement de refroidissement de turbine basse pression (LPT) ; le guide indique haute pression (HPT). Le nom générique du tableau conserve cette incertitude physique.

Ces réserves n'empêchent pas une prédiction statistique dans les échelles du fichier. Elles limitent en revanche les interprétations physiques et les conversions automatiques. Une variable constante sur FD001 peut varier sur FD002–FD004 : la sélection devra être réapprise pour chaque protocole.

## Variables dérivées et séparation apprentissage/validation

Pour un moteur d'apprentissage dont la dernière observation est au cycle `T`, la cible vaut `rul = T - cycle`. Elle est nulle à la fin de vie et reste non plafonnée à cette étape. La connaissance du futur sert uniquement à fabriquer cette cible rétrospective ; `T` et `rul` n'entrent pas dans les caractéristiques du modèle.

| Variable / objet | Définition | Disponible au futur modèle ? |
| --- | --- | --- |
| `rul` | Nombre de cycles restants jusqu'à la fin de vie | Cible uniquement |
| `failure_cycle` | Dernier cycle de la trajectoire complète | Non ; référence pour calculer les cibles |
| `cutoff_cycle` | Cycle d'arrêt tiré une fois pour une trajectoire de validation | Métadonnée de préparation |
| `feature_names_` | Ordre des capteurs conservés, appris sur l'apprentissage | Oui, schéma de l'entrée |
| `X` | Fenêtres de mesures transformées : exemples × cycles × capteurs | Oui |
| `y` | RUL au dernier cycle réel de chaque fenêtre | Cible uniquement |
| `unit_ids` | Moteur auquel appartient chaque fenêtre | Traçabilité ; exclu de `X` |
| `end_cycles` | Dernier cycle réel de chaque fenêtre | Traçabilité ; exclu de `X` |
| `observed_lengths` | Nombre de cycles réels dans chaque fenêtre | Permet d'identifier le remplissage des historiques courts |

Dans le notebook, `train` désigne les moteurs du **groupe d'apprentissage interne**, tandis que `validation_full` désigne les autres moteurs issus du fichier officiel `train_FD001.txt`. La validation est tronquée avant sa fin de vie avec une graine fixée. Ses cibles sont donc strictement positives à la coupure. Le fichier officiel de test reste indépendant de cette validation.

## Documents de référence

- [Readme de l'archive NASA, copie locale](../CMAPSSData/readme.txt) : scénario et ordre des colonnes.
- [Saxena et al., 2008, copie locale](../CMAPSSData/Damage%20Propagation%20Modeling.pdf) : tableau 2 des capteurs et description de la simulation ; [notice primaire NASA NTRS](https://ntrs.nasa.gov/citations/20090029214).
- [Frederick, DeCastro et Litt, guide NASA/TM—2007-215026](https://ntrs.nasa.gov/api/citations/20070034949/downloads/20070034949.pdf?attachment=true) : tableaux 1.2, 1.3 et 1.4, pages imprimées 4–5.
