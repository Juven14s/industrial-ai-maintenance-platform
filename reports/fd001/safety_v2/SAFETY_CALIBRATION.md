# FD001 — calibration croisée de la marge de sécurité

La marge appliquée à chaque pli est estimée uniquement avec les erreurs hors pli des quatre autres
plis. `test_FD001.txt` et `RUL_FD001.txt` ne sont jamais lus. Une marge positive est soustraite à
la prédiction, avec un plancher à zéro cycle.

| Candidat | MAE | RMSE | MAE asymétrique | Surestimation | > 10 cycles | Pire pli | Pire horizon |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hgb_quantile_0075_calibrated_15pct | 15.582 | 24.153 | 18.732 | 15.1 % | 2.5 % | 28.7 % | 31.1 % |
| hgb_quantile_010_calibrated_15pct | 15.092 | 22.970 | 18.961 | 15.1 % | 3.2 % | 28.1 % | 31.1 % |
| hgb_quantile_0075_calibrated_10pct | 16.646 | 25.013 | 19.088 | 10.8 % | 1.9 % | 22.4 % | 24.8 % |
| hgb_quantile_005_calibrated_15pct | 17.115 | 26.855 | 19.386 | 14.9 % | 1.3 % | 24.3 % | 32.8 % |
| hgb_quantile_010_calibrated_10pct | 16.762 | 24.373 | 19.460 | 10.0 % | 2.1 % | 20.1 % | 24.9 % |
| hgb_quantile_005_calibrated_10pct | 18.161 | 27.700 | 19.813 | 10.3 % | 0.9 % | 20.3 % | 24.0 % |
| hgb_quantile_005_calibrated_05pct | 20.029 | 29.220 | 20.941 | 5.1 % | 0.5 % | 10.4 % | 15.9 % |
| hgb_quantile_0075_calibrated_05pct | 19.735 | 27.623 | 21.034 | 5.5 % | 0.7 % | 14.0 % | 15.0 % |
| hgb_quantile_010_calibrated_05pct | 19.693 | 26.972 | 21.379 | 5.7 % | 1.5 % | 15.3 % | 14.0 % |

Le candidat retenu est **hgb_quantile_005_calibrated_05pct** et les contraintes sont **respectées**. Sa marge
finale de **3.969 cycles** est calculée sur toutes les erreurs hors
pli du modèle de base. Le compromis est volontaire : une estimation plus prudente augmente la MAE
et la RMSE classiques, mais réduit le risque de repousser une maintenance à cause d'une RUL trop
optimiste.
