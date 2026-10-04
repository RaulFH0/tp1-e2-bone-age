| Descritor | Modelo | MAE (meses) | RMSE (meses) | R² |
|---|---|---|---|---|
| HOG | Média do treino | 33.69 | 40.85 | -0.000 |
| HOG | SVR | 27.21 | 33.96 | 0.309 |
| HOG | RandomForest | 26.36 | 33.01 | 0.347 |
| HOG | GradientBoosting | 22.14 | 28.29 | 0.520 |
| Textura (LBP+GLCM) | Média do treino | 33.69 | 40.85 | -0.000 |
| Textura (LBP+GLCM) | SVR | 25.17 | 32.34 | 0.373 |
| Textura (LBP+GLCM) | RandomForest | 24.50 | 31.39 | 0.409 |
| Textura (LBP+GLCM) | GradientBoosting | 24.16 | 31.34 | 0.411 |
| Intensidade | Média do treino | 33.69 | 40.85 | -0.000 |
| Intensidade | SVR | 26.96 | 35.15 | 0.259 |
| Intensidade | RandomForest | 24.73 | 31.92 | 0.389 |
| Intensidade | GradientBoosting | 24.91 | 32.02 | 0.385 |

**Melhor combinação no TESTE**: HOG + GradientBoosting, MAE = 22.14 meses.


> Avaliação única no conjunto de teste (600 imagens), com modelos cujos hiperparâmetros foram fixados exclusivamente a partir do treino/validação/CV. Nenhum ajuste foi feito após observar este resultado.
