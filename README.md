# Modelo de aprobación de tarjetas de crédito

Proyecto de Machine Learning que predice si la solicitud de una tarjeta de crédito será **rechazada**, a partir de los datos del cliente. Usa una **Regresión Logística** dentro de un **Pipeline** de scikit-learn.

## Contenido

| Archivo | Descripción |
|---|---|
| `AnteloRamirez_Raul_tarjetas_credito.ipynb` | Notebook ejecutado con el código y las explicaciones de cada decisión |
| `data/Credit_card.csv` | Datos personales y socioeconómicos de cada solicitante (`Ind_ID`) |
| `data/Credit_card_label.csv` | Decisión de cada solicitud: `0` = aprobada, `1` = rechazada |

Dataset: *Credit Card Details* (Kaggle).

## Qué se hace en el notebook

1. **Carga y unión** de los dos CSV por `Ind_ID` (*inner join* uno a uno) y eliminación de duplicados.
2. **Análisis de nulos**: se cuantifican, se estudia su relación con el target y se elige una estrategia para cada columna (mediana, moda, categoría "Desconocido" e indicadores de nulo).
3. **Limpieza de variables**: se eliminan las que no aportan (`Ind_ID`, `Mobile_phone`), se convierten `Birthday_count` y `Employed_days` a edad y años de antigüedad, y se ajustan los tipos de datos.
4. **Codificación de categóricas**: `TargetEncoder` para `Type_Income` y `Type_Occupation`, One-Hot para las nominales y binarias, ordinal para `EDUCATION`.
5. **División** 80 % entrenamiento / 20 % test, estratificada, con `random_state=42`.
6. **Pipeline** con imputación, codificación, escalado y `LogisticRegression`, entrenado solo con train (sin fuga de datos).
7. **Desbalance de clases** (~10 % rechazadas): comparación de un `DummyClassifier`, la regresión sin pesos y con `class_weight="balanced"`, usando validación cruzada sobre train.
8. **Métricas** sobre test con la clase positiva = rechazada: accuracy, precision, recall y F1 (más ROC-AUC y PR-AUC).
9. **Matriz de confusión** interpretada para el negocio, **ajuste del umbral de decisión** (elegido por validación cruzada sobre train según un recall objetivo) y coeficientes del modelo.
10. **Valoración del modelo** con los criterios de los apuntes (AUC, Average Precision, recall, especificidad, F1/F2): el modelo es modesto, no bueno.
11. **Conclusiones**: resultados, limitaciones y posibles mejoras.

## Resultado principal

Con `class_weight="balanced"` el modelo detecta unas 15 de las 27 solicitudes rechazadas del test (recall ≈ 0,56), pero con precision baja (≈ 0,16). Sin tener en cuenta el desbalance, el modelo no predice ninguna rechazada y equivale al Dummy. El rendimiento es modesto, como se esperaba de un modelo lineal con estos datos; lo importante es el proceso y la interpretación de las métricas.

## Cómo ejecutarlo

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install pandas numpy scipy matplotlib scikit-learn jupyter
jupyter notebook AnteloRamirez_Raul_tarjetas_credito.ipynb
```

El notebook se ejecuta de principio a fin con `random_state=42`, por lo que los resultados son reproducibles.

## Uso de IA

Se utilizó Claude Code como herramienta de apoyo para revisar, reorganizar y ampliar el notebook (ajuste del umbral, valoración del modelo y análisis de robustez). Las decisiones sobre qué hacer las tomé yo; Claude Code escribió el código y los textos siguiendo mis indicaciones. El detalle figura en la última sección del propio notebook.
