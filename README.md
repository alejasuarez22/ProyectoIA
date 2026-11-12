# Predicción de la demanda de bicicletas compartidas

Proyecto del segundo corte de IA006 (Inteligencia Artificial). Regresión supervisada con el dataset Bike Sharing Demand de Kaggle (Capital Bikeshare, Washington D.C.): estimar cuántas bicicletas se alquilarán en una hora determinada.

Autores: María Alejandra Suárez Muñoz y Johan Sebastián Valderrama.

Dataset: https://www.kaggle.com/c/bike-sharing-demand (archivo `train.csv`, incluido en `data/`).

Repositorio: https://github.com/alejasuarez22/ProyectoIA

## Archivos

| Archivo | Descripción |
|---|---|
| `IA006_Proyecto_Bike_Sharing.ipynb` | Notebook de Colab: limpieza, EDA, ingeniería de variables, PCA, modelos, evaluación y conclusiones |
| `app.py` | Aplicación de Streamlit |
| `model.py` | Funciones que usa la app para preparar los datos y predecir |
| `models/modelo.joblib` | Pipeline del mejor modelo (Gradient Boosting) con su preprocesamiento |
| `models/resultados.csv` | Métricas de los 6 modelos evaluados |
| `models/importancias.csv` | Variables más importantes del modelo |
| `data/train.csv` | Dataset original |
| `data/muestra_test.csv` | Periodo de prueba (agosto a diciembre de 2012) para la demostración |
| `requirements.txt` | Librerías |

## Notebook

Se puede abrir directamente en Google Colab, ya tiene las salidas. Lee el `train.csv` desde este repositorio, así que no necesita iniciar sesión en Kaggle. Si se vuelve a ejecutar completo tarda unos 7 minutos por la búsqueda de hiperparámetros, y al final genera de nuevo `models/` y `data/muestra_test.csv`.

## Ejecutar la app

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows se activa con `.venv\Scripts\activate`. Se fija `scikit-learn==1.6.1` porque es la versión con la que se guardó el modelo.

## Cómo se conecta el modelo con la app

El notebook guarda en `models/modelo.joblib` el `Pipeline` completo: imputación con la mediana, `StandardScaler`, one-hot de `season` y `weather` y el modelo. `model.py` solo calcula las variables de la fecha (hora, día de la semana, mes, año, seno y coseno de la hora) y aplica los mismos ajustes de limpieza del notebook; el pipeline hace el resto igual que en el entrenamiento. `app.py` usa esas funciones para predecir una hora con las condiciones elegidas o un periodo completo desde un CSV.
