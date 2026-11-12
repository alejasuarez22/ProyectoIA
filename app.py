import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import load_model, predict

WEATHER = {1: '1 - Despejado o pocas nubes', 2: '2 - Nublado o neblina', 3: '3 - Lluvia o nieve ligera'}
SEASON = {1: 'Primavera (ene-mar)', 2: 'Verano (abr-jun)', 3: 'Otoño (jul-sep)', 4: 'Invierno (oct-dic)'}

st.set_page_config(page_title='Demanda de bicicletas', layout='wide')

@st.cache_resource
def get_model():
    return load_model('models/modelo.joblib')

art = get_model()

st.title('Predicción de la demanda de bicicletas compartidas')
st.write('Proyecto IA006 - Bike Sharing Demand (Capital Bikeshare, Washington D.C.) - Regresión supervisada')

st.header('Condiciones de la hora')
col1, col2, col3 = st.columns(3)

with col1:
    day = st.date_input('Fecha', date(2012, 9, 14), min_value=date(2011, 1, 1), max_value=date(2012, 12, 31))
    hour = st.slider('Hora', 0, 23, 17)
    holiday = st.checkbox('Es festivo')

with col2:
    weather = st.selectbox('Clima', list(WEATHER), format_func=lambda k: WEATHER[k])
    temp = st.slider('Temperatura (°C)', 0.0, 41.0, 25.0, 0.5)
    atemp = st.slider('Sensación térmica (°C)', 0.0, 46.0, 28.0, 0.5)

with col3:
    humidity = st.slider('Humedad (%)', 1, 100, 60)
    windspeed = st.slider('Velocidad del viento', 0.0, 57.0, 13.0, 0.5)
    season = (day.month - 1) // 3 + 1
    workingday = int(day.weekday() < 5 and not holiday)
    st.write('Estación:', SEASON[season])
    st.write('Día laboral:', 'Sí' if workingday else 'No')

def build(hours):
    return pd.DataFrame([{
        'datetime': datetime(day.year, day.month, day.day, h),
        'season': season,
        'holiday': int(holiday),
        'workingday': workingday,
        'weather': weather,
        'temp': temp,
        'atemp': atemp,
        'humidity': humidity,
        'windspeed': windspeed
    } for h in hours])

st.header('Resultado')
day_pred = predict(build(range(24)), art)
value = day_pred[hour]

c1, c2 = st.columns([1, 2])
c1.metric('Bicicletas estimadas a las ' + str(hour) + ':00', f'{value:.0f}')
c1.write('Total estimado para el día con estas condiciones: ' + f'{day_pred.sum():,.0f}'.replace(',', '.'))
c1.caption('Error promedio del modelo en test: ' + f"{art['metrics']['MAE']:.0f}" + ' bicicletas por hora')

fig, ax = plt.subplots(figsize=(8,3.5))
ax.bar(range(24), day_pred, color=['tab:orange' if h == hour else 'tab:blue' for h in range(24)])
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('Hora')
ax.set_ylabel('Bicicletas')
ax.set_title('Demanda estimada para todo el día con las mismas condiciones')
c2.pyplot(fig)

st.header('Modelo')
st.write('Modelo en uso:', art['model_name'])
st.write('Hiperparámetros:', art['params'])
m = art['metrics']
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('R²', f"{m['R2']:.3f}")
k2.metric('RMSLE', f"{m['RMSLE test']:.3f}")
k3.metric('RMSE', f"{m['RMSE']:.1f}")
k4.metric('MAE', f"{m['MAE']:.1f}")
k5.metric('MSE', f"{m['MSE']:.0f}")

st.subheader('Comparación de modelos (conjunto de prueba)')
st.dataframe(pd.read_csv('models/resultados.csv'))

importance = pd.read_csv('models/importancias.csv')
fig, ax = plt.subplots(figsize=(7,4))
ax.barh(importance['variable'][::-1], importance['importancia'][::-1])
ax.set_title('Variables más importantes')
st.pyplot(fig)

st.header('Predecir un periodo completo')
source = st.radio('Datos', ['Periodo de prueba (ago-dic 2012)', 'Subir un CSV'])

if source == 'Subir un CSV':
    file = st.file_uploader('CSV con las columnas del train.csv de Bike Sharing Demand', type=['csv'])
    if file is None:
        st.stop()
    df = pd.read_csv(file)
else:
    df = pd.read_csv('data/muestra_test.csv')

needed = ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
missing = [col for col in needed if col not in df.columns]
if missing:
    st.error('Faltan columnas: ' + ', '.join(missing))
    st.stop()

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['prediccion'] = predict(df, art).round(0)

if 'count' in df.columns:
    mse = mean_squared_error(df['count'], df['prediccion'])
    rmsle = np.sqrt(mean_squared_error(np.log1p(df['count']), np.log1p(df['prediccion'])))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric('R²', f"{r2_score(df['count'], df['prediccion']):.3f}")
    p2.metric('RMSLE', f'{rmsle:.3f}')
    p3.metric('RMSE', f'{np.sqrt(mse):.1f}')
    p4.metric('MAE', f"{mean_absolute_error(df['count'], df['prediccion']):.1f}")

days = st.slider('Días a graficar', 1, 14, 7)
view = df[df['datetime'] < df['datetime'].min() + pd.Timedelta(days=days)]
fig, ax = plt.subplots(figsize=(12,4))
if 'count' in view.columns:
    ax.plot(view['datetime'], view['count'], label='Real')
ax.plot(view['datetime'], view['prediccion'], label='Predicción')
ax.set_ylabel('Bicicletas por hora')
ax.legend()
st.pyplot(fig)

st.dataframe(df[[col for col in ['datetime', 'weather', 'temp', 'humidity', 'workingday', 'count', 'prediccion'] if col in df.columns]])
