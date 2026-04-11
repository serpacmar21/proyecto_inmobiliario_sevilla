from dagster import job, op, Out, In
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

@op(out=Out(pd.DataFrame))
def cargar_datos():
    """Carga y filtra datos de Sevilla usando Dask."""
    df_espana_dask = dd.read_csv('data/raw/spanish_houses.csv', dtype=str)
    df_sevilla_dask = df_espana_dask[df_espana_dask['loc_zone'].str.contains('sevilla', case=False, na=False)]
    columnas_modelo = [
        'price', 'm2_real', 'room_num', 'bath_num', 'loc_city', 'loc_district',
        'house_type', 'balcony', 'garage', 'swimming_pool', 'terrace', 'storage_room', 
        'lift', 'garden', 'condition' 
    ]
    df_modelo_dask = df_sevilla_dask[columnas_modelo]
    df_modelo = df_modelo_dask.compute(scheduler='threads')
    df_modelo = df_modelo.reset_index(drop=True)
    return df_modelo

@op(ins={"df": In(pd.DataFrame)}, out=Out(pd.DataFrame))
def limpiar_datos(df):
    """Limpieza de datos."""
    df['room_num'] = df['room_num'].astype(str).replace('sin habitación', '0')
    df['bath_num'] = df['bath_num'].astype(str).replace('sin baños', '0')
    columnas_numericas = ['price', 'm2_real', 'room_num', 'bath_num']
    for col in columnas_numericas:
        texto_limpio = df[col].astype(str).str.replace(',', '.')
        solo_numeros = texto_limpio.str.extract(r'(\d+\.?\d*)')[0]
        df[col] = pd.to_numeric(solo_numeros, errors='coerce')
    df = df.dropna(subset=['price', 'm2_real'])
    df['price_m2'] = df['price'] / df['m2_real']
    df['house_type'] = df['house_type'].str.strip()
    df['house_type'] = df['house_type'].replace({
        'Casa o chalet independiente': 'Casa o chalet',
        'Chalet pareado': 'Chalet',
        'Chalet adosado': 'Chalet',
        'Casa de pueblo': 'Casa',
        'Casa rural': 'Casa',
        'Casa terrera': 'Casa',
        'Torre': 'Piso',
    })
    df = df[(df['price'] >= 30000) & (df['price'] <= 2000000)]
    df = df[(df['m2_real'] >= 25) & (df['m2_real'] <= 800)]
    df = df[(df['price_m2'] >= 350) & (df['price_m2'] <= 5000)]
    if 'condition' in df.columns:
        df['is_needs_renovating'] = df['condition'].str.contains('reformar', case=False, na=False).astype(int)
        df['is_new_development'] = df['condition'].str.contains('obra nueva', case=False, na=False).astype(int)
        df = df.drop(columns=['condition'])
    columnas_extras = ['balcony', 'swimming_pool', 'terrace', 'storage_room', 'garden', 'lift']
    for col in columnas_extras:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['garage'] = df['garage'].notna().astype(int)
    df['loc_district'] = df['loc_district'].fillna('Desconocido')
    df['house_type'] = df['house_type'].fillna('Desconocido')
    return df

@op(ins={"df": In(pd.DataFrame)}, out=Out(dict))
def preparar_modelo(df):
    """Prepara encoders y entrena modelo."""
    if 'price_m2' in df.columns:
        df = df.drop(columns=['price_m2'])
    diccionario_encoders = {}
    columnas_texto = ['loc_city', 'loc_district', 'house_type']
    for col in columnas_texto:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        diccionario_encoders[col] = le
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return {'modelo': modelo, 'encoders': diccionario_encoders, 'X_test': X_test, 'y_test': y_test}

@op(ins={"model_data": In(dict)})
def guardar_modelo(model_data):
    """Guarda modelo y encoders."""
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model_data['modelo'], 'models/modelo_casas_sevilla.pkl')
    joblib.dump(model_data['encoders'], 'models/diccionario_encoders.pkl')
    # Guardar datos limpios
    # Asumiendo que df está disponible, pero para simplicidad, no lo guardamos aquí.

@job
def tasador_pipeline():
    df = cargar_datos()
    df_limpio = limpiar_datos(df)
    model_data = preparar_modelo(df_limpio)
    guardar_modelo(model_data)