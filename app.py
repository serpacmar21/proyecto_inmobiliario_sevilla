import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import os 

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="IA Inmobiliaria Sevilla 2026", 
    page_icon="🏠", 
    layout="wide"
)

# Menú lateral
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Selecciona una herramienta:", 
    ["Tasador Inteligente", "Análisis de Mercado"]
)
st.sidebar.markdown("---")
st.sidebar.write("Proyecto Final - Inteligencia Artificial\n\nGrado en Estadística - Universidad de Sevilla")

# 2. FUNCIONES DE APOYO Y POO
def normalizar_house_type(df):
    df['house_type'] = df['house_type'].astype(str).str.strip()
    df['house_type'] = df['house_type'].replace({
    'Casa o chalet independiente': 'Casa o chalet', 'Chalet pareado': 'Chalet',
    'Chalet adosado': 'Chalet', 'Casa de pueblo': 'Casa', 'Casa rural': 'Casa',
    'Casa terrera': 'Casa', 'Torre': 'Piso',
    })
    return df

def es_extra_logico(tipo, columna):
    if tipo == 'Piso' and columna == 'garden': return False
    return True

# Arquitectura Base de PyTorch
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16, 32)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        return self.dropout(self.relu(self.fc1(self.flatten(self.relu(self.conv1(x))))))

@st.cache_resource
def cargar_recursos():
    # Carga Clásica
    modelo_rf = joblib.load('models/modelo_casas_sevilla.pkl')
    df_datos = pd.read_csv('data/processed/viviendas_sevilla_limpio.csv')
    df_datos = normalizar_house_type(df_datos)
    encoders = joblib.load('models/diccionario_encoders.pkl')
    
    # Carga Deep Learning
    scaler_x = joblib.load('models/scaler_x.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    modelo_pt = torch.load('models/modelo_pytorch.pth', weights_only=False)
    
    return modelo_rf, modelo_pt, scaler_x, scaler_y, df_datos, encoders

@st.cache_data
def calcular_importancia_pytorch(_modelo_pt, _scaler_x, _scaler_y, df_ref, columnas, _encoders):
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    import torch
    import pandas as pd
    
    df_numerico = df_ref.copy()
    columnas_texto = ['loc_city', 'loc_district', 'house_type']
    for col in columnas_texto:
        if col in df_numerico.columns:
            df_numerico[col] = df_numerico[col].map(lambda s: _encoders[col].transform([s])[0] if s in _encoders[col].classes_ else -1)
            
    X_app = df_numerico[columnas]
    y_app = df_numerico['price']
    
    datos_scaled = _scaler_x.transform(X_app)
    X_t_app = torch.tensor(datos_scaled, dtype=torch.float32).unsqueeze(1)
    
    _modelo_pt['base'].eval()
    _modelo_pt['head'].eval()
    
    with torch.no_grad():
        pred_scaled_base = _modelo_pt['head'](_modelo_pt['base'](X_t_app)).numpy()
        pred_base_euros = _scaler_y.inverse_transform(pred_scaled_base).flatten()
    mae_base_app = mean_absolute_error(y_app, pred_base_euros)
    
    importancias = []
    for i in range(datos_scaled.shape[1]):
        X_shuff = datos_scaled.copy()
        np.random.seed(42)
        np.random.shuffle(X_shuff[:, i])
        X_shuff_t = torch.tensor(X_shuff, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            pred_shuff_scaled = _modelo_pt['head'](_modelo_pt['base'](X_shuff_t)).numpy()
            pred_shuff_euros = _scaler_y.inverse_transform(pred_shuff_scaled).flatten()
            
        mae_shuff = mean_absolute_error(y_app, pred_shuff_euros)
        importancias.append(max(0, mae_shuff - mae_base_app))
        
    importancias = np.array(importancias)
    if importancias.sum() > 0:
        importancias = importancias / importancias.sum()
        
    return importancias

class TasadorInteligente:
    def __init__(self, modelo_rf, modelo_pt, scaler_x, scaler_y, df_referencia):
        self.modelo_rf = modelo_rf
        self.modelo_pt = modelo_pt
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.df_referencia = df_referencia
        self.factor_2026 = 1.25

    def predecir_precio(self, datos_entrada, factor_estado, tipo_modelo):
        orden_columnas = self.modelo_rf.feature_names_in_ 
        datos_ordenados = datos_entrada[orden_columnas]
        
        if tipo_modelo == "Random Forest (Clásico)":
            prediccion_base = self.modelo_rf.predict(datos_ordenados)[0]
        else:
            datos_scaled = self.scaler_x.transform(datos_ordenados)
            X_t = torch.tensor(datos_scaled, dtype=torch.float32).unsqueeze(1)
            
            self.modelo_pt['base'].eval()
            self.modelo_pt['head'].eval()
            with torch.no_grad():
                pred_scaled = self.modelo_pt['head'](self.modelo_pt['base'](X_t)).numpy()
            prediccion_base = float(self.scaler_y.inverse_transform(pred_scaled).flatten()[0])

        valor_final = prediccion_base * self.factor_2026 * factor_estado
        return prediccion_base, valor_final

# Cargamos los recursos necesarios (modelos, scalers, datos) y preparamos el tasador inteligente
try:
    modelo_rf, modelo_pt, scaler_x, scaler_y, df, encoders = cargar_recursos()
    tasador = TasadorInteligente(modelo_rf, modelo_pt, scaler_x, scaler_y, df)
except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.stop()


# PÁGINA 1: TASADOR INTELIGENTE

if pagina == "Tasador Inteligente":
    st.title("Sistema de Valoración Inmobiliaria Inteligente")
    st.markdown("### Análisis predictivo para la provincia de Sevilla (Actualizado a 2026)")
    
    st.write("Esta herramienta utiliza Inteligencia Artificial entrenada con datos históricos, ajustada mediante un factor de corrección de mercado para reflejar precios actuales.")
    
    st.markdown("""
    **Metodología Analítica:**
    Para este proyecto hemos aplicado un procesamiento paralelo con *Dask* para limpiar los datos y hemos enriquecido el conjunto original cruzándolo con la renta bruta media (INE) y la distancia a Sevilla capital. Contamos con dos motores predictivos que puedes elegir: un modelo clásico (*Random Forest*) y una Red Neuronal (*PyTorch CNN 1D*).
    """)

    st.markdown("#### Motor de Inteligencia Artificial")
    tipo_modelo = st.radio(
        "Selecciona el modelo predictivo a utilizar:", 
        ["Random Forest (Clásico)", "Deep Learning (PyTorch CNN 1D)"],
        horizontal=True
    )
    st.markdown("---")

    col_izq, col_der = st.columns([1, 1], gap="large")

    with col_izq:
        st.subheader("Ubicación y Tipo de Inmueble")
        lista_ciudades = sorted(df['loc_city'].unique())
        ciudad_sel = st.selectbox("Selecciona el Municipio", lista_ciudades)

        distritos_filtrados = sorted(df[df['loc_city'] == ciudad_sel]['loc_district'].unique())
        distrito_sel = st.selectbox("Selecciona el Distrito o Zona", distritos_filtrados)
        
        tipos_filtrados = sorted(df[(df['loc_city'] == ciudad_sel) & (df['loc_district'] == distrito_sel)]['house_type'].unique())
        tipo_sel = st.selectbox("Tipo de Propiedad", tipos_filtrados)

        st.markdown("---")
        st.subheader("Características Físicas y Estado")
        estado_sel = st.selectbox("Estado de la vivienda", ["Buen estado", "A reformar", "Obra nueva"])
        metros = st.number_input("Superficie Total (m²)", min_value=25, max_value=1000, value=95)
        
        c1, c2 = st.columns(2)
        with c1: habitaciones = st.number_input("Habitaciones", 0, 10, 3)
        with c2: banos = st.number_input("Baños", 1, 8, 2)

    with col_der:
        st.subheader("Equipamiento y Extras")
        st.write("Solo se muestran los extras que existen históricamente para este tipo de inmueble en esta zona.")
        
        subset = df[(df['loc_city'] == ciudad_sel) & (df['loc_district'] == distrito_sel) & (df['house_type'] == tipo_sel)]

        def extra_disponible(columna):
            if len(subset) == 0: return True
            return subset[columna].max() == 1 

        e1, e2 = st.columns(2)
        with e1:
            piscina = st.checkbox("Piscina Privada/Comunitaria") if extra_disponible('swimming_pool') and es_extra_logico(tipo_sel, 'swimming_pool') else False
            garaje = st.checkbox("Plaza de Garaje") if extra_disponible('garage') and es_extra_logico(tipo_sel, 'garage') else False
            ascensor = st.checkbox("Ascensor") if extra_disponible('lift') and es_extra_logico(tipo_sel, 'lift') else False
            terraza = st.checkbox("Terraza") if extra_disponible('terrace') and es_extra_logico(tipo_sel, 'terrace') else False
        with e2:
            jardin = st.checkbox("Jardín") if extra_disponible('garden') and es_extra_logico(tipo_sel, 'garden') else False
            trastero = st.checkbox("Trastero") if extra_disponible('storage_room') and es_extra_logico(tipo_sel, 'storage_room') else False
            balcon = st.checkbox("Balcón / Mirador") if extra_disponible('balcony') and es_extra_logico(tipo_sel, 'balcony') else False

        st.markdown("<br><br>", unsafe_allow_html=True)

        if 'valor_final' not in st.session_state:
            st.session_state.valor_final = None

        if st.button("CALCULAR VALORACIÓN MERCADO 2026", use_container_width=True):
            def codificar_local(columna, valor):
                return encoders[columna].transform([valor])[0]

            ciudad_num = codificar_local('loc_city', ciudad_sel)
            distrito_num = codificar_local('loc_district', distrito_sel)
            tipo_num = codificar_local('house_type', tipo_sel)

            factor_estado = 1.0 if estado_sel == "Buen estado" else (0.85 if estado_sel == "A reformar" else 1.15)

            try:
                datos_ciudad = df[df['loc_city'] == ciudad_sel]
                if not datos_ciudad.empty:
                    renta = datos_ciudad['renta_bruta_media'].iloc[0]
                    distancia = datos_ciudad['distancia_centro_sevilla_km'].iloc[0]
                else:
                    renta, distancia = df['renta_bruta_media'].mean(), df['distancia_centro_sevilla_km'].mean()
            except:
                renta, distancia = 20000, 10 

            datos_entrada = pd.DataFrame({
                'm2_real': [metros], 'room_num': [habitaciones], 'bath_num': [banos],
                'loc_city': [ciudad_num], 'loc_district': [distrito_num], 'house_type': [tipo_num],
                'balcony': [1 if balcon else 0], 'garage': [1 if garaje else 0],
                'swimming_pool': [1 if piscina else 0], 'terrace': [1 if terraza else 0],
                'storage_room': [1 if trastero else 0], 'lift': [1 if ascensor else 0],
                'garden': [1 if jardin else 0], 'is_needs_renovating': [0],  
                'is_new_development': [0], 'distancia_centro_sevilla_km': [distancia], 
                'renta_bruta_media': [renta]                
            })

            prediccion_base, valor_final = tasador.predecir_precio(datos_entrada, factor_estado, tipo_modelo)
            
            st.session_state.valor_final = valor_final
            st.session_state.prediccion_base = prediccion_base
            st.session_state.factor_estado = factor_estado
            st.session_state.modelo_usado = tipo_modelo

        if st.session_state.valor_final is not None:
            valor_final_mem = st.session_state.valor_final
            prediccion_base_mem = st.session_state.prediccion_base
            factor_estado_mem = st.session_state.factor_estado

            st.markdown("---")
            st.success(f"## Valor Estimado: {valor_final_mem:,.2f} €")
            
            col_graf, col_datos = st.columns([1.5, 1])
            
            with col_graf:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = valor_final_mem,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Termómetro de Mercado (€)", 'font': {'size': 24}},
                    delta = {'reference': prediccion_base_mem, 'increasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, valor_final_mem * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"}, 'bgcolor': "white",
                        'borderwidth': 2, 'bordercolor': "gray",
                        'steps': [{'range': [0, prediccion_base_mem], 'color': 'lightgray'},
                                  {'range': [prediccion_base_mem, valor_final_mem], 'color': 'lightblue'}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': valor_final_mem}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col_datos:
                st.write(f"**Modelo IA:** {st.session_state.modelo_usado}")
                st.write(f"**Tipo:** {tipo_sel} ({metros} m²)")
                st.write(f"**Zona:** {distrito_sel} ({ciudad_sel})")
                st.write(f"**Precio base histórico:** {prediccion_base_mem:,.2f} €")
                st.write(f"**Factor ({estado_sel}):** {factor_estado_mem:.2f}")
                st.write(f"**Precio por m²:** {int(valor_final_mem/metros)} €/m²")
            
            texto_informe = f"""
            INFORME DE TASACION INTELIGENTE - IA SEVILLA 2026
            --------------------------------------------------
            Modelo Usado: {st.session_state.modelo_usado}
            Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            Municipio: {ciudad_sel} | Distrito: {distrito_sel}
            Propiedad: {tipo_sel} | Estado: {estado_sel}
            Superficie: {metros} m2 | Habitaciones: {habitaciones} | Baños: {banos}
            
            VALORACION FINAL ESTIMADA: {valor_final_mem:,.2f} EUR
            --------------------------------------------------
            """
            st.download_button(
                label="Descargar Informe Completo (.txt)",
                data=texto_informe, file_name=f"tasacion_{ciudad_sel}_{metros}m2.txt", mime="text/plain", use_container_width=True
            )


# PÁGINA 2: ANÁLISIS DE MERCADO

elif pagina == "Análisis de Mercado":
    st.title("Análisis del Mercado Inmobiliario en Sevilla")
    st.write("A continuación se exponen las métricas y tendencias clave extraídas de nuestro dataset una vez depurado.")
    
    st.markdown("""
    **Nota Importante sobre la temporalidad de los datos:**
    Los precios, valores por metro cuadrado y gráficos mostrados en esta sección de Análisis de Mercado reflejan la fotografía histórica de **2021** (año de recolección de nuestra base de datos original). Es por este motivo que nuestro algoritmo de la pestaña *Tasador Inteligente* aplica internamente un factor de corrección sobre sus predicciones para adaptar los precios base aprendidos por la IA a la realidad del escenario económico actual en **2026**.
    """)
    st.markdown("---")

    # 1. Tabla Dinámica
    st.subheader("Resumen por Municipio y Tipo de Vivienda")
    st.write("La siguiente tabla muestra el precio medio por metro cuadrado y el conteo total de viviendas en oferta según municipio y tipología. Permite identificar de un vistazo las zonas con mayor disponibilidad y las de precio más elevado.")
    
    pivot_precio = pd.pivot_table(
        df,
        values='price_m2',
        index='loc_city',
        columns='house_type',
        aggfunc=['mean', 'count'],
        fill_value=0
    )
    st.dataframe(pivot_precio, use_container_width=True)

    st.markdown("---")
    st.subheader("Visualización de Tendencias en 2021")
    st.write("Como se aprecia en el **Histograma**, la mayoría de las viviendas ofertadas en la provincia se concentraban entre los 100.000€ y 300.000€, mostrando una clara asimetría positiva típica del sector inmobiliario. En el **Gráfico de Dispersión** observamos una evidente relación lineal positiva entre la superficie y el precio, aunque la varianza aumenta en casas de gran tamaño. Por último, el **Ranking de Municipios** confirma que el Aljarafe (Tomares, Mairena) y la Capital aglutinan el suelo más caro.")

    # 2. Generación de los 3 Gráficos usando Matplotlib/Seaborn
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    sns.histplot(df['price'], kde=True, color='#45B39D', ax=axes[0])
    axes[0].set_title('Distribución de Precios', fontweight='bold')
    axes[0].set_xlabel('Precio de Venta')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k €'))

    sns.scatterplot(data=df, x='m2_real', y='price', alpha=0.4, color='#3498DB', ax=axes[1])
    axes[1].set_title('Precio vs Metros Cuadrados', fontweight='bold')
    axes[1].set_xlim(0, 900)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k €'))
    axes[1].set_xlabel('Metros Cuadrados (m²)')
    axes[1].set_ylabel('Precio')

    top_10_ciudades = df.groupby('loc_city')['price_m2'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_10_ciudades.index, y=top_10_ciudades.values, palette='magma', hue=top_10_ciudades.index, legend=False, ax=axes[2])
    axes[2].set_title('Precio Medio por m² (Top 10)', fontweight='bold')
    axes[2].set_ylabel('Precio € / m²')
    axes[2].set_xlabel('Municipio')
    axes[2].set_xticks(range(len(top_10_ciudades)))
    axes[2].set_xticklabels(top_10_ciudades.index, rotation=45, ha='right')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)} €'))

    plt.tight_layout()
    st.pyplot(fig)


    # 3. IMPORTANCIA DE VARIABLES (Explicabilidad de IA)
    st.markdown("---")
    st.subheader("Explicabilidad del Modelo: ¿Qué dicta el precio?")
    st.write("No basta con predecir el precio; es necesario entender por qué la Inteligencia Artificial toma sus decisiones. A continuación, puedes auditar el razonamiento interno de ambos algoritmos.")

    tipo_imp = st.radio(
        "Selecciona el modelo para auditar su razonamiento:", 
        ["Random Forest (Clásico)", "Deep Learning (PyTorch CNN 1D)"], 
        horizontal=True
    )

    columnas = modelo_rf.feature_names_in_
    
    if tipo_imp == "Random Forest (Clásico)":
        importancias = modelo_rf.feature_importances_
        color = '#8E44AD'
        titulo = 'Importancia de Variables (Random Forest)'
    else:
        importancias = calcular_importancia_pytorch(modelo_pt, scaler_x, scaler_y, df, list(columnas), encoders)
        color = '#E74C3C'
        titulo = 'Importancia por Permutación (PyTorch CNN 1D)'

    indices = np.argsort(importancias)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    ax_imp.barh(range(len(indices)), importancias[indices], color=color, align='center')
    ax_imp.set_yticks(range(len(indices)))
    ax_imp.set_yticklabels([columnas[i] for i in indices])
    ax_imp.set_xlabel('Importancia Relativa')
    ax_imp.set_title(titulo, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_imp)
    
    st.markdown("##### Comparativa de Rendimiento (Evaluación en el Test Set)")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Random Forest", "R²: 0.8526", "MAE: 28.218 €", delta_color="off")
    col_m2.metric("PyTorch CNN 1D", "R²: 0.7959", "MAE: 32.799 €", delta_color="off")
    
    st.write("El modelo **Random Forest** ha demostrado ser superior para este conjunto de datos tabulares, logrando un menor margen de error (MAE) y explicando mejor la varianza del mercado (R² de 0.85). Por su parte, la red neuronal (**PyTorch CNN 1D** con *Fine-Tuning*), aunque ligeramente menos precisa (R² de 0.79), ofrece un enfoque de computación moderna muy potente extrayendo patrones ocultos. A nivel de explicabilidad, ambos modelos coinciden en su razonamiento: la **renta bruta media del municipio**, los **metros cuadrados** y la **distancia a Sevilla capital** son los tres pilares indiscutibles a la hora de tasar un inmueble en la provincia.")

    # 4. Gráfico generado por R (Interoperabilidad)
    st.markdown("---")
    st.subheader("Análisis Avanzado de Correlación (Python-R)")
    st.write("Finalmente, aprovechamos la interoperabilidad de nuestro ecosistema para ejecutar un script de R (`ggplot2`) desde Python. El resultado confirma visualmente lo que nuestra Inteligencia Artificial aprendió de forma matemática: existe una fuerte relación entre el nivel de renta de un municipio y el valor de venta de los inmuebles en el mismo.")
    
    ruta_grafico_r = "notebooks/grafico_renta_precio.png"
    if os.path.exists(ruta_grafico_r):
        st.image(ruta_grafico_r, use_container_width=True)
    else:
        st.warning("No se ha encontrado el gráfico generado por R.")