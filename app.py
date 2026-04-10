import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="IA Inmobiliaria Sevilla 2026", 
    page_icon="🏠", 
    layout="wide"
)

# Estilo y Título
st.title("🏠 Sistema de Valoración Inmobiliaria Inteligente")
st.markdown("### Análisis predictivo para la provincia de Sevilla (Actualizado a 2026)")
st.write("Esta herramienta utiliza un modelo **Random Forest** entrenado con datos históricos, ajustado mediante un factor de corrección de mercado para reflejar precios actuales.")
st.markdown("---")

# 2. CARGA DE RECURSOS (Modelo y Datos)
@st.cache_resource
def cargar_recursos():
    # Cargamos el modelo entrenado (.pkl)
    modelo = joblib.load('models/modelo_casas_sevilla.pkl')
    # Cargamos el CSV limpio para los filtros interactivos
    df_datos = pd.read_csv('data/processed/viviendas_sevilla_limpio.csv')
    return modelo, df_datos

try:
    modelo, df = cargar_recursos()

    # 3. INTERFAZ DE USUARIO (Dos columnas principales)
    col_izq, col_der = st.columns([1, 1], gap="large")

    with col_izq:
        st.subheader("📍 Ubicación y Tipo de Inmueble")
        
        # FILTROS EN CASCADA
        lista_ciudades = sorted(df['loc_city'].unique())
        ciudad_sel = st.selectbox("Selecciona el Municipio", lista_ciudades)

        distritos_filtrados = sorted(df[df['loc_city'] == ciudad_sel]['loc_district'].unique())
        distrito_sel = st.selectbox("Selecciona el Distrito o Zona", distritos_filtrados)
        
        tipos_filtrados = sorted(df[(df['loc_city'] == ciudad_sel) & (df['loc_district'] == distrito_sel)]['house_type'].unique())
        tipo_sel = st.selectbox("Tipo de Propiedad", tipos_filtrados)

        st.markdown("---")
        st.subheader("📏 Características Físicas y Estado")
        
        # Estado de la vivienda (NUEVA VARIABLE)
        estado_sel = st.selectbox(
            "Estado de la vivienda", 
            ["Buen estado", "A reformar", "Obra nueva"],
            help="El estado afecta significativamente a la valoración final del inmueble."
        )

        metros = st.slider("Superficie Total (m²)", 25, 1000, 95)
        
        c1, c2 = st.columns(2)
        with c1:
            habitaciones = st.number_input("Habitaciones", 0, 10, 3)
        with c2:
            banos = st.number_input("Baños", 1, 8, 2)

    with col_der:
        st.subheader("✨ Equipamiento y Extras")
        st.info("💡 Solo se muestran los extras que existen históricamente para este tipo de inmueble en esta zona.")
        
        # --- LÓGICA DE EXTRAS DINÁMICOS ---
        subset = df[(df['loc_city'] == ciudad_sel) & 
                    (df['loc_district'] == distrito_sel) & 
                    (df['house_type'] == tipo_sel)]

        def extra_disponible(columna):
            if len(subset) == 0: return True 
            return subset[columna].max() == 1 

        # Checkboxes condicionales
        e1, e2 = st.columns(2)
        with e1:
            piscina = st.checkbox("Piscina Privada/Comunitaria 🏊‍♂️") if extra_disponible('swimming_pool') else False
            garaje = st.checkbox("Plaza de Garaje 🚗") if extra_disponible('garage') else False
            ascensor = st.checkbox("Ascensor 🛗") if extra_disponible('lift') else False
            terraza = st.checkbox("Terraza ☀️") if extra_disponible('terrace') else False
        with e2:
            jardin = st.checkbox("Jardín 🌳") if extra_disponible('garden') else False
            trastero = st.checkbox("Trastero 📦") if extra_disponible('storage_room') else False
            balcon = st.checkbox("Balcón / Mirador 🖼️") if extra_disponible('balcony') else False

        st.markdown("<br><br>", unsafe_allow_html=True)

        # 4. BOTÓN Y LÓGICA DE PREDICCIÓN 
        if st.button("💰 CALCULAR VALORACIÓN MERCADO 2026", use_container_width=True):
            
            # Traductor en caliente para la IA
            def codificar_local(columna, valor):
                le = LabelEncoder()
                le.fit(df[columna])
                return le.transform([valor])[0]

            ciudad_num = codificar_local('loc_city', ciudad_sel)
            distrito_num = codificar_local('loc_district', distrito_sel)
            tipo_num = codificar_local('house_type', tipo_sel)

            # Lógica para transformar el selector de estado a unos y ceros
            reforma_num = 1 if estado_sel == "A reformar" else 0
            obra_nueva_num = 1 if estado_sel == "Obra nueva" else 0

            # Preparamos los datos EXACTAMENTE en el orden del entrenamiento
            datos_entrada = pd.DataFrame({
                'm2_real': [metros],
                'room_num': [habitaciones],
                'bath_num': [banos],
                'loc_city': [ciudad_num],
                'loc_district': [distrito_num],
                'house_type': [tipo_num],
                'balcony': [1 if balcon else 0],
                'garage': [1 if garaje else 0],
                'swimming_pool': [1 if piscina else 0],
                'terrace': [1 if terraza else 0],
                'storage_room': [1 if trastero else 0],
                'lift': [1 if ascensor else 0],
                'garden': [1 if jardin else 0],
                'is_needs_renovating': [reforma_num],
                'is_new_development': [obra_nueva_num]
            })

            # Predicción base (según datos históricos)
            prediccion_base = modelo.predict(datos_entrada)[0]

            # Factor de Actualización (+25% estimado)
            factor_2026 = 1.25 
            valor_final = prediccion_base * factor_2026

            # Resultados
            st.success(f"## Valor Estimado: {valor_final:,.2f} €")
            st.balloons()
            
            with st.expander("Ver análisis detallado del precio"):
                st.write(f"**Precio base histórico:** {prediccion_base:,.2f} €")
                st.write(f"**Ajuste por mercado actual (2026):** +25%")
                st.write(f"**Estado del inmueble:** {estado_sel}")
                st.write(f"**Precio estimado por m²:** {int(valor_final/metros)} €/m²")
                st.info("Este cálculo se ha ajustado para reflejar la evolución del mercado y la condición física seleccionada.")

except Exception as e:
    st.error(f"Se ha producido un error al iniciar la aplicación: {e}")
    st.info("Asegúrate de ejecutar primero el Notebook 02 para generar el modelo actualizado.")