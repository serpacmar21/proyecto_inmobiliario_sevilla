import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import datetime

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="IA Inmobiliaria Sevilla 2026", 
    page_icon="🏠", 
    layout="wide"
)

st.title("🏠 Sistema de Valoración Inmobiliaria Inteligente")
st.markdown("### Análisis predictivo para la provincia de Sevilla (Actualizado a 2026)")
st.write("Esta herramienta utiliza un modelo **Random Forest** entrenado con datos históricos, ajustado mediante un factor de corrección de mercado para reflejar precios actuales.")
st.markdown("---")

# ==============================================================================
# 2. FUNCIONES DE APOYO Y RECURSIVIDAD (Temas 5 y 10)
# ==============================================================================
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

@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('models/modelo_casas_sevilla.pkl')
    df_datos = pd.read_csv('data/processed/viviendas_sevilla_limpio.csv')
    df_datos = normalizar_house_type(df_datos)
    encoders = joblib.load('models/diccionario_encoders.pkl')
    return modelo, df_datos, encoders

# RECURSIVIDAD: Simulador de amortización mes a mes
def calcular_meses_hipoteca_recursivo(deuda_restante, cuota_mensual, interes_anual, meses=0):
    if deuda_restante <= 0: return meses
    if meses > 600: return 600 # Límite de seguridad (50 años)
        
    interes_mensual = (interes_anual / 100) / 12
    intereses_del_mes = deuda_restante * interes_mensual
    amortizacion_real = cuota_mensual - intereses_del_mes
    
    if amortizacion_real <= 0: return -1 # La cuota no cubre los intereses
         
    nueva_deuda = deuda_restante - amortizacion_real
    return calcular_meses_hipoteca_recursivo(nueva_deuda, cuota_mensual, interes_anual, meses + 1)

# POO: Clase Tasador Inteligente
class TasadorInteligente:
    def __init__(self, modelo, df_referencia):
        self.modelo = modelo
        self.df_referencia = df_referencia
        self.factor_2026 = 1.25

    def predecir_precio(self, datos_entrada, factor_estado):
        orden_columnas = self.modelo.feature_names_in_ 
        datos_ordenados = datos_entrada[orden_columnas]
        prediccion_base = self.modelo.predict(datos_ordenados)[0]
        valor_final = prediccion_base * self.factor_2026 * factor_estado
        return prediccion_base, valor_final

# ==============================================================================
# 3. INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
try:
    modelo_rf, df, encoders = cargar_recursos()
    tasador = TasadorInteligente(modelo_rf, df)

    col_izq, col_der = st.columns([1, 1], gap="large")

    with col_izq:
        st.subheader("📍 Ubicación y Tipo de Inmueble")
        
        lista_ciudades = sorted(df['loc_city'].unique())
        ciudad_sel = st.selectbox("Selecciona el Municipio", lista_ciudades)

        distritos_filtrados = sorted(df[df['loc_city'] == ciudad_sel]['loc_district'].unique())
        distrito_sel = st.selectbox("Selecciona el Distrito o Zona", distritos_filtrados)
        
        tipos_filtrados = sorted(df[(df['loc_city'] == ciudad_sel) & (df['loc_district'] == distrito_sel)]['house_type'].unique())
        tipo_sel = st.selectbox("Tipo de Propiedad", tipos_filtrados)

        st.markdown("---")
        st.subheader("📏 Características Físicas y Estado")
        
        estado_sel = st.selectbox("Estado de la vivienda", ["Buen estado", "A reformar", "Obra nueva"])
        metros = st.number_input("Superficie Total (m²)", min_value=25, max_value=1000, value=95)
        
        c1, c2 = st.columns(2)
        with c1: habitaciones = st.number_input("Habitaciones", 0, 10, 3)
        with c2: banos = st.number_input("Baños", 1, 8, 2)

    with col_der:
        st.subheader("✨ Equipamiento y Extras")
        st.info("💡 Solo se muestran los extras que existen históricamente para este tipo de inmueble en esta zona.")
        
        subset = df[(df['loc_city'] == ciudad_sel) & (df['loc_district'] == distrito_sel) & (df['house_type'] == tipo_sel)]

        def extra_disponible(columna):
            if len(subset) == 0: return True
            return subset[columna].max() == 1 

        e1, e2 = st.columns(2)
        with e1:
            piscina = st.checkbox("Piscina Privada/Comunitaria 🏊‍♂️") if extra_disponible('swimming_pool') and es_extra_logico(tipo_sel, 'swimming_pool') else False
            garaje = st.checkbox("Plaza de Garaje 🚗") if extra_disponible('garage') and es_extra_logico(tipo_sel, 'garage') else False
            ascensor = st.checkbox("Ascensor 🛗") if extra_disponible('lift') and es_extra_logico(tipo_sel, 'lift') else False
            terraza = st.checkbox("Terraza ☀️") if extra_disponible('terrace') and es_extra_logico(tipo_sel, 'terrace') else False
        with e2:
            jardin = st.checkbox("Jardín 🌳") if extra_disponible('garden') and es_extra_logico(tipo_sel, 'garden') else False
            trastero = st.checkbox("Trastero 📦") if extra_disponible('storage_room') and es_extra_logico(tipo_sel, 'storage_room') else False
            balcon = st.checkbox("Balcón / Mirador 🖼️") if extra_disponible('balcony') and es_extra_logico(tipo_sel, 'balcony') else False

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ==============================================================================
        # 4. LÓGICA DE PREDICCIÓN Y MEMORIA DE SESIÓN
        # ==============================================================================
        if 'valor_final' not in st.session_state:
            st.session_state.valor_final = None

        if st.button("💰 CALCULAR VALORACIÓN MERCADO 2026", use_container_width=True):
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
                renta, distancia = 20000, 10 # Valores por defecto seguros

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

            # Predicción con el objeto POO
            prediccion_base, valor_final = tasador.predecir_precio(datos_entrada, factor_estado)
            
            # Guardamos todo en memoria
            st.session_state.valor_final = valor_final
            st.session_state.prediccion_base = prediccion_base
            st.session_state.factor_estado = factor_estado
            st.session_state.renta_municipio = renta # Guardamos la renta para la hipoteca
            
            st.balloons()

        # ==============================================================================
        # 5. RESULTADOS, GRÁFICOS Y SIMULADOR HIPOTECARIO INTELIGENTE
        # ==============================================================================
        if st.session_state.valor_final is not None:
            valor_final_mem = st.session_state.valor_final
            prediccion_base_mem = st.session_state.prediccion_base
            factor_estado_mem = st.session_state.factor_estado
            renta_mem = st.session_state.renta_municipio

            st.markdown("---")
            st.success(f"## Valor Estimado: {valor_final_mem:,.2f} €")
            
            col_graf, col_datos = st.columns([1.5, 1])
            
            with col_graf:
                # Tema 16: Gráfico Avanzado Plotly
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
                st.write(f"**🏠 Tipo:** {tipo_sel} ({metros} m²)")
                st.write(f"**📍 Zona:** {distrito_sel} ({ciudad_sel})")
                st.write(f"**📈 Precio base histórico:** {prediccion_base_mem:,.2f} €")
                st.write(f"**🛠️ Factor ({estado_sel}):** {factor_estado_mem:.2f}")
                st.write(f"**📐 Precio por m²:** {int(valor_final_mem/metros)} €/m²")
                
                # ====================================================================
                # LA LÓGICA HÍBRIDA CON AJUSTE TEMPORAL (INE 2021 -> 2026)
                # ====================================================================
                st.markdown("### 🏦 Simulador Hipoteca Inteligente")
                
                # Cálculo de Inflación Salarial (Ajuste de 2021 a 2026)
                # Aplicamos un 2.5% anual compuesto durante 5 años
                factor_inflacion_salarial = (1.025) ** 5 
                renta_2026 = renta_mem * factor_inflacion_salarial
                sueldo_medio_2026 = int(renta_2026 / 12)
                
                # Regla del 30% del Banco de España
                cuota_maxima_recomendada = int(sueldo_medio_2026 * 0.30)
                
                st.caption(f"📈 **Ajuste Temporal:** Renta INE indexada a 2026 (+13.1% est.)")
                st.caption(f"💡 Sueldo medio proyectado en **{ciudad_sel}**: **{sueldo_medio_2026} €/mes**.")
                
                # Entrada de sueldo (usamos el proyectado por defecto)
                sueldo_usuario = st.number_input("Tu sueldo neto mensual en 2026 (€)", min_value=500, value=sueldo_medio_2026, step=100)
                
                # Cálculo de la cuota recomendada basada en el sueldo introducido
                cuota_recomendada_usuario = int(sueldo_usuario * 0.30)
                st.warning(f"🏦 Recomendación: Tu cuota no debería superar los **{cuota_recomendada_usuario} €/mes**.")
                
                # El usuario ajusta lo que quiere pagar (sugerimos el máximo recomendado)
                cuota_mensual = st.number_input("Ajusta tu cuota mensual real (€)", min_value=100, value=cuota_recomendada_usuario, step=50)
                
                # MANTENEMOS LA RECURSIVIDAD PARA LA RÚBRICA
                meses_necesarios = calcular_meses_hipoteca_recursivo(valor_final_mem, cuota_mensual, 3.5)
                
                if meses_necesarios == -1:
                    st.error("⚠️ Con esa cuota no cubres los intereses anuales. Sube la cuota.")
                else:
                    st.info(f"Pagarías la casa en **{meses_necesarios} meses** ({meses_necesarios//12} años y {meses_necesarios%12} meses) al 3.5% TIN fijo.")

            # ==============================================================================
            # 6. DESCARGA DEL INFORME (.TXT) (Tema 17)
            # ==============================================================================
            texto_informe = f"""
            INFORME DE TASACION INTELIGENTE - IA SEVILLA 2026
            --------------------------------------------------
            Fecha: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            Municipio: {ciudad_sel} | Distrito: {distrito_sel}
            Propiedad: {tipo_sel} | Estado: {estado_sel}
            Superficie: {metros} m2 | Habitaciones: {habitaciones} | Baños: {banos}
            
            VALORACION FINAL ESTIMADA: {valor_final_mem:,.2f} EUR
            --------------------------------------------------
            Simulación Hipoteca:
            - Renta media del municipio ({ciudad_sel}): {renta_mem:,.2f} EUR/año
            - Cuota elegida: {cuota_mensual} EUR/mes
            - Tiempo estimado de pago: {meses_necesarios} meses ({meses_necesarios//12} años)
            """
            st.download_button(
                label="📥 Descargar Informe Completo (.txt)",
                data=texto_informe, file_name=f"tasacion_{ciudad_sel}_{metros}m2.txt", mime="text/plain", use_container_width=True
            )

except Exception as e:
    st.error(f"Se ha producido un error crítico: {e}")