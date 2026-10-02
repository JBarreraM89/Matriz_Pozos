import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.special import exp1
import folium
from streamlit_folium import st_folium

# --- GESTOR DE ESTADO PARA EL MAPA ---
if 'puntos_mapa' not in st.session_state:
    st.session_state.puntos_mapa = pd.DataFrame(columns=['Identificador', 'Latitud', 'Longitud', 'Tipo'])

# --- FUNCIÓN: Algoritmo de Detección de Flujo Radial ---
def detectar_flujo_radial(df, col_t, col_s):
    d = df[df[col_t] > 0].dropna(subset=[col_t, col_s]).reset_index(drop=True)
    if len(d) < 4: return float(d[col_t].min()), float(d[col_t].max())
    log_t = np.log10(d[col_t])
    s = d[col_s]
    derivada = np.gradient(s, log_t)
    mejor_ventana = None
    menor_var = float('inf')
    window_size = min(4, len(d))
    for i in range(len(d) - window_size + 1):
        ventana = derivada[i:i+window_size]
        promedio_pendiente = np.mean(ventana)
        if promedio_pendiente > 0.5:  
            varianza = np.var(ventana)
            if varianza < menor_var:
                menor_var = varianza
                mejor_ventana = (float(d[col_t].iloc[i]), float(d[col_t].iloc[i+window_size-1]))
    if mejor_ventana: return mejor_ventana
    else: return float(d[col_t].min()), float(d[col_t].iloc[len(d)//3])

st.set_page_config(page_title="Análisis de Aforo de Pozos", layout="wide")
st.title("💧 Sistema de Análisis de Pruebas de Aforo")

# ==========================================
# PANEL IZQUIERDO: GASTO ESTIMADO POR POBLACIÓN (MAPAS)
# ==========================================
st.sidebar.header("👥 Cálculo de Gasto Comunitario")
st.sidebar.markdown("Estimación basada en los parámetros del **Manual de Agua Potable, Alcantarillado y Saneamiento (MAPAS)** de la CONAGUA.")

# Variables Editables
poblacion = st.sidebar.number_input("Número de Habitantes (Población de Proyecto):", min_value=1, value=10000, step=100)

clima = st.sidebar.selectbox("Clima Predominante de la Región:", 
                             ["Cálido Húmedo", "Cálido Subhúmedo", "Seco o Muy Seco", "Templado o Frío"])

nivel_socioeconomico = st.sidebar.selectbox("Nivel Socioeconómico:", 
                                            ["Promedio", "Bajo", "Medio", "Alto"])

# Diccionario extraído de la Tabla 2.2 del MAPAS
consumos_mapas = {
    "Cálido Húmedo": {"Bajo": 198, "Medio": 206, "Alto": 243, "Promedio": 201},
    "Cálido Subhúmedo": {"Bajo": 175, "Medio": 203, "Alto": 217, "Promedio": 191},
    "Seco o Muy Seco": {"Bajo": 184, "Medio": 191, "Alto": 202, "Promedio": 190},
    "Templado o Frío": {"Bajo": 140, "Medio": 142, "Alto": 145, "Promedio": 142}
}

consumo_base = consumos_mapas[clima][nivel_socioeconomico]

st.sidebar.markdown("---")
st.sidebar.markdown("**Ajustes Hidráulicos y Operativos**")
perdidas = st.sidebar.slider("Pérdidas Físicas Estimadas en la Red (%):", min_value=0, max_value=50, value=25, step=1)
horas_bombeo = st.sidebar.slider("Horas de operación de bombeo al día:", min_value=1, max_value=24, value=18, step=1)
cvd = st.sidebar.slider("Coeficiente de Variación Diaria (Cvd):", min_value=1.20, max_value=1.40, value=1.30, step=0.01)
cvh = st.sidebar.number_input("Coeficiente de Variación Horaria (Cvh):", value=1.55, format="%.2f")

# Cálculos Hidráulicos Oficiales
dotacion = consumo_base / (1 - (perdidas / 100))
q_med = (poblacion * dotacion) / 86400
q_md = q_med * cvd
q_mh = q_md * cvh
q_bombeo = (q_md * 24) / horas_bombeo if horas_bombeo > 0 else 0

# Mostrar Resultados en el Panel Lateral
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Demandas de Diseño")
st.sidebar.metric("Consumo Base de Agua", f"{consumo_base} l/hab/d")
st.sidebar.metric("Dotación Requerida", f"{dotacion:.2f} l/hab/d")
st.sidebar.metric("Gasto Medio Diario (Qmed)", f"{q_med:.2f} l/s")
st.sidebar.metric("Gasto Máximo Diario (QMd)", f"{q_md:.2f} l/s")
st.sidebar.metric("Gasto Máximo Horario (QMh)", f"{q_mh:.2f} l/s")
st.sidebar.metric("Gasto de Bombeo (Qb)", f"{q_bombeo:.2f} l/s")

# Explicación Detallada Dinámica
with st.sidebar.expander("📖 ¿Qué significan estos resultados y cómo se obtuvieron?"):
    st.markdown(f"""
    Los resultados mostrados se han adaptado de forma específica a los datos ingresados:
    
    *   **Consumo Base ({consumo_base} l/hab/d):** Representa la cantidad teórica de agua que consume directamente cada habitante. Se obtuvo cruzando la selección de clima '{clima}' y estrato socioeconómico '{nivel_socioeconomico}' con la *Tabla 2.2* del MAPAS de CONAGUA.
    *   **Dotación Requerida ({dotacion:.2f} l/hab/d):** Es la cantidad total de agua que el organismo operador debe inyectar a la red por cada habitante. Surge de sumar al consumo base el **{perdidas}%** de agua que se estimó se perderá por fugas o tomas clandestinas durante su transporte.
    *   **Gasto Medio Diario (Qmed = {q_med:.2f} l/s):** Indica el caudal promedio continuo que la fuente de abastecimiento debe producir de manera ininterrumpida durante el año para satisfacer a los {poblacion} habitantes. Se obtiene al transformar el volumen de la dotación diaria a litros por segundo.
    *   **Coeficiente de Variación Diaria (Cvd = {cvd}):** Es un factor normativo que representa cómo fluctúa la demanda de agua en el día de mayor consumo del año respecto a un día promedio.
    *   **Gasto Máximo Diario (QMd = {q_md:.2f} l/s):** Representa el caudal que demandará la ciudad en el día de mayor consumo del año. Se calculó afectando el Qmed con el factor de variación diaria de **{cvd:.2f}**. Este dato sirve para dimensionar las líneas de conducción principales y las plantas potabilizadoras.
    *   **Coeficiente de Variación Horaria (Cvh = {cvh}):** Representa la fluctuación de la demanda dentro de un mismo día. Indica qué tanto se eleva el consumo durante la "hora pico" (cuando la mayoría de la población usa el agua simultáneamente) en comparación con el promedio de ese día.
    *   **Gasto Máximo Horario (QMh = {q_mh:.2f} l/s):** Es el caudal de mayor impacto en la red, ocurrido en la hora pico del día de mayor consumo. Se calculó multiplicando el Gasto Máximo Diario por el factor normativo de **{cvh:.2f}**. Se utiliza para seleccionar los diámetros de las tuberías de distribución en las calles, asegurando presiones adecuadas.
    *   **Gasto de Bombeo (Qb = {q_bombeo:.2f} l/s):** Determina el caudal real que debe extraer el equipo de bombeo de la captación, compensando el hecho de que no operará continuamente. Se calculó multiplicando el Gasto Máximo Diario por 24 horas y dividiéndolo entre las **{horas_bombeo}** horas de operación seleccionadas. Esta cifra es el punto de diseño electromecánico para la selección de la bomba del pozo.
    """)

# ==========================================
# SECCIÓN 1: CARGA DE DATOS Y MAPEO
# ==========================================
st.header("📊 1. Carga de Datos y Configuración")
uploaded_file = st.file_uploader("Sube tu archivo de aforo aquí (.csv)", type=["csv"], key="uploader_csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columnas_csv = df.columns.tolist()
    
    st.subheader("⚙️ Configuración de Variables")
    col_sel1, col_sel2, col_sel3, col_sel4, col_sel5 = st.columns(5)
    
    with col_sel1:
        col_tiempo = st.selectbox("Tiempo (Horas):", columnas_csv, index=columnas_csv.index('Tiempo_hrs') if 'Tiempo_hrs' in columnas_csv else 0)
    with col_sel2:
        col_caudal = st.selectbox("Caudal (LPS):", columnas_csv, index=columnas_csv.index('LPS') if 'LPS' in columnas_csv else 0)
    with col_sel3:
        col_abat = st.selectbox("Abatimiento (m):", columnas_csv, index=columnas_csv.index('Abatimiento') if 'Abatimiento' in columnas_csv else 0)
    with col_sel4:
        col_hz = st.selectbox("Frecuencia (Hz):", columnas_csv, index=columnas_csv.index('Hz') if 'Hz' in columnas_csv else 0)
    with col_sel5:
        col_estatico = st.selectbox("Nivel Estático (m):", columnas_csv, index=columnas_csv.index('Nivel_Estatico') if 'Nivel_Estatico' in columnas_csv else 0)

    for col in [col_tiempo, col_caudal, col_abat, col_hz, col_estatico]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Capacidad_Especifica'] = np.where(df[col_abat] > 0, df[col_caudal] / df[col_abat], 0)

    with st.expander("Ver tabla de datos procesados"):
        st.dataframe(df)
    st.divider()

    # ==========================================
    # SECCIÓN 2: ANÁLISIS TÉCNICO
    # ==========================================
    st.header("📉 2. Análisis del Comportamiento del Pozo (Técnico)")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Abatimiento Lineal", 
        "Abatimiento (Invertido)", 
        "Cooper-Jacob (Auto-Ajuste)", 
        "Caudal",
        "Capacidad Específica",
        "Curva de Operación"
    ])

    with tab1:
        st.markdown("**Utilidad:** Observar la tendencia general del descenso del nivel dinámico y confirmar el momento de estabilización.")
        fig1 = px.line(df, x=col_tiempo, y=col_abat, markers=True, title="Evolución del Abatimiento vs Tiempo")
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("**Utilidad:** Perspectiva física intuitiva de la rapidez con la que 'cae' el nivel de agua en el pozo.")
        fig2 = px.line(df, x=col_abat, y=col_tiempo, markers=True, title="Tiempo transcurrido vs Abatimiento")
        fig2.update_layout(xaxis_title="Abatimiento (m)", yaxis_title="Tiempo (Horas)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("**Utilidad:** La app analiza la derivada del abatimiento para evadir la zona de estabilización y recomendar el segmento de flujo radial óptimo para calcular la Transmisividad (T).")
        df_log = df[df[col_tiempo] > 0].dropna(subset=[col_tiempo, col_abat, col_caudal])
        
        if not df_log.empty:
            t_rec_inicio, t_rec_fin = detectar_flujo_radial(df_log, col_tiempo, col_abat)
            rango_t = st.slider("Segmento de flujo radial (Recomendación automática):", 
                                min_value=float(df_log[col_tiempo].min()), 
                                max_value=float(df_log[col_tiempo].max()), 
                                value=(t_rec_inicio, t_rec_fin), step=0.1)
            
            mask = (df_log[col_tiempo] >= rango_t[0]) & (df_log[col_tiempo] <= rango_t[1])
            df_fit = df_log[mask]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_log[col_tiempo], y=df_log[col_abat], mode='markers', name='Datos Aforo'))
            
            if len(df_fit) > 1:
                slope, intercept = np.polyfit(np.log10(df_fit[col_tiempo]), df_fit[col_abat], 1)
                delta_s = abs(slope)
                
                if delta_s > 0.01:
                    caudal_ajuste_lps = df_fit[col_caudal].mean()
                    T_calc = (0.183 * (caudal_ajuste_lps * 86.4)) / delta_s
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric("Pendiente (Δs)", f"{delta_s:.2f} m")
                    col_res2.metric("Caudal Tramo (Q)", f"{caudal_ajuste_lps:.2f} LPS")
                    col_res3.metric("Transmisividad (T)", f"{T_calc:.2f} m2/dia")
                    
                    x_line = np.linspace(rango_t[0], rango_t[1], 50)
                    fig3.add_trace(go.Scatter(x=x_line, y=slope*np.log10(x_line)+intercept, mode='lines', name='Ajuste Lineal', line=dict(color='red', dash='dash')))
                else:
                    st.warning("El segmento seleccionado es demasiado plano (estabilizado). Amplía el rango hacia la izquierda.")
                    
            fig3.update_layout(xaxis_type="log", xaxis_title="Tiempo (Horas) [Log]", yaxis_title="Abatimiento (m)")
            st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("**Utilidad:** Verifica la estabilidad del bombeo operativo.")
        fig4 = px.line(df, x=col_tiempo, y=col_caudal, markers=True, title="Comportamiento del Caudal (LPS)")
        fig4.update_traces(line_color='green')
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        st.markdown("**Utilidad:** Muestra la eficiencia hidráulica (Q/s).")
        fig5 = px.line(df, x=col_tiempo, y='Capacidad_Especifica', markers=True, title="Evolución de la Capacidad Específica (LPS/m)")
        fig5.update_traces(line_color='purple')
        st.plotly_chart(fig5, use_container_width=True)

    with tab6:
        st.markdown("**Utilidad:** Desempeño electromecánico para justificar dimensión de la bomba.")
        fig6 = px.scatter(df, x=col_hz, y=col_caudal, color=col_tiempo, title="Desempeño Electromecánico (Hz vs LPS)", color_continuous_scale='viridis')
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # ==========================================
    # SECCIÓN 3: SIMULACIÓN DEL CONO
    # ==========================================
    st.header("🌐 3. Simulación Analítica del Cono de Abatimiento")
    caudal_final_lps = df[col_caudal].dropna().iloc[-1]
    tiempo_final_hrs = df[col_tiempo].max()
    
    st.info(f"📌 **Condiciones extraídas:** Caudal = **{caudal_final_lps} LPS** | Tiempo = **{tiempo_final_hrs} Horas**")
    
    col_t, col_s = st.columns(2)
    with col_t:
        T_m2d = st.slider("Transmisividad Teórica (m2/dia) [T]", min_value=10.0, max_value=5000.0, value=150.0, step=10.0)
    with col_s:
        S_coef = st.number_input("Almacenamiento Teórico [S]", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")

    radios = np.logspace(-1, 2.7, 100)
    u = (radios**2 * S_coef) / (4 * T_m2d * (tiempo_final_hrs / 24.0))
    abatimientos = ((caudal_final_lps * 86.4) / (4 * np.pi * T_m2d)) * exp1(u)

    tab_3d, tab_2d = st.tabs(["Superficie 3D", "Perfil 2D"])

    with tab_3d:
        theta = np.linspace(0, 2 * np.pi, 50)
        R, Theta = np.meshgrid(radios, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Z = -np.tile(abatimientos, (50, 1))
        
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
        fig_3d.update_layout(scene=dict(zaxis=dict(range=[np.min(Z)*1.1, 0])), margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_2d:
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(x=radios, y=abatimientos, mode='lines', fill='tozeroy'))
        fig_2d.update_layout(xaxis_type="log", yaxis=dict(autorange="reversed"), xaxis_title="Distancia (m)", yaxis_title="Abatimiento (m)", height=400)
        st.plotly_chart(fig_2d, use_container_width=True)
        
    st.divider()

    # ==========================================
    # SECCIÓN 4: RESUMEN EJECUTIVO (TOMA DE DECISIONES)
    # ==========================================
    st.header("💡 4. Resultados para Toma de Decisiones (Resumen Ejecutivo)")
    st.markdown("""
    Esta sección traduce los datos técnicos de la prueba en respuestas claras para proteger la inversión electromecánica y comprender el potencial real del pozo. 
    **Modifique los parámetros físicos de la instalación para simular escenarios operativos.**
    """)

    st.subheader("🛠️ Variables de la Instalación (Editables)")
    col_var1, col_var2, col_var3, col_var4 = st.columns(4)
    
    with col_var1:
        nivel_estatico_val = df[col_estatico].dropna().iloc[0] if col_estatico in df.columns else 38.1
        ui_estatico = st.number_input("Nivel estático (m):", min_value=0.0, value=float(nivel_estatico_val), step=1.0)
    with col_var2:
        ui_bomba = st.number_input("Profundidad de instalación de la bomba (m):", min_value=10.0, value=91.68, step=1.0)
    with col_var3:
        ui_margen = st.number_input("Margen de seguridad sobre la bomba (m):", min_value=0.0, value=10.0, step=1.0)
    with col_var4:
        caudal_estab_val = df[col_caudal].dropna().iloc[-1]
        ui_caudal = st.number_input("Caudal de estabilización (LPS):", min_value=0.1, value=float(caudal_estab_val), step=0.1)

    # Extracción de métrica de abatimiento del aforo
    abatimiento_max = df[col_abat].dropna().max()
    
    # 1. Cálculo del Caudal Óptimo Seguro
    capacidad_especifica_real = ui_caudal / abatimiento_max if abatimiento_max > 0 else 0
    abatimiento_maximo_permitido = ui_bomba - ui_estatico - ui_margen
    caudal_optimo = capacidad_especifica_real * abatimiento_maximo_permitido

    # 2. Cálculo de Transmisividad (Método de Logan)
    ce_m3_dia_m = capacidad_especifica_real * 86.4
    transmisividad_logan = 1.22 * ce_m3_dia_m

    # Mostrar Resultados con Tarjetas (Metrics)
    st.markdown("### 🎯 Resultados Clave")
    res1, res2, res3 = st.columns(3)
    res1.metric("Caudal de Estabilización (Configurado)", f"{ui_caudal:.2f} LPS")
    res2.metric("Caudal Óptimo Recomendado", f"{caudal_optimo:.2f} LPS")
    res3.metric("Transmisividad Inferida (Acuífero)", f"{transmisividad_logan:.2f} m2/dia")

    # Explicaciones Ejecutivas
    st.markdown("### 📖 ¿Qué significan estos resultados?")
    
    with st.expander("1. Sobre el Caudal Óptimo Recomendado (Protección del Equipo)", expanded=True):
        st.markdown(f"""
        **¿Qué es?** Es el volumen máximo de agua que se puede extraer de manera continua sin correr el riesgo de que el nivel del agua descienda tanto que la bomba trabaje en vacío y sufra daños.
        
        **¿Cómo se calcula?**
        1. Se observa que el pozo rinde **{capacidad_especifica_real:.4f} Litros por Segundo** por cada metro que desciende el agua (Capacidad Específica).
        2. Se calcula el espacio disponible para que el agua descienda: Si la bomba se encuentra a **{ui_bomba} m**, el agua inicia en **{ui_estatico} m**, y se desea dejar un colchón de seguridad de **{ui_margen} m**, el agua solo tiene permitido descender un máximo de **{abatimiento_maximo_permitido:.2f} m** (Abatimiento Permitido).
        3. Se multiplica la capacidad del pozo por el espacio disponible: `{capacidad_especifica_real:.4f} * {abatimiento_maximo_permitido:.2f} = {caudal_optimo:.2f} LPS`.
        
        **Decisión:** Si el Caudal Óptimo resulta menor al Caudal de la Prueba, significa que el pozo fue forzado durante el aforo y la bomba debe seleccionarse considerando este nuevo valor conservador para asegurar su vida útil.
        """)

    with st.expander("2. Sobre la Transmisividad Inferida (Potencial del Acuífero)", expanded=True):
        st.markdown(f"""
        **¿Qué es?** La transmisividad (T) es un indicador que califica la facilidad con la que fluye el agua a través del medio geológico subterráneo. Valores altos indican acuíferos abundantes (gravas, arenas); valores bajos indican formaciones compactas o arcillosas que liberan el agua lentamente.
        
        **¿Cómo se calcula?**
        Dado que los datos iniciales de las pruebas a menudo presentan ruido hidrodinámico por la turbulencia dentro del pozo, se utiliza el **Método Empírico de Logan**, el cual se basa en la fase de estabilización final (el dato más confiable). 
        * Fórmula: `T = 1.22 * Capacidad Especifica (en m3/dia/m)`
        * Sustitución: `T = 1.22 * {ce_m3_dia_m:.2f} = {transmisividad_logan:.2f} m2/dia`
        
        **Decisión:** El valor de **{transmisividad_logan:.2f} m2/dia** representa la cifra que se utiliza como punto de partida para alimentar los modelos matemáticos hidrogeológicos. Este parámetro sustenta técnicamente si la obra de captación se encuentra en una zona de alta transmisividad o en una formación limitada.
        """)

    st.divider()

    # ==========================================
    # SECCIÓN 5: GEOVISOR ESPACIAL
    # ==========================================
    st.header("🗺️ 5. Geovisor Espacial de Captaciones")
    st.markdown("Integra tus puntos de extracción en el entorno espacial. Alterna entre mapas topográficos, hidrográficos y de localidades en el control de capas (esquina superior derecha del mapa).")

    # Controles para agregar puntos
    col_map1, col_map2 = st.columns([1, 2])
    
    with col_map1:
        st.subheader("📍 Agregar Puntos")
        tab_manual, tab_csv = st.tabs(["Ingreso Manual", "Cargar Archivo CSV"])
        
        with tab_manual:
            with st.form("form_mapa_manual"):
                nombre_punto = st.text_input("Identificador del Pozo:")
                lat_punto = st.number_input("Latitud (Decimales):", value=23.6345, format="%.6f")
                lon_punto = st.number_input("Longitud (Decimales):", value=-102.5528, format="%.6f")
                tipo_punto = st.selectbox("Tipo:", ["Pozo de Bombeo", "Piezómetro de Observación", "Manantial"])
                
                if st.form_submit_button("➕ Agregar al Mapa"):
                    nuevo_punto = pd.DataFrame([{'Identificador': nombre_punto, 'Latitud': lat_punto, 'Longitud': lon_punto, 'Tipo': tipo_punto}])
                    st.session_state.puntos_mapa = pd.concat([st.session_state.puntos_mapa, nuevo_punto], ignore_index=True)
                    st.success(f"Punto '{nombre_punto}' agregado.")
                    
        with tab_csv:
            st.info("El CSV debe contener las columnas: 'Identificador', 'Latitud', 'Longitud', 'Tipo'.")
            csv_mapa = st.file_uploader("Subir coordenadas (.csv)", type=["csv"], key="map_csv")
            if csv_mapa is not None:
                if st.button("📥 Importar Puntos"):
                    df_nuevos = pd.read_csv(csv_mapa)
                    st.session_state.puntos_mapa = pd.concat([st.session_state.puntos_mapa, df_nuevos], ignore_index=True)
                    st.success("Puntos importados correctamente.")
                    
        if st.button("🗑️ Limpiar todos los puntos del mapa"):
            st.session_state.puntos_mapa = pd.DataFrame(columns=['Identificador', 'Latitud', 'Longitud', 'Tipo'])
            st.rerun()

    with col_map2:
        if not st.session_state.puntos_mapa.empty:
            centro_lat = st.session_state.puntos_mapa['Latitud'].mean()
            centro_lon = st.session_state.puntos_mapa['Longitud'].mean()
            zoom_inicial = 10
        else:
            centro_lat, centro_lon = 23.6345, -102.5528
            zoom_inicial = 5

        m = folium.Map(location=[centro_lat, centro_lon], zoom_start=zoom_inicial)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Topografía y Localidades',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='OpenStreetMap', 
            name='Calles estándar (OSM)',
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satélite (Esri)',
            overlay=False,
            control=True
        ).add_to(m)

        colores_tipo = {"Pozo de Bombeo": "blue", "Piezómetro de Observación": "green", "Manantial": "lightblue"}

        for idx, row in st.session_state.puntos_mapa.iterrows():
            color_icono = colores_tipo.get(row.get('Tipo', 'Pozo de Bombeo'), "gray")
            folium.Marker(
                location=[row['Latitud'], row['Longitud']],
                popup=folium.Popup(f"<b>{row['Identificador']}</b><br>Lat: {row['Latitud']}<br>Lon: {row['Longitud']}", max_width=300),
                tooltip=row['Identificador'],
                icon=folium.Icon(color=color_icono, icon='tint')
            ).add_to(m)

        folium.LayerControl().add_to(m)

        st_folium(m, width=800, height=500, returned_objects=[])

    if not st.session_state.puntos_mapa.empty:
        st.subheader("📋 Tabla de Atributos y Simbología")
        st.dataframe(st.session_state.puntos_mapa, use_container_width=True)

    st.divider()

    # ==========================================
    # SECCIÓN 6: REINICIO DE CÁLCULOS
    # ==========================================
    if st.button("🔄 Empezar un nuevo cálculo (Subir otro CSV y limpiar mapa)", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

else:
    st.info("Esperando archivo CSV... Sube el documento para mapear las columnas y generar el análisis técnico y el resumen ejecutivo.")
