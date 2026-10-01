import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.special import exp1

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
# SECCIÓN 1: CARGA DE DATOS Y MAPEO
# ==========================================
st.header("📊 1. Carga de Datos y Configuración")
uploaded_file = st.file_uploader("Sube tu archivo de aforo aquí (.csv)", type=["csv"])

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
        # Intentar detectar la columna de Nivel Estático si existe para usarla después
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
    tab1, tab2, tab3, tab4 = st.tabs(["Abatimiento", "Cooper-Jacob", "Eficiencia", "Operación"])

    with tab1:
        fig1 = px.line(df, x=col_tiempo, y=col_abat, markers=True, title="Evolución del Abatimiento vs Tiempo")
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        df_log = df[df[col_tiempo] > 0].dropna(subset=[col_tiempo, col_abat, col_caudal])
        if not df_log.empty:
            t_rec_inicio, t_rec_fin = detectar_flujo_radial(df_log, col_tiempo, col_abat)
            rango_t = st.slider("Segmento de flujo radial (Recomendación automática):", min_value=float(df_log[col_tiempo].min()), max_value=float(df_log[col_tiempo].max()), value=(t_rec_inicio, t_rec_fin), step=0.1)
            
            mask = (df_log[col_tiempo] >= rango_t[0]) & (df_log[col_tiempo] <= rango_t[1])
            df_fit = df_log[mask]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_log[col_tiempo], y=df_log[col_abat], mode='markers', name='Datos'))
            
            if len(df_fit) > 1:
                slope, intercept = np.polyfit(np.log10(df_fit[col_tiempo]), df_fit[col_abat], 1)
                delta_s = abs(slope)
                if delta_s > 0.01:
                    T_calc = (0.183 * (df_fit[col_caudal].mean() * 86.4)) / delta_s
                    x_line = np.linspace(rango_t[0], rango_t[1], 50)
                    fig3.add_trace(go.Scatter(x=x_line, y=slope*np.log10(x_line)+intercept, mode='lines', name='Ajuste', line=dict(color='red', dash='dash')))
            fig3.update_layout(xaxis_type="log", xaxis_title="Tiempo [Log]", yaxis_title="Abatimiento (m)")
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        fig5 = px.line(df, x=col_tiempo, y='Capacidad_Especifica', markers=True, title="Capacidad Específica (LPS/m)")
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        fig6 = px.scatter(df, x=col_hz, y=col_caudal, color=col_tiempo, title="Desempeño Electromecánico (Hz vs LPS)")
        st.plotly_chart(fig6, use_container_width=True)
    st.divider()

    # ==========================================
    # SECCIÓN 3: SIMULACIÓN DEL CONO
    # ==========================================
    st.header("🌐 3. Simulación Analítica del Cono de Abatimiento")
    caudal_final_lps = df[col_caudal].dropna().iloc[-1]
    tiempo_final_hrs = df[col_tiempo].max()
    
    col_t, col_s = st.columns(2)
    with col_t:
        T_m2d = st.slider("Transmisividad Teórica (m²/día) [T]", min_value=10.0, max_value=5000.0, value=150.0, step=10.0)
    with col_s:
        S_coef = st.number_input("Almacenamiento Teórico [S]", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")

    radios = np.logspace(-1, 2.7, 100)
    u = (radios**2 * S_coef) / (4 * T_m2d * (tiempo_final_hrs / 24.0))
    abatimientos = ((caudal_final_lps * 86.4) / (4 * np.pi * T_m2d)) * exp1(u)

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
    **Modifica los parámetros físicos de la instalación para simular escenarios operativos.**
    """)

    st.subheader("🛠️ Variables de la Instalación (Editables)")
    col_var1, col_var2, col_var3 = st.columns(3)
    
    with col_var1:
        nivel_estatico_val = df[col_estatico].dropna().iloc[0] if col_estatico in df.columns else 38.1
        ui_estatico = st.number_input("Nivel del agua en reposo (m):", min_value=0.0, value=float(nivel_estatico_val), step=1.0)
    with col_var2:
        ui_bomba = st.number_input("Profundidad de instalación de la bomba (m):", min_value=10.0, value=91.68, step=1.0)
    with col_var3:
        ui_margen = st.number_input("Margen de seguridad sobre la bomba (m):", min_value=0.0, value=10.0, step=1.0)

    # Extracción de métricas clave estabilizadas del aforo
    abatimiento_max = df[col_abat].dropna().max()
    caudal_estab = df[col_caudal].dropna().iloc[-1]
    
    # 1. Cálculo del Caudal Óptimo Seguro
    capacidad_especifica_real = caudal_estab / abatimiento_max if abatimiento_max > 0 else 0
    abatimiento_maximo_permitido = ui_bomba - ui_estatico - ui_margen
    caudal_optimo = capacidad_especifica_real * abatimiento_maximo_permitido

    # 2. Cálculo de Transmisividad (Método de Logan)
    # Convertimos Capacidad Específica de LPS/m a (m3/día)/m multiplicando por 86.4
    ce_m3_dia_m = capacidad_especifica_real * 86.4
    transmisividad_logan = 1.22 * ce_m3_dia_m

    # Mostrar Resultados con Tarjetas (Metrics)
    st.markdown("### 🎯 Resultados Clave")
    res1, res2, res3 = st.columns(3)
    res1.metric("Caudal de Estabilización (Prueba)", f"{caudal_estab:.2f} LPS")
    res2.metric("Caudal Óptimo Recomendado", f"{caudal_optimo:.2f} LPS")
    res3.metric("Transmisividad Inferida (Acuífero)", f"{transmisividad_logan:.2f} m²/día")

    # Explicaciones Ejecutivas
    st.markdown("### 📖 ¿Qué significan estos resultados?")
    
    with st.expander("1. Sobre el Caudal Óptimo Recomendado (Protección del Equipo)", expanded=True):
        st.markdown(f"""
        **¿Qué es?** Es el volumen máximo de agua que puedes extraer de manera continua sin correr el riesgo de que el nivel del agua baje tanto que la bomba trabaje en vacío y se queme.
        
        **¿Cómo se calculó?**
        1. Observamos que el pozo rinde **{capacidad_especifica_real:.4f} Litros por Segundo** por cada metro que desciende el agua (Capacidad Específica).
        2. Calculamos el espacio disponible para que el agua baje: Si la bomba está a **{ui_bomba} m**, el agua inicia en **{ui_estatico} m**, y queremos dejar un colchón de seguridad de **{ui_margen} m**, el agua solo tiene permiso de bajar un máximo de **{abatimiento_maximo_permitido:.2f} m** (Abatimiento Permitido).
        3. Multiplicamos la capacidad del pozo por ese espacio disponible: `{capacidad_especifica_real:.4f} × {abatimiento_maximo_permitido:.2f} = {caudal_optimo:.2f} LPS`.
        
        **Decisión:** Si el Caudal Óptimo es menor al Caudal de la Prueba, significa que el pozo fue forzado durante el aforo y la bomba debe seleccionarse considerando este nuevo valor conservador.
        """)

    with st.expander("2. Sobre la Transmisividad Inferida (Potencial del Acuífero)", expanded=True):
        st.markdown(f"""
        **¿Qué es?** La transmisividad ($T$) es una calificación de qué tan fácil fluye el agua a través de las rocas y tierras subterráneas. Valores altos indican acuíferos abundantes (gravas, arenas); valores bajos indican rocas apretadas o arcillosas que sueltan el agua muy lentamente.
        
        **¿Cómo se calculó?**
        Debido a que los datos iniciales de las pruebas a menudo presentan ruido por la turbulencia dentro del pozo, utilizamos el **Método Empírico de Logan**, el cual se basa en la fase de estabilización final (el dato más confiable). 
        * Fórmula: `$T = 1.22 \times \text{Capacidad Específica (en m³/día/m)}$`
        * Sustitución: `$T = 1.22 \times {ce_m3_dia_m:.2f} = {transmisividad_logan:.2f} \text{ m²/día}$`
        
        **Decisión:** Un valor de **{transmisividad_logan:.2f} m²/día** es la cifra que los geohidrólogos utilizarán como punto de partida para alimentar los modelos matemáticos regionales. Ayuda a sustentar si la obra de captación se encuentra en una zona de alta recarga o en una formación limitada.
        """)

else:
    st.info("Esperando archivo CSV... Sube el documento para mapear las columnas y generar el análisis técnico y el resumen ejecutivo.")
