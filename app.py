import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.special import exp1

st.set_page_config(page_title="Análisis de Aforo de Pozos", layout="wide")

st.title("💧 Sistema de Análisis de Pruebas de Aforo")

# ==========================================
# SECCIÓN 1: CARGA DE DATOS Y MAPEO DE COLUMNAS
# ==========================================
st.header("📊 1. Carga de Datos y Configuración")
st.markdown("Sube tu archivo `.csv` y selecciona las columnas correspondientes para realizar los cálculos.")

uploaded_file = st.file_uploader("Sube tu archivo de aforo aquí (.csv)", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    df = pd.read_csv(uploaded_file)
    columnas_csv = df.columns.tolist()
    
    st.success("¡Archivo cargado correctamente!")
    
    st.subheader("⚙️ Configuración de Variables")
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    with col_sel1:
        # Intentar pre-seleccionar si encuentra nombres comunes
        idx_t = columnas_csv.index('Tiempo_hrs') if 'Tiempo_hrs' in columnas_csv else 0
        col_tiempo = st.selectbox("Columna de Tiempo (Horas):", columnas_csv, index=idx_t)
        
    with col_sel2:
        idx_q = columnas_csv.index('LPS') if 'LPS' in columnas_csv else 0
        col_caudal = st.selectbox("Columna de Caudal (LPS):", columnas_csv, index=idx_q)
        
    with col_sel3:
        idx_s = columnas_csv.index('Abatimiento') if 'Abatimiento' in columnas_csv else 0
        col_abat = st.selectbox("Columna de Abatimiento (m):", columnas_csv, index=idx_s)

    # Convertir a numérico por seguridad, forzando errores a NaN
    df[col_tiempo] = pd.to_numeric(df[col_tiempo], errors='coerce')
    df[col_caudal] = pd.to_numeric(df[col_caudal], errors='coerce')
    df[col_abat] = pd.to_numeric(df[col_abat], errors='coerce')
    
    # Calcular Capacidad Específica (evitando división por cero)
    df['Capacidad_Especifica'] = np.where(df[col_abat] > 0, df[col_caudal] / df[col_abat], 0)

    # Mostrar tabla resumen
    with st.expander("Ver tabla de datos procesados"):
        st.dataframe(df)

    st.divider()

    # ==========================================
    # SECCIÓN 2: GRÁFICOS REALES
    # ==========================================
    st.header("📉 2. Análisis del Comportamiento del Pozo")
    
    tab_abatimiento, tab_jacob, tab_caudal = st.tabs([
        "Abatimiento vs Tiempo", 
        "Cooper-Jacob", 
        "Caudal y Cap. Específica"
    ])

    with tab_abatimiento:
        fig_abat = px.line(df, x=col_tiempo, y=col_abat, markers=True,
                           title="Evolución del Abatimiento durante el Aforo")
        fig_abat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_abat, use_container_width=True)

    with tab_jacob:
        df_log = df[df[col_tiempo] > 0]
        fig_jacob = px.scatter(df_log, x=col_tiempo, y=col_abat, log_x=True,
                               title="Método de Cooper-Jacob (Escala Semilogarítmica)")
        st.plotly_chart(fig_jacob, use_container_width=True)

    with tab_caudal:
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df[col_tiempo], y=df[col_caudal], mode='lines+markers', name='Caudal (LPS)'))
        fig_q.add_trace(go.Scatter(x=df[col_tiempo], y=df['Capacidad_Especifica'], mode='lines+markers', name='Cap. Esp. (LPS/m)', yaxis='y2'))
        fig_q.update_layout(
            title="Evolución del Caudal y Capacidad Específica",
            yaxis=dict(title='Caudal (LPS)', side='left'),
            yaxis2=dict(title='Capacidad Específica (LPS/m)', side='right', overlaying='y')
        )
        st.plotly_chart(fig_q, use_container_width=True)

    st.divider()

    # ==========================================
    # SECCIÓN 3: SIMULACIÓN TEÓRICA DEL CONO (THEIS)
    # ==========================================
    st.header("🌐 3. Simulación Analítica del Cono de Abatimiento")
    st.markdown("Cálculo del radio de influencia mediante la solución de Theis. **El caudal y el tiempo se han extraído automáticamente de la estabilización del aforo.**")
    
    # Extraer valores reales del DataFrame para la simulación
    caudal_final_lps = df[col_caudal].iloc[-1]
    tiempo_final_hrs = df[col_tiempo].max()
    
    st.info(f"📌 **Datos utilizados del CSV:** Caudal de estabilización = **{caudal_final_lps} LPS** | Tiempo de bombeo = **{tiempo_final_hrs} Horas**")

    st.subheader("Parámetros del Acuífero (Editables)")
    col_t, col_s = st.columns(2)
    with col_t:
        T_m2d = st.slider("Transmisividad Asumida (m²/día) [T]", min_value=10, max_value=1000, value=150, step=10)
    with col_s:
        S_coef = st.number_input("Almacenamiento Asumido [S] (adimensional)", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")

    # Cálculos analíticos de Theis
    Q_m3d = caudal_final_lps * 86.4
    t_d = tiempo_final_hrs / 24.0
    radios = np.logspace(-1, 2.7, 100) # Radios desde 0.1 hasta ~500m
    
    # Ecuaciones de Theis: u = (r^2 * S) / (4 * T * t) y s = (Q / 4*pi*T) * W(u)
    u = (radios**2 * S_coef) / (4 * T_m2d * t_d)
    abatimientos = (Q_m3d / (4 * np.pi * T_m2d)) * exp1(u)

    tab_3d, tab_2d = st.tabs(["Superficie 3D (Cono)", "Perfil 2D"])

    with tab_3d:
        theta = np.linspace(0, 2 * np.pi, 50)
        R, Theta = np.meshgrid(radios, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Z = -np.tile(abatimientos, (50, 1))
        
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
        fig_3d.update_layout(
            scene=dict(
                zaxis=dict(range=[np.min(Z)*1.1, 0]),
                xaxis_title="Radio X (m)",
                yaxis_title="Radio Y (m)",
                zaxis_title="Abatimiento (m)"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_2d:
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(x=radios, y=abatimientos, mode='lines', fill='tozeroy'))
        fig_2d.update_layout(
            xaxis_type="log", 
            yaxis=dict(autorange="reversed"), 
            xaxis_title="Distancia Radial desde el pozo (m)", 
            yaxis_title="Abatimiento Teórico (m)"
        )
        st.plotly_chart(fig_2d, use_container_width=True)

else:
    st.info("Esperando archivo CSV... Sube el documento para mapear las columnas y visualizar el análisis hidrogeológico.")
