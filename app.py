import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.special import exp1

st.set_page_config(page_title="Análisis de Aforo de Pozos", layout="wide")

st.title("💧 Sistema de Análisis de Pruebas de Aforo")

# ==========================================
# SECCIÓN 1: CARGA DE DATOS Y GRÁFICOS REALES
# ==========================================
st.header("📊 1. Análisis de Datos de Campo (Carga de CSV)")
st.markdown("""
**Instrucciones:** Sube un archivo `.csv` que contenga exactamente estas columnas:
`Tiempo_hrs`, `Hz`, `LPS`, `Nivel_Estatico`, `Nivel_Dinamico`, `Abatimiento`
""")

uploaded_file = st.file_uploader("Sube tu archivo de aforo aquí (.csv)", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    df = pd.read_csv(uploaded_file)
    
    # Calcular Capacidad Específica (evitando división por cero)
    df['Capacidad_Especifica'] = np.where(df['Abatimiento'] > 0, df['LPS'] / df['Abatimiento'], 0)
    
    st.success("¡Archivo cargado correctamente!")
    
    # Mostrar tabla resumen
    with st.expander("Ver tabla de datos procesados"):
        st.dataframe(df)

    # Crear pestañas para organizar los gráficos
    tab_abatimiento, tab_jacob, tab_caudal, tab_motor = st.tabs([
        "Abatimiento vs Tiempo", 
        "Cooper-Jacob", 
        "Caudal y Cap. Específica", 
        "Operación del Motor"
    ])

    with tab_abatimiento:
        fig_abat = px.line(df, x='Tiempo_hrs', y='Abatimiento', markers=True,
                           title="Evolución del Abatimiento durante el Aforo")
        fig_abat.update_yaxes(autorange="reversed") # Invertir eje Y
        st.plotly_chart(fig_abat, use_container_width=True)

    with tab_jacob:
        df_log = df[df['Tiempo_hrs'] > 0] # Evitar log(0)
        fig_jacob = px.scatter(df_log, x='Tiempo_hrs', y='Abatimiento', log_x=True,
                               title="Método de Cooper-Jacob (Escala Semilogarítmica)")
        st.plotly_chart(fig_jacob, use_container_width=True)

    with tab_caudal:
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df['Tiempo_hrs'], y=df['LPS'], mode='lines+markers', name='Caudal (LPS)'))
        fig_q.add_trace(go.Scatter(x=df['Tiempo_hrs'], y=df['Capacidad_Especifica'], mode='lines+markers', name='Cap. Esp. (LPS/m)', yaxis='y2'))
        fig_q.update_layout(
            title="Evolución del Caudal y Capacidad Específica",
            yaxis=dict(title='Caudal (LPS)', side='left'),
            yaxis2=dict(title='Capacidad Específica (LPS/m)', side='right', overlaying='y')
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with tab_motor:
        fig_op = px.scatter(df, x='Hz', y='LPS', color='Tiempo_hrs',
                            title="Desempeño Electromecánico (Hz vs LPS)")
        st.plotly_chart(fig_op, use_container_width=True)

else:
    st.info("Esperando archivo CSV para generar los gráficos reales del pozo...")

st.divider()

# ==========================================
# SECCIÓN 2: SIMULACIÓN TEÓRICA DEL CONO
# ==========================================
st.header("🌐 2. Simulación Analítica del Cono de Abatimiento")
st.warning("⚠️️ Nota: Esta es una estimación teórica basada en la ecuación de Theis. Los parámetros son asumidos y editables.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    Q_lps = st.slider("Caudal (LPS)", min_value=0.1, max_value=10.0, value=1.8, step=0.1)
with col2:
    T_m2d = st.slider("Transmisividad (m²/día)", min_value=10, max_value=1000, value=150, step=10)
with col3:
    S_coef = st.number_input("Almacenamiento (S)", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")
with col4:
    t_hrs = st.slider("Tiempo (Horas)", min_value=1, max_value=240, value=72, step=1)

# Cálculos Theis
Q_m3d = Q_lps * 86.4
t_d = t_hrs / 24.0
radios = np.logspace(-1, 2.7, 100)
u = (radios**2 * S_coef) / (4 * T_m2d * t_d)
abatimientos = (Q_m3d / (4 * np.pi * T_m2d)) * exp1(u)

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
    fig_2d.update_layout(xaxis_type="log", yaxis=dict(autorange="reversed"), xaxis_title="Distancia (m)", yaxis_title="Abatimiento (m)")
    st.plotly_chart(fig_2d, use_container_width=True)
