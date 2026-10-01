import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import exp1

st.header("Simulación Analítica del Cono de Abatimiento")
st.warning("⚠️ Nota: La geometría mostrada es una **estimación analítica** basada en la ecuación de Theis. La Transmisividad y el Coeficiente de Almacenamiento son valores asumidos teóricamente, ya que el aforo carece de pozos de observación espaciales.")

# Controles interactivos para manipular los parámetros en tiempo real
st.subheader("Parámetros Estimados (Editables)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Q se inicializa en 1.8 LPS según el aforo
    Q_lps = st.slider("Caudal de Extracción (LPS)", min_value=0.1, max_value=10.0, value=1.8, step=0.1)
with col2:
    T_m2d = st.slider("Transmisividad Estimada (m²/día)", min_value=10, max_value=1000, value=150, step=10)
with col3:
    # S asume un escenario semi-confinado por defecto (e.g., 0.001)
    S_coef = st.number_input("Almacenamiento Estimado (S)", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")
with col4:
    # Tiempo evaluado, inicializado en el tiempo final del aforo (72 hrs)
    t_hrs = st.slider("Tiempo de Bombeo (Horas)", min_value=1, max_value=240, value=72, step=1)

# Conversión de unidades para la fórmula analítica
Q_m3d = Q_lps * 86.4  # LPS a m3/día
t_d = t_hrs / 24.0    # Horas a días

# Generación del dominio espacial (Radio desde el pozo, de 0.1 m a 500 m)
radios = np.logspace(-1, 2.7, 100) # Distancias radiales logarítmicas para mayor detalle cerca del pozo

# Función de Theis para calcular el abatimiento estimado
def calcular_abatimiento(r, Q, T, S, t):
    u = (r**2 * S) / (4 * T * t)
    s = (Q / (4 * np.pi * T)) * exp1(u)  # exp1 es la aproximación de W(u)
    return s

# Cálculo de los abatimientos en el vector de distancias
abatimientos = calcular_abatimiento(radios, Q_m3d, T_m2d, S_coef, t_d)

# --- Renderizado de Gráficos ---
tab1, tab2 = st.tabs(["Superficie 3D (Cono Espacial)", "Perfil 2D (Sección Transversal)"])

with tab1:
    # Creación de la malla 3D (Superficie de revolución)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, Theta = np.meshgrid(radios, theta)
    
    # Coordenadas Cartesianas
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Expandir el vector de abatimiento a la malla circular e invertirlo para visualizar la depresión
    Z = -np.tile(abatimientos, (50, 1))
    
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
    
    fig_3d.update_layout(
        title="Estimación del Cono de Depresión (3D)",
        scene=dict(
            xaxis_title="Distancia X (m)",
            yaxis_title="Distancia Y (m)",
            zaxis_title="Abatimiento (m)",
            zaxis=dict(range=[np.min(Z)*1.1, 0])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    # Perfil radial 2D clásico
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=radios, y=abatimientos, mode='lines', fill='tozeroy', line=dict(color='blue', width=3)))
    
    fig_2d.update_layout(
        title="Perfil Estimado de Abatimiento Radial",
        xaxis_title="Distancia desde el pozo de bombeo (m)",
        yaxis_title="Abatimiento Estimado (m)",
        yaxis=dict(autorange="reversed"), # Invertir eje Y para mostrar descenso
        xaxis_type="log" # Escala logarítmica para ver mejor el radio cercano
    )
    st.plotly_chart(fig_2d, use_container_width=True)
