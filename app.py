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

uploaded_file = st.file_uploader("Sube tu archivo de aforo aquí (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columnas_csv = df.columns.tolist()
    
    st.subheader("⚙️ Configuración de Variables")
    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
    
    with col_sel1:
        idx_t = columnas_csv.index('Tiempo_hrs') if 'Tiempo_hrs' in columnas_csv else 0
        col_tiempo = st.selectbox("Tiempo (Horas):", columnas_csv, index=idx_t)
        
    with col_sel2:
        idx_q = columnas_csv.index('LPS') if 'LPS' in columnas_csv else 0
        col_caudal = st.selectbox("Caudal (LPS):", columnas_csv, index=idx_q)
        
    with col_sel3:
        idx_s = columnas_csv.index('Abatimiento') if 'Abatimiento' in columnas_csv else 0
        col_abat = st.selectbox("Abatimiento (m):", columnas_csv, index=idx_s)
        
    with col_sel4:
        idx_hz = columnas_csv.index('Hz') if 'Hz' in columnas_csv else 0
        col_hz = st.selectbox("Frecuencia (Hz):", columnas_csv, index=idx_hz)

    # Conversión numérica de seguridad
    for col in [col_tiempo, col_caudal, col_abat, col_hz]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Cálculo de Capacidad Específica (Q/s)
    df['Capacidad_Especifica'] = np.where(df[col_abat] > 0, df[col_caudal] / df[col_abat], 0)

    with st.expander("Ver tabla de datos procesados"):
        st.dataframe(df)

    st.divider()

    # ==========================================
    # SECCIÓN 2: GRÁFICOS REALES Y DESCRIPCIONES
    # ==========================================
    st.header("📉 2. Análisis del Comportamiento del Pozo")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Abatimiento Lineal", 
        "Abatimiento (Ejes Invertidos)", 
        "Cooper-Jacob", 
        "Caudal de Extracción",
        "Capacidad Específica",
        "Curva de Operación"
    ])

    with tab1:
        st.markdown("**Utilidad:** Observar la tendencia general del descenso del nivel dinámico y confirmar el momento exacto en que la curva se vuelve asintótica (estabilización).")
        fig1 = px.line(df, x=col_tiempo, y=col_abat, markers=True, title="Evolución del Abatimiento vs Tiempo")
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("**Utilidad:** Proporciona una perspectiva física más intuitiva de la rapidez con la que 'cae' el nivel de agua en el pozo conforme avanza la prueba.")
        fig2 = px.line(df, x=col_abat, y=col_tiempo, markers=True, title="Tiempo transcurrido vs Abatimiento")
        fig2.update_layout(xaxis_title="Abatimiento (m)", yaxis_title="Tiempo (Horas)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("**Utilidad:** Aísla la fase de flujo radial. Al graficar en escala semilogarítmica, los datos tienden a formar una recta. La pendiente de esta recta permite calcular la Transmisividad ($T$) del acuífero.")
        df_log = df[df[col_tiempo] > 0]
        fig3 = px.scatter(df_log, x=col_tiempo, y=col_abat, log_x=True, title="Método de Cooper-Jacob")
        fig3.update_layout(xaxis_title="Tiempo (Horas) [Escala Log]", yaxis_title="Abatimiento (m)")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("**Utilidad:** Verifica la estabilidad del bombeo. Permite identificar variaciones operativas antes de alcanzar el gasto de diseño recomendado.")
        fig4 = px.line(df, x=col_tiempo, y=col_caudal, markers=True, title="Comportamiento del Caudal (LPS)")
        fig4.update_traces(line_color='green')
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        st.markdown("**Utilidad:** Muestra la eficiencia hidráulica ($Q/s$). Una caída abrupta o sostenida puede indicar turbulencia excesiva en la rejilla o la intercepción de fronteras impermeables en el acuífero.")
        fig5 = px.line(df, x=col_tiempo, y='Capacidad_Especifica', markers=True, title="Evolución de la Capacidad Específica (LPS/m)")
        fig5.update_traces(line_color='purple')
        st.plotly_chart(fig5, use_container_width=True)

    with tab6:
        st.markdown("**Utilidad:** Relaciona los Hz del motor frente a los LPS extraídos. Fundamental para justificar técnicamente el dimensionamiento y selección del equipo electromecánico definitivo.")
        fig6 = px.scatter(df, x=col_hz, y=col_caudal, color=col_tiempo, title="Desempeño Electromecánico (Hz vs LPS)", color_continuous_scale='viridis')
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # ==========================================
    # SECCIÓN 3: SIMULACIÓN TEÓRICA DEL CONO (THEIS)
    # ==========================================
    st.header("🌐 3. Simulación Analítica del Cono de Abatimiento")
    st.markdown("Cálculo iterativo del radio de influencia mediante la solución de Theis. Ideal para validaciones analíticas previas a la configuración de modelos numéricos.")
    
    caudal_final_lps = df[col_caudal].dropna().iloc[-1]
    tiempo_final_hrs = df[col_tiempo].max()
    
    st.info(f"📌 **Condiciones de frontera extraídas del aforo:** Caudal = **{caudal_final_lps} LPS** | Tiempo = **{tiempo_final_hrs} Horas**")

    col_t, col_s = st.columns(2)
    with col_t:
        T_m2d = st.slider("Transmisividad (m²/día) [T]", min_value=10, max_value=1000, value=150, step=10)
    with col_s:
        S_coef = st.number_input("Almacenamiento [S]", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")

    # Theis
    Q_m3d = caudal_final_lps * 86.4
    t_d = tiempo_final_hrs / 24.0
    radios = np.logspace(-1, 2.7, 100)
    u = (radios**2 * S_coef) / (4 * T_m2d * t_d)
    abatimientos = (Q_m3d / (4 * np.pi * T_m2d)) * exp1(u)

    tab_3d, tab_2d = st.tabs(["Superficie 3D (Radio de Influencia)", "Perfil 2D Transversal"])

    with tab_3d:
        st.markdown("**Utilidad:** Representación espacial de la depresión del nivel piezométrico alrededor del pozo. Ayuda a prever interferencias con captaciones vecinas.")
        theta = np.linspace(0, 2 * np.pi, 50)
        R, Theta = np.meshgrid(radios, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Z = -np.tile(abatimientos, (50, 1))
        
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
        fig_3d.update_layout(scene=dict(zaxis=dict(range=[np.min(Z)*1.1, 0])), margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_2d:
        st.markdown("**Utilidad:** Cuantifica la magnitud del abatimiento a diferentes distancias específicas ($r$) desde el eje de extracción.")
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(x=radios, y=abatimientos, mode='lines', fill='tozeroy'))
        fig_2d.update_layout(xaxis_type="log", yaxis=dict(autorange="reversed"), xaxis_title="Distancia Radial (m)", yaxis_title="Abatimiento Teórico (m)")
        st.plotly_chart(fig_2d, use_container_width=True)

else:
    st.info("Esperando archivo CSV... Sube el documento para mapear las columnas y generar todos los gráficos del aforo.")
