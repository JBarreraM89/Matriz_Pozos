import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.special import exp1

# --- NUEVA FUNCIÓN: Algoritmo de Detección de Flujo Radial ---
def detectar_flujo_radial(df, col_t, col_s):
    """
    Escanea la serie de tiempo para encontrar el segmento donde la pendiente 
    semilogarítmica es más constante (menor varianza de la derivada), 
    excluyendo zonas donde el abatimiento ya se estabilizó (pendiente nula).
    """
    d = df[df[col_t] > 0].dropna(subset=[col_t, col_s]).reset_index(drop=True)
    if len(d) < 4:
        return float(d[col_t].min()), float(d[col_t].max())
    
    # Calcular log(t) y la primera derivada punto a punto
    log_t = np.log10(d[col_t])
    s = d[col_s]
    derivada = np.gradient(s, log_t)
    
    mejor_ventana = None
    menor_var = float('inf')
    window_size = min(4, len(d)) # Evaluar en bloques de 4 registros
    
    for i in range(len(d) - window_size + 1):
        ventana = derivada[i:i+window_size]
        promedio_pendiente = np.mean(ventana)
        
        # Filtro: Ignorar la estabilización del nivel (donde Delta_s tiende a 0)
        if promedio_pendiente > 0.5:  
            varianza = np.var(ventana)
            if varianza < menor_var:
                menor_var = varianza
                mejor_ventana = (float(d[col_t].iloc[i]), float(d[col_t].iloc[i+window_size-1]))
                
    if mejor_ventana:
        return mejor_ventana
    else:
        # Fallback si no encuentra un buen segmento
        return float(d[col_t].min()), float(d[col_t].iloc[len(d)//3])


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

    for col in [col_tiempo, col_caudal, col_abat, col_hz]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Capacidad_Especifica'] = np.where(df[col_abat] > 0, df[col_caudal] / df[col_abat], 0)

    with st.expander("Ver tabla de datos procesados"):
        st.dataframe(df)

    st.divider()

    # ==========================================
    # SECCIÓN 2: ANÁLISIS
    # ==========================================
    st.header("📉 2. Análisis del Comportamiento del Pozo")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Abatimiento Lineal", 
        "Abatimiento (Invertido)", 
        "Cooper-Jacob (Auto-Ajuste)", 
        "Caudal",
        "Capacidad Específica",
        "Curva de Operación"
    ])

    with tab1:
        fig1 = px.line(df, x=col_tiempo, y=col_abat, markers=True, title="Evolución del Abatimiento vs Tiempo")
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = px.line(df, x=col_abat, y=col_tiempo, markers=True, title="Tiempo transcurrido vs Abatimiento")
        fig2.update_layout(xaxis_title="Abatimiento (m)", yaxis_title="Tiempo (Horas)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("**Utilidad:** La app analiza la derivada del abatimiento para evadir la zona de estabilización y recomendar el segmento de flujo radial óptimo para calcular $T$.")
        
        df_log = df[df[col_tiempo] > 0].dropna(subset=[col_tiempo, col_abat, col_caudal])
        
        if not df_log.empty:
            min_t_global, max_t_global = float(df_log[col_tiempo].min()), float(df_log[col_tiempo].max())
            
            # Ejecutar algoritmo de recomendación
            t_rec_inicio, t_rec_fin = detectar_flujo_radial(df_log, col_tiempo, col_abat)
            
            # El slider arranca con los valores recomendados por la app
            rango_t = st.slider("Segmento de flujo radial (Recomendación automática):", 
                                min_value=min_t_global, max_value=max_t_global, 
                                value=(t_rec_inicio, t_rec_fin), step=0.1)
            
            mask = (df_log[col_tiempo] >= rango_t[0]) & (df_log[col_tiempo] <= rango_t[1])
            df_fit = df_log[mask]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_log[col_tiempo], y=df_log[col_abat], mode='markers', name='Datos Aforo', marker=dict(color='blue')))
            
            if len(df_fit) > 1:
                x_fit = np.log10(df_fit[col_tiempo])
                y_fit = df_fit[col_abat]
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                
                caudal_ajuste_lps = df_fit[col_caudal].mean() 
                Q_m3d_ajuste = caudal_ajuste_lps * 86.4
                delta_s = abs(slope)
                
                if delta_s > 0.01: # Evitar división por cero si el usuario fuerza un rango plano
                    T_calc = (0.183 * Q_m3d_ajuste) / delta_s
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric("Pendiente (Δs)", f"{delta_s:.2f} m")
                    col_res2.metric("Caudal Tramo (Q)", f"{caudal_ajuste_lps:.2f} LPS")
                    col_res3.metric("Transmisividad (T)", f"{T_calc:.2f} m²/día")
                    
                    x_line = np.linspace(rango_t[0], rango_t[1], 50)
                    y_line = slope * np.log10(x_line) + intercept
                    fig3.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Ajuste Lineal', line=dict(color='red', dash='dash', width=3)))
                else:
                    st.warning("El segmento seleccionado es demasiado plano (estabilizado). Amplía el rango hacia la izquierda para atrapar la fase de descenso.")
            
            fig3.update_layout(title="Método de Cooper-Jacob", xaxis_type="log", xaxis_title="Tiempo (Horas) [Log]", yaxis_title="Abatimiento (m)")
            st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = px.line(df, x=col_tiempo, y=col_caudal, markers=True, title="Comportamiento del Caudal (LPS)")
        fig4.update_traces(line_color='green')
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        fig5 = px.line(df, x=col_tiempo, y='Capacidad_Especifica', markers=True, title="Capacidad Específica (LPS/m)")
        fig5.update_traces(line_color='purple')
        st.plotly_chart(fig5, use_container_width=True)

    with tab6:
        fig6 = px.scatter(df, x=col_hz, y=col_caudal, color=col_tiempo, title="Desempeño Electromecánico (Hz vs LPS)", color_continuous_scale='viridis')
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # ==========================================
    # SECCIÓN 3: SIMULACIÓN TEÓRICA DEL CONO
    # ==========================================
    st.header("🌐 3. Simulación Analítica del Cono de Abatimiento")
    
    caudal_final_lps = df[col_caudal].dropna().iloc[-1]
    tiempo_final_hrs = df[col_tiempo].max()
    
    st.info(f"📌 **Condiciones extraídas:** Caudal = **{caudal_final_lps} LPS** | Tiempo = **{tiempo_final_hrs} Horas**")

    T_default = int(T_calc) if 'T_calc' in locals() and 10 <= T_calc <= 5000 else 150

    col_t, col_s = st.columns(2)
    with col_t:
        T_m2d = st.slider("Transmisividad (m²/día) [T]", min_value=10, max_value=5000, value=T_default, step=10)
    with col_s:
        S_coef = st.number_input("Almacenamiento [S]", min_value=0.00001, max_value=0.3, value=0.001, format="%.5f")

    Q_m3d = caudal_final_lps * 86.4
    t_d = tiempo_final_hrs / 24.0
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

else:
    st.info("Esperando archivo CSV...")
