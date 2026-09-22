import streamlit as st
import math
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Cotizador de Pozos", page_icon="💧", layout="wide")

st.title("💧 Matriz Dinámica: Construcción y Equipamiento de Pozos")
st.markdown("Cotizador paramétrico basado en variables hidrogeológicas.")

# --- BARRA LATERAL: VARIABLES DE ENTRADA ---
st.sidebar.header("1. Variables de Diseño")

litologia = st.sidebar.selectbox("Litología Dominante", ["Material Suave (Granular/Arcillas)", "Material Duro (Basaltos/Calizas)"])
profundidad = st.sidebar.number_input("Profundidad Total (m)", min_value=50, max_value=800, value=250, step=10)
caudal = st.sidebar.number_input("Caudal Esperado (L/s)", min_value=1, max_value=200, value=30, step=5)
calidad_agua = st.sidebar.selectbox("Calidad del Agua", ["Agua Dulce", "Agua Corrosiva/Termal"])
nivel_dinamico = st.sidebar.number_input("Nivel Dinámico (m)", min_value=10, max_value=600, value=120, step=10)

# --- LÓGICA DE PARÁMETROS TÉCNICOS ---
# 1. Perforación
metodo_perf = "Rotatoria con Lodo" if "Suave" in litologia else "Rotopercusión con Aire"

# 2. Diámetros
if caudal < 15:
    bomba, ademe, rima = "6 pulgadas", "8 pulgadas", "14 pulgadas"
elif caudal <= 40:
    bomba, ademe, rima = "8 pulgadas", "12 pulgadas", "18 pulgadas"
else:
    bomba, ademe, rima = "10 a 12 pulgadas", "16 pulgadas", "24 pulgadas"

# 3. Potencia (Fórmula: Q * CDT / 76 * Eficiencia)
cdt = nivel_dinamico + 20 # Se suman 20m por pérdidas por fricción y carga en superficie
hp = math.ceil((caudal * cdt) / (76 * 0.65))

# --- LÓGICA DE COSTOS PARAMÉTRICOS ---
# Etapa 1
costo_e1 = 250000 

# Etapa 2
costo_perf_metro = 3000 if "Suave" in litologia else 6500
costo_e2 = (profundidad * costo_perf_metro) + 150000 # Incluye movilización

# Etapa 3
costo_acero_metro = 3500 if calidad_agua == "Agua Dulce" else 8500
costo_e3 = (profundidad * costo_acero_metro) + (profundidad * 1000) # Incluye desarrollo

# Etapa 4
costo_e4 = 120000

# Etapa 5
base_equip = 150000 if caudal < 15 else (250000 if caudal <= 40 else 450000)
costo_e5 = (hp * 3000) + (nivel_dinamico * 1500) + base_equip

total = costo_e1 + costo_e2 + costo_e3 + costo_e4 + costo_e5

# --- INTERFAZ DE USUARIO ---
st.subheader("2. Parámetros Técnicos Calculados")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Método", "Rotatoria" if "Suave" in litologia else "Percusión")
col2.metric("Ademe", ademe)
col3.metric("Rima", rima)
col4.metric("Bomba", bomba)
col5.metric("Potencia (HP)", f"{hp} HP")

st.markdown("---")
st.subheader("3. Presupuesto Dinámico Estimado")

# Tabla de desglose
datos_presupuesto = {
    "Etapa": ["ETAPA 1: Estudios Previos", "ETAPA 2: Perforación", "ETAPA 3: Ademe y Desarrollo", "ETAPA 4: Aforo y Calidad", "ETAPA 5: Equipamiento"],
    "Descripción": [
        "Geofísica, MIA, Permisos", 
        f"Perforación a {profundidad}m + Registro", 
        f"Ademe ({'Inoxidable' if calidad_agua != 'Agua Dulce' else 'Acero Carbón'}) + Grava", 
        "Prueba de bombeo escalonada", 
        f"Bomba {hp} HP + Tren de descarga + Eléctrico"
    ],
    "Costo (MXN)": [costo_e1, costo_e2, costo_e3, costo_e4, costo_e5]
}

df = pd.DataFrame(datos_presupuesto)
st.dataframe(
    df.style.format({"Costo (MXN)": "${:,.2f}"}),
    hide_index=True, 
    use_container_width=True
)

st.markdown(f"### Inversión Total Estimada: **${total:,.2f} MXN**")
