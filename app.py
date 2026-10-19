from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modelo_eta_nacala.pkl"

st.set_page_config(page_title="ETA Porto de Nacala", page_icon="⚓", layout="centered")
st.title("Porto de Nacala - Previsão de ETA (AIS & ML)")
st.write("Introduza os dados AIS e de navegação para estimar o tempo restante até à chegada.")

if not MODEL_FILE.is_file():
    st.error("Modelo não encontrado. Execute `python train.py` antes de iniciar a aplicação.")
    st.stop()

artifact = joblib.load(MODEL_FILE)
model = artifact["model"]
features = artifact["features"]

with st.form("ais_input"):
    st.subheader("Dados do navio")
    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("Distância restante até Nacala (nm)", min_value=0.0, value=100.0, step=1.0)
        speed = st.number_input("Velocidade atual (SOG, nós)", min_value=0.0, value=12.0, step=0.5)
        course = st.number_input("Rumo (COG, graus)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
    with col2:
        heading = st.number_input("Proa (Heading, graus)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
        travelled = st.number_input("Distância percorrida (nm)", min_value=0.0, value=500.0, step=1.0)
        total_route = st.number_input("Distância total da rota (nm)", min_value=0.0, value=600.0, step=1.0)

    submitted = st.form_submit_button("Calcular ETA", type="primary")

if submitted:
    values = {
        "dist_to_nacala_nm": distance,
        "sog_kn": speed,
        "cog_deg": course,
        "heading_deg": heading,
        "dist_travelled_nm": travelled,
        "total_route_nm": total_route,
    }
    input_data = pd.DataFrame([[values[name] for name in features]], columns=features)
    estimated_hours = max(0.0, float(model.predict(input_data)[0]))
    arrival_time = datetime.now() + timedelta(hours=estimated_hours)

    st.subheader("Previsão")
    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Horas restantes", f"{estimated_hours:.2f} h")
    result_col2.metric("Chegada prevista", arrival_time.strftime("%d/%m/%Y %H:%M"))

st.divider()
st.subheader("Diagnóstico do modelo")
importance = pd.DataFrame(
    {"Variável": features, "Importância": model.feature_importances_}
).sort_values("Importância", ascending=False)
st.bar_chart(importance.set_index("Variável"))
st.dataframe(importance, hide_index=True, use_container_width=True)