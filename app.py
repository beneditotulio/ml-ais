from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modelo_eta_nacala.pkl"
DATA_FILE = BASE_DIR / "nacala_eta_base_de_dados.xlsx"
FEATURES = [
    "dist_to_nacala_nm",
    "sog_kn",
    "cog_deg",
    "heading_deg",
    "dist_travelled_nm",
    "total_route_nm",
]


@st.cache_data
def load_latest_vessel_records() -> pd.DataFrame:
    ais = pd.read_excel(DATA_FILE, sheet_name="Dados_AIS")
    vessels = pd.read_excel(DATA_FILE, sheet_name="Navios")
    ais["timestamp_utc"] = pd.to_datetime(ais["timestamp_utc"], errors="coerce")
    ais[FEATURES] = ais[FEATURES].apply(pd.to_numeric, errors="coerce")
    ais = ais.dropna(subset=["mmsi", "timestamp_utc", *FEATURES])
    latest = ais.sort_values("timestamp_utc").groupby("mmsi", as_index=False).tail(1)
    registry_columns = ["mmsi", "carrier", "flag", "loa_m", "capacity_teu"]
    latest = latest.merge(vessels[registry_columns], on="mmsi", how="left", suffixes=("", "_registry"))
    latest["vessel_label"] = latest.apply(
        lambda row: f"{row['ship_name']} · MMSI {int(row['mmsi'])}", axis=1
    )
    return latest.sort_values("ship_name").reset_index(drop=True)


def apply_selected_vessel(record: pd.Series) -> None:
    for feature in FEATURES:
        st.session_state[f"input_{feature}"] = float(record[feature])


def set_vessel_inputs(label: str, records: pd.DataFrame) -> None:
    record = records.loc[records["vessel_label"] == label].iloc[0]
    apply_selected_vessel(record)
    st.session_state.pop("eta_result", None)


def select_vessel(records: pd.DataFrame) -> None:
    set_vessel_inputs(st.session_state["selected_vessel"], records)


def move_vessel(step: int, records: pd.DataFrame) -> None:
    labels = records["vessel_label"].tolist()
    current = st.session_state.get("selected_vessel", labels[0])
    new_index = (labels.index(current) + step) % len(labels)
    st.session_state["selected_vessel"] = labels[new_index]
    set_vessel_inputs(labels[new_index], records)

st.set_page_config(page_title="ETA Porto de Nacala", page_icon="⚓", layout="centered")
st.title("Porto de Nacala - Previsão de ETA (AIS & ML)")
st.write("Selecione um navio para pré-visualizar o último registo AIS e estimar o tempo até à chegada.")

if not MODEL_FILE.is_file():
    st.error("Modelo não encontrado. Execute `python train.py` antes de iniciar a aplicação.")
    st.stop()
if not DATA_FILE.is_file():
    st.error("Ficheiro AIS não encontrado. Coloque `nacala_eta_base_de_dados.xlsx` na pasta do projeto.")
    st.stop()

artifact = joblib.load(MODEL_FILE)
model = artifact["model"]
features = artifact["features"]
vessel_records = load_latest_vessel_records()
if vessel_records.empty:
    st.error("Não foram encontrados registos AIS completos para pré-visualizar.")
    st.stop()

vessel_labels = vessel_records["vessel_label"].tolist()
if "selected_vessel" not in st.session_state or st.session_state["selected_vessel"] not in vessel_labels:
    st.session_state["selected_vessel"] = vessel_labels[0]
selected_record = vessel_records.loc[
    vessel_records["vessel_label"] == st.session_state["selected_vessel"]
].iloc[0]

for feature in FEATURES:
    key = f"input_{feature}"
    if key not in st.session_state:
        st.session_state[key] = float(selected_record[feature])

st.subheader("Navio selecionado")
selector_col, previous_col, next_col = st.columns([6, 1, 1], vertical_alignment="bottom")
with selector_col:
    st.selectbox(
        "Escolher navio",
        options=vessel_labels,
        key="selected_vessel",
        label_visibility="collapsed",
        on_change=select_vessel,
        args=(vessel_records,),
    )
with previous_col:
    st.button("←", help="Navio anterior", on_click=move_vessel, args=(-1, vessel_records), use_container_width=True)
with next_col:
    st.button("→", help="Navio seguinte", on_click=move_vessel, args=(1, vessel_records), use_container_width=True)

preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)
preview_col1.metric("Velocidade SOG", f"{selected_record['sog_kn']:.1f} kn")
preview_col2.metric("Distância a Nacala", f"{selected_record['dist_to_nacala_nm']:.1f} nm")
preview_col3.metric("Estado AIS", str(selected_record.get("nav_status", "Não indicado")))
if pd.notna(selected_record.get("latitude")) and pd.notna(selected_record.get("longitude")):
    preview_col4.metric(
        "Posição AIS",
        f"{selected_record['latitude']:.4f}, {selected_record['longitude']:.4f}",
    )

details = [f"Último registo no ficheiro: {selected_record['timestamp_utc']:%d/%m/%Y %H:%M} UTC"]
for column, label in [("origin_port", "Origem"), ("carrier", "Operador"), ("flag", "Bandeira")]:
    if pd.notna(selected_record.get(column)):
        details.append(f"{label}: {selected_record[column]}")
details[0] = details[0].replace("Último registo no ficheiro", "Último registo AIS completo")
st.caption(" · ".join(details) + ". Dados de demonstração locais, não AIS em tempo real.")

with st.form("ais_input"):
    st.subheader("Dados para a previsão (editáveis)")
    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("Distância restante até Nacala (nm)", min_value=0.0, step=1.0, key="input_dist_to_nacala_nm")
        speed = st.number_input("Velocidade atual (SOG, nós)", min_value=0.0, step=0.5, key="input_sog_kn")
        course = st.number_input("Rumo (COG, graus)", min_value=0.0, max_value=360.0, step=1.0, key="input_cog_deg")
    with col2:
        heading = st.number_input("Proa (Heading, graus)", min_value=0.0, max_value=360.0, step=1.0, key="input_heading_deg")
        travelled = st.number_input("Distância percorrida (nm)", min_value=0.0, step=1.0, key="input_dist_travelled_nm")
        total_route = st.number_input("Distância total da rota (nm)", min_value=0.0, step=1.0, key="input_total_route_nm")

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
    st.session_state["eta_result"] = {
        "vessel": st.session_state["selected_vessel"],
        "hours": estimated_hours,
        "arrival": arrival_time,
    }

eta_result = st.session_state.get("eta_result")
if eta_result and eta_result["vessel"] == st.session_state["selected_vessel"]:
    st.subheader("Previsão")
    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Horas restantes", f"{eta_result['hours']:.2f} h")
    result_col2.metric("Chegada prevista", eta_result["arrival"].strftime("%d/%m/%Y %H:%M"))

st.divider()
st.subheader("Diagnóstico do modelo")
importance = pd.DataFrame(
    {"Variável": features, "Importância": model.feature_importances_}
).sort_values("Importância", ascending=False)
st.bar_chart(importance.set_index("Variável"))
st.dataframe(importance, hide_index=True, use_container_width=True)