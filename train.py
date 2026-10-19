from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "nacala_eta_base_de_dados.xlsx"
MODEL_FILE = BASE_DIR / "modelo_eta_nacala.pkl"
SHEET_NAME = "Dados_AIS"
FEATURES = [
    "dist_to_nacala_nm",
    "sog_kn",
    "cog_deg",
    "heading_deg",
    "dist_travelled_nm",
    "total_route_nm",
]
TARGET = "hours_to_arrival"


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Ficheiro de dados não encontrado: {DATA_FILE}")

    data = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Colunas em falta na folha {SHEET_NAME}: {missing_columns}")

    clean_data = data[required_columns].apply(pd.to_numeric, errors="coerce")
    clean_data = clean_data.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(clean_data) < 5:
        raise ValueError("São necessários pelo menos 5 registos válidos para treinar e avaliar o modelo.")

    X = clean_data[FEATURES]
    y = clean_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    evaluation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    print(f"Registos válidos: {len(clean_data)}")
    print(f"Erro Absoluto Médio (MAE): {mean_absolute_error(y_test, predictions):.2f} horas")
    print(f"R²: {r2_score(y_test, predictions):.4f}")

    # Depois da avaliação, treina o artefacto final com todos os registos válidos.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
    print(f"Modelo guardado em: {MODEL_FILE}")


if __name__ == "__main__":
    main()