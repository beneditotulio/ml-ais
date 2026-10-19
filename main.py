import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Carregar os dados da folha AIS
df = pd.read_excel('nacala_eta_base_de_dados.xlsx', sheet_name='Dados_AIS')

# 2. Definir variáveis preditoras (apenas AIS e rotas) e o alvo
features = [
    'dist_to_nacala_nm', 'sog_kn', 'cog_deg', 'heading_deg', 
    'dist_travelled_nm', 'total_route_nm'
]
target = 'hours_to_arrival'

df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

# 3. Dividir em dados de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar o Modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Avaliar o Modelo
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} horas")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")