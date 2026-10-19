# Previsão de ETA - Porto de Nacala

Protótipo local para estimar, em horas, o tempo até à chegada de navios ao Porto de Nacala com dados AIS e um modelo Random Forest. O treino, o modelo e a interface funcionam no computador local, sem servidores externos nem base de dados.

## Requisitos

- Python 3.9 ou superior
- O ficheiro `nacala_eta_base_de_dados.xlsx` na pasta do projeto, com a folha `Dados_AIS`

## Instalação e execução

Abra um terminal na pasta do projeto e instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Treine o modelo e inicie a aplicação:

```bash
python train.py
streamlit run app.py
```

O treino apresenta MAE e R² no terminal e guarda `modelo_eta_nacala.pkl` na pasta do projeto. O Streamlit abre a interface no navegador; por omissão, o endereço local é `http://localhost:8501`.

## Dados utilizados

O treino lê a folha `Dados_AIS` do Excel local. Utiliza `dist_to_nacala_nm`, `sog_kn`, `cog_deg`, `heading_deg`, `dist_travelled_nm` e `total_route_nm` para prever `hours_to_arrival`. Registos sem valores válidos nestas colunas são excluídos.

## Estrutura

- `train.py`: carrega, avalia e treina o modelo final com os dados válidos.
- `app.py`: interface local para introduzir dados AIS, consultar a previsão e ver a importância das variáveis.
- `modelo_eta_nacala.pkl`: modelo criado por `train.py`.
- `nacala_eta_base_de_dados.xlsx`: dados locais de treino.