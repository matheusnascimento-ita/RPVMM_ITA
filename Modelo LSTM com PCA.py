# Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import time
import warnings
warnings.filterwarnings("ignore")

# =====================================================================

# 1. Definição de parâmetros e tratamento dos dados

EXCEL_PATH = "data\Desafio 2 - Tiragem - Base de Dados.xlsx"
SHEET_NAME = "Base por Coleção"
DATE_COL = "Data"
TARGET_COL = "Quantidade_Total"
AGG_FREQ = "D"
N_SPLITS = 5                    # Agregação diária
EPOCHS = 80
BATCH_SIZE = 18
PATIENCE = 8

lookback_list = [3, 6, 12, 18]
neurons_list = [32, 64, 96, 128]
dropout_rate = 0.3
FORECAST_HORIZON = 15           # Dias à frente

# Carregar e agregar base de dados
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.dropna(subset=[DATE_COL, TARGET_COL])

df['Quantidade_Total'] = df['Quantidade_Total'].rolling(window=3, center=False).mean()
df.dropna(inplace=True)

df = df[(df['Quantidade_Total'] < df['Quantidade_Total'].quantile(0.99))]               # Remoção de outliers

# Agregar total diário
df_diario = df.groupby(pd.Grouper(key=DATE_COL, freq=AGG_FREQ))[TARGET_COL].sum().reset_index()
df_diario = df_diario.sort_values(DATE_COL).reset_index(drop=True)

dates = df_diario[DATE_COL]
series = df_diario[[TARGET_COL]].copy()

# =====================================================================

# 2. Aplicação de PCA - captura de PC1 e PC2

# Criar lags para enriquecer a dimensionalidade
for lag in range(1, 4):
    series[f"lag_{lag}"] = series[TARGET_COL].shift(lag)

series = series.dropna().reset_index(drop=True)
dates = dates.iloc[3:].reset_index(drop=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

pca = PCA(n_components=2)
pcs = pca.fit_transform(scaled)
pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])

# Funções auxiliares
def create_sequences_pca(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])                        # Usar PC1 como variável alvo
    return np.array(X), np.array(y)

# =====================================================================

# 3. Grid-Search e treinamento do LSTM

results = []
param_grid = list(itertools.product(lookback_list, neurons_list))

for lookback, neurons in param_grid:
    X, y = create_sequences_pca(pca_df.values, lookback)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    maes, rmses, mapes, times = [], [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = Sequential([
            LSTM(neurons, input_shape=(lookback, 2)),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
        
        t0 = time.time()
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
        t1 = time.time()
        times.append(t1 - t0)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-6, y_test))) * 100
        
        maes.append(mae); rmses.append(rmse); mapes.append(mape)
    
    results.append({
        "lookback": lookback,
        "neurons": neurons,
        "mean_MAE": np.mean(maes),
        "mean_RMSE": np.mean(rmses),
        "mean_MAPE": np.mean(mapes),
        "mean_time_s": np.mean(times)
    })
    print(f"Lookback={lookback}, neurônios={neurons} -> MAPE médio: {np.mean(mapes):.2f}%")

results_df = pd.DataFrame(results).sort_values("mean_MAPE")
display(results_df)

# =====================================================================

# 4. Treinamento do modelo com melhor configuração

best = results_df.iloc[0]
best_lookback = int(best["lookback"])
best_neurons = int(best["neurons"])

X_all, y_all = create_sequences_pca(pca_df.values, best_lookback)
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

model_final = Sequential([
    LSTM(best_neurons, input_shape=(best_lookback, 2)),
    Dropout(dropout_rate),
    Dense(1)
])
model_final.compile(optimizer='adam', loss='mse')

es_final = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
history = model_final.fit(X_train, y_train, validation_data=(X_test, y_test),
                          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es_final], verbose=0)

# =====================================================================

# 5. Métricas do modelo e gráficos das previsões

y_pred = model_final.predict(X_test)
mae_f = mean_absolute_error(y_test, y_pred)
rmse_f = np.sqrt(mean_squared_error(y_test, y_pred))
mape_f = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-6, y_test))) * 100

print("\nMÉTRICAS FINAIS")
print(f"MAE: {mae_f:.4f}")
print(f"RMSE: {rmse_f:.4f}")
print(f"MAPE: {mape_f:.2f}%\n")

# Plot erro treino vs erro validacao
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="Erro de Treino")
plt.plot(history.history["val_loss"], label="Erro de validação")
plt.title("Evolução dos erros de treino e validação")
plt.xlabel("Épocas")
plt.legend()
plt.grid(True)
plt.show()

# Plot previsões vs reais
test_dates = dates[best_lookback + split_idx:]
plt.figure(figsize=(12,5))
plt.plot(dates, pca_df["PC1"], label="Histórico (PC1)", color="black", alpha=0.4)
plt.plot(test_dates, y_test, label="Real (teste)", color="blue")
plt.plot(test_dates, y_pred, label="Previsão LSTM", color="orange", linestyle="--")
plt.title(f"LSTM (lookback={best_lookback}, neurônios={best_neurons}) - PC1")
plt.xlabel("Data")
plt.ylabel("PC1")
plt.legend()
plt.grid(True)
plt.show()

# Plot previsões vs reais
test_dates = dates[best_lookback + split_idx:]
plt.figure(figsize=(12,5))
plt.plot(test_dates, y_test, label="Real (teste)", color="blue")
plt.plot(test_dates, y_pred, label="Previsão LSTM", color="orange", linestyle="--")
plt.title(f"LSTM análise (lookback={best_lookback}, neurônios={best_neurons}) - PC1 (proxy de tiragens)")
plt.xlabel("Data")
plt.ylabel("PC1")
plt.legend()
plt.grid(True)
plt.show()

# Forecast de 15 dias
forecast_input = pca_df.values[-best_lookback:].copy()
future_preds = []

for _ in range(FORECAST_HORIZON):
    X_input = forecast_input[-best_lookback:].reshape((1, best_lookback, 2))
    next_pred = model_final.predict(X_input)[0][0]
    # Usar o último valor de PC2 como referência
    next_pc2 = forecast_input[-1, 1]
    next_input = np.array([[next_pred, next_pc2]])
    forecast_input = np.vstack((forecast_input, next_input))
    future_preds.append(next_pred)

# Criação de datas futuras
last_date = dates.iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D")

# Plot forecast
plt.figure(figsize=(12,5))
plt.plot(dates, pca_df["PC1"], label="Histórico (PC1)", color="blue")
plt.plot(future_dates, future_preds, label="Forecast 15 dias (PC1)", color="red", linestyle="--")
plt.title("Previsão de 15 dias futuros (PC1 como proxy da tiragem)")
plt.xlabel("Data")
plt.ylabel("PC1")
plt.legend()
plt.grid(True)
plt.show()