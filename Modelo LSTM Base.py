# BIBLIOTECAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================================

# 1. Setup de parâmetros e limpeza da base de dados

EXCEL_PATH = "data\Desafio 2 - Tiragem - Base de Dados.xlsx"
SHEET_NAME = "Base por Coleção"
DATE_COL = "Data"
TARGET_COL = "Quantidade_Total"
AGG_FREQ = "M"
N_SPLITS = 5                        #Número de splits para TimeSeriesSplit
EPOCHS = 80
BATCH_SIZE = 18
PATIENCE = 8

# Grid (leve) para testar
lookback_list = [3, 6, 12, 18] # janelas de meses
neurons_list = [32, 64, 96, 128] #unidades LSTM a testar no grid
dropout_rate = 0.3

# Carregar dados e agrupar por mês
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.dropna(subset=[DATE_COL, TARGET_COL])

# Remoção de duplicatas
df = df.drop_duplicates(subset=[DATE_COL, TARGET_COL], keep='last')

# Remoção de datas inválidas
df = df.dropna(subset=[DATE_COL])
df = df[df[DATE_COL].notnull()]
df = df.sort_values(DATE_COL)

# Garantir que a coluna alvo, das tiragens, é numérica
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])

# Remoção de outliers
q1 = df[TARGET_COL].quantile(0.25)
q3 = df[TARGET_COL].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Ordenar série cronologicamente
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Fazer a soma das tiragens para cada mês
df_mensal = df.groupby(pd.Grouper(key=DATE_COL, freq=AGG_FREQ))[TARGET_COL].sum().reset_index()
df_mensal = df_mensal.sort_values(DATE_COL).reset_index(drop=True)

# Armazenar séries
dates = df_mensal[DATE_COL].copy()
series = df_mensal[TARGET_COL].values.reshape(-1, 1)

# Normalizar séries
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

# =====================================================================

# 2. Criação de janelas de tempo na série histórica (TimeSeriesSplit)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# =====================================================================

# 3. Grid Search e treinamento da LSTM

results = []  # Armazenar métricas de cada configuração

param_grid = list(itertools.product(lookback_list, neurons_list))

for lookback, neurons in param_grid:
    # Criar sequências para este lookback
    X, y = create_sequences(series_scaled, lookback)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    fold = 0
    maes, rmses, mapes = [], [], []
    times = []
    
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Construir modelo
        model = Sequential([
            LSTM(neurons, input_shape=(lookback, 1)),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Early stopping
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
        
        # Treinar
        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )
        t1 = time.time()
        times.append(t1 - t0)
        
        # Prever no fold de teste e desnormalizar
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Métricas
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-6, y_true))) * 100
        
        maes.append(mae); rmses.append(rmse); mapes.append(mape)
    
    # Consolidar média e desvio
    results.append({
        "lookback": lookback,
        "neurons": neurons,
        "dropout": dropout_rate,
        "mean_MAE": np.mean(maes),
        "std_MAE": np.std(maes),
        "mean_RMSE": np.mean(rmses),
        "std_RMSE": np.std(rmses),
        "mean_MAPE": np.mean(mapes),
        "std_MAPE": np.std(mapes),
        "mean_time_s": np.mean(times),
        "n_splits": N_SPLITS
    })
    print(f"Grid feito: lookback={lookback}, neurônios={neurons} -> MAPE (mean) = {np.mean(mapes):.2f}%")

# Resultados
results_df = pd.DataFrame(results).sort_values("mean_MAPE")
display(results_df)

# =====================================================================

# 4. Treinamento do melhor modelo

best = results_df.iloc[0]
best_lookback = int(best['lookback'])
best_neurons = int(best['neurons'])

# Criar sequências com melhor lookback
X_all, y_all = create_sequences(series_scaled, best_lookback)

# Usar 80% final como treino/teste para plot final (ou treinar até penúltimo fold)
split_idx = int(len(X_all) * 0.7)
X_train_final, X_test_final = X_all[:split_idx], X_all[split_idx:]
y_train_final, y_test_final = y_all[:split_idx], y_all[split_idx:]

# Modelo final
model_final = Sequential([
    LSTM(best_neurons, input_shape=(best_lookback, 1)),
    Dropout(dropout_rate),
    Dense(1)
])
model_final.compile(optimizer='adam', loss='mse')

es_final = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)
model_final.fit(X_train_final, y_train_final, validation_data=(X_test_final, y_test_final),
                epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es_final], verbose=0)

# =====================================================================

# 5. Validação do modelo e gráfico de sua previsão

# Coleção de previsões e desnormalização destas
y_pred_scaled_final = model_final.predict(X_test_final)
y_pred_final = scaler.inverse_transform(y_pred_scaled_final.reshape(-1, 1)).flatten()
y_true_final = scaler.inverse_transform(y_test_final.reshape(-1, 1)).flatten()

# Métricas finais
mae_f = mean_absolute_error(y_true_final, y_pred_final)
rmse_f = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
mape_f = np.mean(np.abs((y_true_final - y_pred_final) / np.where(y_true_final==0, 1e-6, y_true_final))) * 100

print("\n\nSumário do melhor modelo final:")
print(best.to_dict())
print(f"Teste final:\tMAE: {mae_f:.2f}\t|\tRMSE: {rmse_f:.2f}\t|\tMAPE: {mape_f:.2f}%\n\n")

# Plot: série real vs previsões do melhor modelo
start_idx = best_lookback + split_idx
test_dates = dates[start_idx : start_idx + len(y_true_final)].reset_index(drop=True)

plt.figure(figsize=(12,5))
plt.plot(dates, series.flatten(), label='Série Histórica', color='black', alpha=0.4)
plt.plot(test_dates, y_true_final, label='Real', color='blue')
plt.plot(test_dates, y_pred_final, label='Previsão - LSTM', color='orange', linestyle='--')
plt.title(f"LSTM - best by Grid Search (lookback={best_lookback}, neurons={best_neurons})")
plt.xlabel("Data")
plt.ylabel("Quantidade Total")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================================