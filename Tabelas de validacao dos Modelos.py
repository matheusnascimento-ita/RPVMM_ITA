# Bibliotecas

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings

# Suprimir avisos para manter a saída limpa
warnings.filterwarnings("ignore")

# =====================================================================
# Configurações

EXCEL_PATH = "Codigos\\data\\Desafio 2 - Tiragem - Base de Dados.xlsx"
SHEET_NAME = "Base por Coleção"
DATE_COL = "Data"
TARGET_COL = "Quantidade_Total"
PRODUCT_COL = "Nome_Produto" 

# Definição temporal conforme explicado no relatório
# Treino: Jan/2021 a Jun/2024
# Teste: Jul/2024 em diante
TRAIN_START = '2021-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'

# =====================================================================

def load_and_preprocess(file_path, sheet_name):
    """
    Carrega os dados e limpa.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    
    df = df[df[DATE_COL].dt.year >= 2021]
    df = df.dropna(subset=[DATE_COL, TARGET_COL])
    
    return df

# =====================================================================

def aggregate_monthly(df):
    """
    Realiza a agregação mensal da Quantidade Total.
    """
    df_temp = df.copy()
    
    # Agrupamento mensal
    df_mensal = df_temp.groupby(pd.Grouper(key=DATE_COL, freq='MS'))[TARGET_COL].sum().reset_index()
    df_mensal = df_mensal.sort_values(DATE_COL).reset_index(drop=True)

    return df_mensal

# =====================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calcula MAE, RMSE e MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE com proteção contra divisão por zero
    # Se o valor real for 0, usa 1 para evitar erro
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    
    return mae, rmse, mape

# =====================================================================

def calculate_advanced_metrics(y_true, y_pred, dates):
    """
    Calcula MAPE, Acerto Temporal e Erro de Amplitude.
    """
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    
    # DataFrame auxiliar para análise de picos
    df_comp = pd.DataFrame({'Data': dates, 'Real': y_true, 'Pred': y_pred})
    
    # Se não houver dados, acerto temporal e erro de amplitude não podem ser definidos
    if df_comp['Real'].max() == 0:
        return mape, "Indefinido", "Indefinido"

    # 2. Acerto Temporal (Comparação de Picos)
    # Encontra a data onde ocorreu o valor máximo
    peak_real_idx = df_comp['Real'].idxmax()
    peak_pred_idx = df_comp['Pred'].idxmax()
    
    date_real = df_comp.loc[peak_real_idx, 'Data']
    date_pred = df_comp.loc[peak_pred_idx, 'Data']
    
    # Diferença em meses absolutos
    diff_months = abs((date_real.year - date_pred.year) * 12 + (date_real.month - date_pred.month))
    
    if diff_months == 0:
        acerto_temporal = "Alto (0 meses)"
    elif diff_months == 1:
        acerto_temporal = "Médio (1 mês)"
    else:
        acerto_temporal = f"Baixo ({diff_months} meses)"

    # 3. Erro de Amplitude (Diferença de magnitude no pico)
    peak_real_val = df_comp.loc[peak_real_idx, 'Real']
    peak_pred_val = df_comp.loc[peak_pred_idx, 'Pred']
    
    amp_error_pct = abs(peak_pred_val - peak_real_val) / peak_real_val
    
    if amp_error_pct < 0.20:
        erro_amplitude = f"Baixo ({amp_error_pct:.1%})"
    elif amp_error_pct < 0.50:
        erro_amplitude = f"Médio ({amp_error_pct:.1%})"
    else:
        erro_amplitude = f"Alto ({amp_error_pct:.1%})"
        
    return mape, acerto_temporal, erro_amplitude

# =====================================================================

def run_table_4_baselines(df_full):
    """
    Calcula as métricas exigidas para Média histórica e Naive sazonal
    """
    print(f"\n{'='*20} TABELA 4: MÉDIA E NAIVE {'='*20}")
    
    # Agregação global (todas as coleções juntas)
    df_agg = aggregate_monthly(df_full)
    
    # Split Treino/Teste
    train = df_agg[(df_agg[DATE_COL] >= TRAIN_START) & (df_agg[DATE_COL] <= TRAIN_END)]
    test = df_agg[df_agg[DATE_COL] >= TEST_START]
    
    if len(test) == 0:
        print("Aviso: Nenhum dado encontrado para o período de teste.")
        return

    y_train = train[TARGET_COL].values
    y_test = test[TARGET_COL].values
    
    results = []

    # Média Histórica
    mean_val = np.mean(y_train)
    pred_mean = np.full(len(y_test), mean_val)
    mae_mean, rmse_mean, mape_mean = calculate_metrics(y_test, pred_mean)
    
    results.append({
        "Modelo": "Média Histórica",
        "MAPE (%)": f"{mape_mean:.2f}",
        "RMSE": f"{rmse_mean:.2f}",
        "MAE": f"{mae_mean:.2f}"
    })

    # Naive Sazonal
    history = list(y_train)
    preds_naive = []
    
    for i in range(len(y_test)):
        # Pega valor de 12 meses atrás
        lag_idx = len(history) - 12
        if lag_idx >= 0:
            pred = history[lag_idx]
        else:
            pred = np.mean(history) # Fallback
            
        preds_naive.append(pred)
        history.append(y_test[i])
        
    mae_naive, rmse_naive, mape_naive = calculate_metrics(y_test, np.array(preds_naive))
    
    results.append({
        "Modelo": "Naive Sazonal",
        "MAPE (%)": f"{mape_naive:.2f}",
        "RMSE": f"{rmse_naive:.2f}",
        "MAE": f"{mae_naive:.2f}"
    })
    
    # Exibir Tabela
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

# =====================================================================

def calculate_ef_metrics(df_agg, train_end_date):
    """
    Calcula métricas específicas exigidas para EF.
    """
    # Histórico (Meses com venda > 0)
    months_with_sales = len(df_agg[df_agg[TARGET_COL] > 0])
    total_months = len(df_agg)
    
    # Intermitência (% de meses zerados)
    if total_months == 0:
        intermitencia = "Indefinida"
    else:
        zeros = len(df_agg[df_agg[TARGET_COL] == 0])
        pct_zeros = zeros / total_months
        
        # Classificação
        if pct_zeros > 0.50:
            intermitencia = f"Muito Alta ({pct_zeros:.0%})"
        elif pct_zeros > 0.20:
            intermitencia = f"Alta ({pct_zeros:.0%})"
        else:
            intermitencia = f"Baixa/Média ({pct_zeros:.0%})"

    # Tendência (Slope da Regressão Linear)
    if total_months > 1:
        X = np.arange(total_months).reshape(-1, 1)
        y = df_agg[TARGET_COL].values
        
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        mean_vol = np.mean(y)
        
        # Normaliza slope para definir relevância
        norm_slope = slope / mean_vol if mean_vol > 0 else 0
        
        if norm_slope > 0.01:
            tendencia = "Crescimento"
        elif norm_slope < -0.01:
            tendencia = "Decrescimento"
        else:
            tendencia = "Variável"
    else:
        tendencia = "Indefinida"

    # MAPE
    train = df_agg[df_agg[DATE_COL] <= train_end_date]
    test = df_agg[df_agg[DATE_COL] > train_end_date]
    
    if len(test) > 0 and len(train) >= 12:
        y_test = test[TARGET_COL].values
        history = list(train[TARGET_COL].values)
        preds = []
        for i in range(len(y_test)):
            lag_idx = len(history) - 12
            val = history[lag_idx] if lag_idx >= 0 else np.mean(history)
            preds.append(val)
            history.append(y_test[i])
            
        y_pred = np.array(preds)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1))) * 100
        mape_str = f"{mape:.2f}"
    else:
        mape_str = "Insuf. Dados"

    return mape_str, intermitencia, months_with_sales, tendencia

# =====================================================================

def generate_table_5(df_full):
    """
    Gera as métricas específicas exigidas para as coleções do Segmento EM/PV
    """
    print(f"\n{'='*20} GERANDO TABELA 5: Segmento EM/PV {'='*20}")
    
    collections_em_pv = ["Hexa", "Lumen"]
    results = []
    
    # Tenta identificar coluna de produto
    col_prod = PRODUCT_COL
    if col_prod not in df_full.columns:
        possible = [c for c in df_full.columns if "prod" in c.lower() or "nom" in c.lower()]
        if possible: col_prod = possible[0]
    
    for col_name in collections_em_pv:
        # Filtra coleção
        df_col = df_full[df_full[col_prod].str.contains(col_name, case=False, na=False)]
        df_agg = aggregate_monthly(df_col)
        
        if len(df_agg) == 0:
            print(f"Aviso: Sem dados para {col_name}")
            continue
            
        # Split Treino/Teste
        train = df_agg[(df_agg[DATE_COL] >= TRAIN_START) & (df_agg[DATE_COL] <= TRAIN_END)]
        test = df_agg[df_agg[DATE_COL] >= TEST_START]
        
        # Volume Médio
        vol_medio = df_agg[TARGET_COL].mean()
        
        if len(test) > 0 and len(train) >= 12:
            y_test = test[TARGET_COL].values
            test_dates = test[DATE_COL].values
            
            # Gerar Previsão
            history = list(train[TARGET_COL].values)
            preds = []
            for i in range(len(y_test)):
                preds.append(history[len(history)-12])
                history.append(y_test[i])
            y_pred = np.array(preds)
            
            # Calcular Métricas
            mape, acerto_temp, erro_amp = calculate_advanced_metrics(y_test, y_pred, test_dates)
            
            results.append({
                "Coleção": col_name,
                "MAPE (%)": f"{mape:.2f}",
                "Acerto Temporal": acerto_temp,
                "Erro de Amplitude": erro_amp,
                "Volume Médio": f"{vol_medio:.2f}"
            })
        else:
            results.append({
                "Coleção": col_name, 
                "MAPE (%)": "N/A", 
                "Acerto Temporal": "Insuf. Dados", 
                "Erro de Amplitude": "Insuf. Dados",
                "Volume Médio": f"{vol_medio:.2f}"
            })
            
    # Exibir tabela
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

# =====================================================================

def generate_table_6(df_full):
    """
    Gera as métricas específicas exigidas para as coleções do Segmento EF
    """
    print(f"\n{'='*20} GERANDO TABELA 6: Segmento EF (Phases/Callis) {'='*20}")
    
    collections_ef = ["Phases", "Callis"]
    results = []
    
    # Identifica coluna de produto
    col_prod = PRODUCT_COL
    if col_prod not in df_full.columns:
        possible = [c for c in df_full.columns if "prod" in c.lower() or "nom" in c.lower()]
        if possible: col_prod = possible[0]
    
    for col_name in collections_ef:
        # Filtra coleção
        df_col = df_full[df_full[col_prod].str.contains(col_name, case=False, na=False)]
        
        if not df_col.empty:
            # Garante range de datas completo para detectar zeros corretamente
            min_date = df_col[DATE_COL].min()
            max_date = df_col[DATE_COL].max()
            full_range = pd.date_range(start=min_date, end=max_date, freq='MS')
            
            df_agg = aggregate_monthly(df_col)
            # Reindexa preenchendo faltantes com 0
            df_agg = df_agg.set_index(DATE_COL).reindex(full_range, fill_value=0).reset_index()
            df_agg.rename(columns={'index': DATE_COL}, inplace=True)
            
            mape, intermitencia, historico, tendencia = calculate_ef_metrics(df_agg, TRAIN_END)
            
            results.append({
                "Coleção": col_name,
                "MAPE (%)": mape,
                "Intermitência": intermitencia,
                "Histórico (meses)": historico,
                "Tendência": tendencia
            })
        else:
            print(f"Aviso: Dados não encontrados para coleção '{col_name}'")

    # Exibe Tabela
    if results:
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))
    else:
        print("Nenhum resultado gerado.")

# =====================================================================

if __name__ == "__main__":
    try:
        print("Carregando arquivo Excel...\n\n")
        df_raw = load_and_preprocess(EXCEL_PATH, SHEET_NAME)
        
        run_table_4_baselines(df_raw)
        generate_table_5(df_raw)
        generate_table_6(df_raw)
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{EXCEL_PATH}' não encontrado. Verifique se o caminho dado está correto.")
    except ValueError as e:
        print(f"ERRO DE LEITURA: {e}. Verifique se a aba '{SHEET_NAME}' existe no Excel.")
    except Exception as e:
        print(f"ERRO INESPERADO: {e}")