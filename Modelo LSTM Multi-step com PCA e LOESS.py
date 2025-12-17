# Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Usar CPU se tiver problemas com GPU ou limitar memória utilizada da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# =====================================================================

# 1. Definição de parâmetros e carregamento dos dados

class Config:
    # Parâmetros do dataset
    EXCEL_PATH = "./data/Desafio 2 - Tiragem - Base de Dados.xlsx"
    SHEET_NAME = "Base por Coleção"
    DATE_COL = "Data"
    TARGET_COL = "Quantidade_Total"
    AGG_FREQ = "M"  # MENSAL
    
    # LOESS
    LOESS_FRAC = 0.25
    LOESS_IT = 3
    
    # PCA
    PCA_COMPONENTS = 4
    
    FORECAST_HORIZON = 2   # 3 meses, melhor precisão
    N_SPLITS = 2           # Poucos dados, poucos splits
    EPOCHS = 100
    BATCH_SIZE = 4
    PATIENCE = 15
    
    # Grid Search - Escala mensal
    LOOKBACK_LIST = [6, 12]  # Meses: 6m, 1a, 1.5a, 2a
    NEURONS_LIST = [32, 64]
    DROPOUT_RATES = [0.2]
    LEARNING_RATES = [0.001]
    
    def explain_temporal_settings(self):
        print("CONFIGURAÇÃO TEMPORAL MENSAL:")
        print(f"   Agregação: {self.AGG_FREQ}")
        print(f"   Forecast: {self.FORECAST_HORIZON} meses")
        print(f"   Lookbacks testados: {self.LOOKBACK_LIST} meses")

config = Config()

# =====================================================================

# 2. Agregar dados de vendas totais por mês e tratá-los

def aggregate_monthly_data(df, date_col, target_col):
    """
    Agrega dados diários para mensais com tratamento robusto
    """
    
    # Garantir que a data é datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Agregar por mês, com a soma total, número de dias e média diária
    df_monthly = df.groupby(pd.Grouper(key=date_col, freq='M'))[target_col].agg(['sum', 'count', 'mean']).reset_index()
    
    # Renomear colunas
    df_monthly = df_monthly.rename(columns={
        'sum': target_col, 
        'count': 'n_days', 
        'mean': 'daily_avg'
    })
    
    # Filtrar meses incompletos
    df_monthly = df_monthly[df_monthly['n_days'] >= 10]
    
    # Ordenar por data
    df_monthly = df_monthly.sort_values(date_col).reset_index(drop=True)
    
    print(f"   Dados originais: {len(df)} registros diários")
    print(f"   Dados agregados: {len(df_monthly)} meses completos")
    print(f"   Período: {df_monthly[date_col].min().strftime('%Y-%m')} a {df_monthly[date_col].max().strftime('%Y-%m')}")
    
    return df_monthly[[date_col, target_col]]

# =====================================================================

# 3. Aplicação de LOESS

class LoessPreprocessor:
    def __init__(self, frac=0.25, it=3):
        self.frac = frac
        self.it = it
        self.scaler = MinMaxScaler()
        
    def apply_loess(self, series, dates=None):
        """Aplica suavização LOESS na série temporal MENSAL"""
        if dates is None:
            x = np.arange(len(series))
        else:
            start_date = min(dates)
            x = np.array([(d.year - start_date.year) * 12 + (d.month - start_date.month) for d in dates])
        
        smoothed = lowess(series, x, frac=self.frac, it=self.it, return_sorted=False)
        residual = series - smoothed
        
        return smoothed, residual
    
    def create_enhanced_features(self, df, target_col, date_col, lags=12):
        """Cria features enriquecidas com LOESS para os dados"""
        df = df.copy()
        dates = df[date_col]
        series = df[target_col].values
        
        # Aplicar LOESS
        series_smooth, residual = self.apply_loess(series, dates)
        
        # Adicionar ao DataFrame
        df['original'] = series
        df['loess_smooth'] = series_smooth
        df['residual'] = residual
        
        # Features temporais
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Lags da série suavizada (até 24 meses = 2 anos)
        max_lags = min(24, len(df) // 3)  # Não exceder 1/3 dos dados
        for lag in range(1, max_lags + 1):
            df[f'smooth_lag_{lag}'] = df['loess_smooth'].shift(lag)
        
        # Lags sazonais (12, 24 meses)
        seasonal_lags = [12, 24]
        for lag in seasonal_lags:
            if lag < len(df):
                df[f'smooth_lag_sazonal_{lag}'] = df['loess_smooth'].shift(lag)
        
        # Médias móveis e volatilidade (escala mensal)
        windows = [3, 6, 12]  # 3, 6, 12 meses
        for window in windows:
            if window < len(df):
                df[f'smooth_rolling_mean_{window}'] = df['loess_smooth'].shift(1).rolling(window).mean()
                df[f'smooth_rolling_std_{window}'] = df['loess_smooth'].shift(1).rolling(window).std()
                df[f'smooth_rolling_min_{window}'] = df['loess_smooth'].shift(1).rolling(window).min()
                df[f'smooth_rolling_max_{window}'] = df['loess_smooth'].shift(1).rolling(window).max()
        
        # Features de diferença (sazonais)
        df['smooth_diff_1'] = df['loess_smooth'].diff(1)  # Diferença mensal
        df['smooth_diff_12'] = df['loess_smooth'].diff(12)  # Diferença anual
        
        # Crescimento anual e mensal
        df['yoy_growth'] = df['loess_smooth'].pct_change(12)  # Year-over-year
        df['mom_growth'] = df['loess_smooth'].pct_change(1)   # Month-over-month
        
        # Remover NaN (mais dados perdidos devido a lags maiores)
        df_features = df.dropna().reset_index(drop=True)
        
        loess_info = {
            'variance_explained': 1 - (np.var(residual) / np.var(series)),
            'residual_std': np.std(residual),
            'n_months': len(df_features),
            'total_months': len(df)
        }
        
        print(f"LOESS aplicado - {loess_info['n_months']}/{loess_info['total_months']} meses - Variância explicada: {loess_info['variance_explained']:.3f}")
        
        return df_features, loess_info

# =====================================================================

# 4. Aplicação do PCA - captura de PC1 e PC2

class PCAManager:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        
    def apply_pca(self, df_features):
        """Aplica PCA nas features após LOESS para dados MENSAlS"""
        # Identificar colunas de features
        exclude_cols = ['original', 'loess_smooth', 'residual', 'Data', 'Date', 'data', config.DATE_COL]
        self.feature_columns = [col for col in df_features.columns 
                               if col not in exclude_cols and not col.startswith('smooth_lag_')]
        
        # Incluir alguns lags importantes
        important_lags = ['smooth_lag_1', 'smooth_lag_12', 'smooth_lag_sazonal_12']
        for lag in important_lags:
            if lag in df_features.columns:
                self.feature_columns.append(lag)
        
        # Normalizar e aplicar PCA
        features_scaled = self.scaler.fit_transform(df_features[self.feature_columns])
        pca_components = self.pca.fit_transform(features_scaled)
        
        # Criar DataFrame com componentes
        pca_columns = [f'PC{i+1}' for i in range(pca_components.shape[1])]
        pca_df = pd.DataFrame(pca_components, columns=pca_columns)
        
        # Adicionar informações originais
        pca_df['original'] = df_features['original'].values
        pca_df['loess_smooth'] = df_features['loess_smooth'].values
        pca_df['residual'] = df_features['residual'].values
        pca_df[config.DATE_COL] = df_features[config.DATE_COL].values
        
        pca_info = {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'components': self.pca.components_,
            'feature_names': self.feature_columns
        }
        
        print(f"PCA aplicado - Variância total: {pca_info['cumulative_variance'][-1]:.3f}")
        print(f"Variância por componente: {[f'{v:.3f}' for v in pca_info['explained_variance_ratio']]}")
        
        return pca_df, pca_info
    
# =====================================================================

# 5. Processo de validação cruzada (Cross-Validation)

class TimeSeriesCrossValidator:
    def __init__(self, n_splits=4):
        self.n_splits = n_splits
        self.results = {}
        
    def nested_cross_validation(self, X, y, model_fn, param_grid, lookback, forecast_horizon):
        """Cross Validation aninhado para séries temporais MENSAlS"""
        print(f"Iniciando Nested CV com {self.n_splits} splits...")
        
        tscv_outer = TimeSeriesSplit(n_splits=self.n_splits)
        outer_fold_results = []
        
        fold = 0
        for train_idx, test_idx in tscv_outer.split(X):
            fold += 1
            print(f"\nFOLD EXTERNO {fold}/{self.n_splits}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # CV interno para tuning
            best_params, inner_scores = self._inner_cv(
                X_train_outer, y_train_outer, model_fn, param_grid, lookback, forecast_horizon
            )
            
            # Treinar modelo final
            best_model, train_time = self._train_final_model(
                X_train_outer, y_train_outer, model_fn, best_params, lookback, forecast_horizon
            )

            if best_model is None:
                print(f"   AVISO: pulando FOLD {fold} por falta de dados.")
                continue
            
            # Avaliação no fold externo
            metrics, y_true, y_pred = self._evaluate_fold(
                best_model, X_test_outer, y_test_outer, lookback, forecast_horizon
            )
            
            fold_result = {
                'fold': fold,
                'best_params': best_params,
                'train_time': train_time,
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            outer_fold_results.append(fold_result)
            print(f"Fold {fold} - MAPE: {metrics['overall_mape']:.2f}%")
        
        self.results['folds'] = outer_fold_results
        return self._summarize_results()
    
    def _inner_cv(self, X, y, model_fn, param_grid, lookback, forecast_horizon):
        """CV interno para tuning de hiperparâmetros"""
        tscv_inner = TimeSeriesSplit(n_splits=3)
        best_score = float('inf')
        best_params = None
        inner_scores = []
        
        param_combinations = self._create_param_combinations(param_grid)
        
        print(f"Testando {len(param_combinations)} combinações de parâmetros (Grid Search)")
        
        # VERIFICAÇÃO CRÍTICA: Tem dados suficientes?
        if len(X) < lookback + forecast_horizon + 1:
            print(f"   AVISO: Dados insuficientes. X.shape: {X.shape}, lookback: {lookback}")
            # Retorna parâmetros padrão
            return {'units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.001}, []
        
        valid_combinations = 0
        
        for params in param_combinations:
            fold_scores = []
            
            for train_idx, val_idx in tscv_inner.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                X_seq_train, y_seq_train = self._create_sequences_multi_step(
                    X_train, y_train, lookback, forecast_horizon
                )
                X_seq_val, y_seq_val = self._create_sequences_multi_step(
                    X_val, y_val, lookback, forecast_horizon
                )
                
                # Pular se não há sequências válidas
                if len(X_seq_train) == 0 or len(X_seq_val) == 0:
                    continue
                
                try:
                    model = model_fn(**params, lookback=lookback, 
                                n_features=X_seq_train.shape[2], 
                                forecast_horizon=forecast_horizon)
                    
                    history = model.fit(
                        X_seq_train, y_seq_train,
                        validation_data=(X_seq_val, y_seq_val),
                        epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        verbose=0,
                        callbacks=[EarlyStopping(patience=config.PATIENCE, restore_best_weights=True)]
                    )
                    
                    score = model.evaluate(X_seq_val, y_seq_val, verbose=0)[0]
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"      Erro nos parâmetros {params}: {e}")
                    continue
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                inner_scores.append({'params': params, 'score': mean_score})
                valid_combinations += 1
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params
        
        # VERIFICAÇÃO CRÍTICA: Se nenhum parâmetro funcionou
        if best_params is None:
            print("   AVISO: Nenhuma combinação de parâmetros funcionou. Usando padrão.")
            best_params = {'units': 32, 'dropout_rate': 0.2, 'learning_rate': 0.001}
        
        print(f"   Combinações válidas: {valid_combinations}/{len(param_combinations)}")

        for params in param_combinations:
            fold_scores = []
        
        for train_idx, val_idx in tscv_inner.split(X):
            # ... criação de sequências ...
            
            if len(X_seq_train) == 0 or len(X_seq_val) == 0:
                continue
            
            try:
                model = model_fn(**params, lookback=lookback, 
                               n_features=X_seq_train.shape[2], 
                               forecast_horizon=forecast_horizon)
                
                history = model.fit(
                    X_seq_train, y_seq_train,
                    validation_data=(X_seq_val, y_seq_val),
                    epochs=min(50, config.EPOCHS),  # 🔽 Reduzir épocas no CV interno
                    batch_size=config.BATCH_SIZE,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]  # 🔽 Paciência menor
                )
                
                # CORREÇÃO: Verificar se evaluation retorna resultados
                evaluation_result = model.evaluate(X_seq_val, y_seq_val, verbose=0)
                if isinstance(evaluation_result, list) and len(evaluation_result) > 0:
                    score = evaluation_result[0]  # loss
                else:
                    score = float('inf')  # fallback
                    
                fold_scores.append(score)
                
            except Exception as e:
                print(f"      Erro: {e}")
                continue
        
        return best_params, inner_scores
    
    def _train_final_model(self, X, y, model_fn, best_params, lookback, forecast_horizon):
        """Treina modelo final com melhores parâmetros"""
        import time
        t0 = time.time()
        
        X_seq, y_seq = self._create_sequences_multi_step(X, y, lookback, forecast_horizon)

        if len(X_seq) == 0:
            print(f"   AVISO: Sequências de treino vazias no fold. "
              f"Tamanho X={len(X)}, lookback={lookback}, horizon={forecast_horizon}.")
            return None, 0.0   # ou algum fallback
        
        model = model_fn(**best_params, lookback=lookback,
                       n_features=X_seq.shape[2], forecast_horizon=forecast_horizon)
        
        callbacks = [
            EarlyStopping(patience=config.PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_seq, y_seq,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=0,
            callbacks=callbacks
        )
        
        t1 = time.time()
        
        return model, t1 - t0
    
    def _evaluate_fold(self, model, X_test, y_test, lookback, forecast_horizon):
        """Avalia modelo no fold de teste - MENSAL"""
        try:
            X_seq_test, y_seq_test = self._create_sequences_multi_step(
                X_test, y_test, lookback, forecast_horizon
            )
            
            if len(X_seq_test) == 0:
                print(f"   AVISO: Sequências de teste vazias")
                return {'overall_mape': 100.0, 'overall_mae': 0, 'overall_rmse': 0}, None, None
            
            # CORREÇÃO: Usar try-except no predict
            try:
                y_pred = model.predict(X_seq_test, verbose=0)
            except:
                print(f"   ERRO: Falha no predict")
                return {'overall_mape': 100.0, 'overall_mae': 0, 'overall_rmse': 0}, None, None
                
            y_true = y_seq_test
            
            # Métricas por horizonte (meses)
            horizon_metrics = {}
            for h in range(forecast_horizon):
                y_true_h = y_true[:, h]
                y_pred_h = y_pred[:, h]
                
                mae = mean_absolute_error(y_true_h, y_pred_h)
                rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
                
                # CORREÇÃO: Evitar divisão por zero no MAPE
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((y_true_h - y_pred_h) / np.maximum(np.abs(y_true_h), 1))) * 100
                    if np.isnan(mape):
                        mape = 100.0
                
                horizon_metrics[f'month_{h+1}'] = {
                    'MAE': mae, 'RMSE': rmse, 'MAPE': mape
                }
            
            # Métricas agregadas
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                overall_mape = np.mean(np.abs((y_true_flat - y_pred_flat) / 
                                            np.maximum(np.abs(y_true_flat), 1))) * 100
                if np.isnan(overall_mape):
                    overall_mape = 100.0
            
            metrics = {
                'horizon_metrics': horizon_metrics,
                'overall_mape': overall_mape,
                'overall_mae': mean_absolute_error(y_true_flat, y_pred_flat),
                'overall_rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
            }
            
            return metrics, y_true, y_pred
        
        except Exception as e:
            print(f"   ERRO na avaliação: {e}")
            return {'overall_mape': 100.0, 'overall_mae': 0, 'overall_rmse': 0}, None, None
    
    def _create_sequences_multi_step(self, features, target, lookback, forecast_horizon):
        """Cria sequências para forecast multi-step"""
        X, y = [], []
        for i in range(lookback, len(features) - forecast_horizon):
            X.append(features[i-lookback:i])
            y.append(target[i:i + forecast_horizon])
        return np.array(X), np.array(y)
    
    def _create_param_combinations(self, param_grid):
        """Cria combinações de parâmetros"""
        keys = param_grid.keys()
        values = param_grid.values()
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    def _summarize_results(self):
        """Resume resultados do CV"""
        folds = self.results['folds']
        
        overall_mape = np.mean([fold['metrics']['overall_mape'] for fold in folds])
        overall_mae = np.mean([fold['metrics']['overall_mae'] for fold in folds])
        overall_rmse = np.mean([fold['metrics']['overall_rmse'] for fold in folds])
        
        # Melhor fold
        best_fold_idx = np.argmin([fold['metrics']['overall_mape'] for fold in folds])
        best_fold = folds[best_fold_idx]
        
        summary = {
            'mean_mape': overall_mape,
            'mean_mae': overall_mae,
            'mean_rmse': overall_rmse,
            'std_mape': np.std([fold['metrics']['overall_mape'] for fold in folds]),
            'best_fold': best_fold_idx + 1,
            'best_params': best_fold['best_params'],
            'best_mape': best_fold['metrics']['overall_mape'],
            'folds_details': folds
        }
        
        return summary
    
# =====================================================================

# 6. Criação do modelo LSTM

def create_multi_step_lstm(units=50, dropout_rate=0.2, learning_rate=0.001, 
                          lookback=12, n_features=4, forecast_horizon=6):
    """Cria modelo LSTM para forecast multi-step MENSAL"""
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, 
             input_shape=(lookback, n_features)),
        Dropout(dropout_rate),
        LSTM(units // 2, activation='relu', return_sequences=False),
        Dropout(dropout_rate),
        RepeatVector(forecast_horizon),
        LSTM(units // 2, activation='relu', return_sequences=True),
        Dropout(dropout_rate),
        TimeDistributed(Dense(1))
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# =====================================================================

# 7. Visualização de resultados

def plot_monthly_results(df_features, pca_info, loess_info):
    """Plota resultados específicos para dados mensais"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Série original vs LOESS (mensal)
    axes[0, 0].plot(df_features[config.DATE_COL], df_features['original'], 
                   label='Original', alpha=0.7, linewidth=1.5, marker='o', markersize=4)
    axes[0, 0].plot(df_features[config.DATE_COL], df_features['loess_smooth'], 
                   label='LOESS', linewidth=2, color='red')
    axes[0, 0].set_title('Série temporal mensal\nOriginal X LOESS Smooth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Resíduos
    axes[0, 1].plot(df_features[config.DATE_COL], df_features['residual'], 
                   color='orange', alpha=0.7, marker='o', markersize=4)
    axes[0, 1].set_title('Resíduos do LOESS')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sazonalidade mensal
    monthly_avg = df_features.groupby('month')['original'].mean()
    axes[0, 2].bar(monthly_avg.index, monthly_avg.values, alpha=0.7, color='green')
    axes[0, 2].set_title('Padrão Sazonal Mensal')
    axes[0, 2].set_xlabel('Mês')
    axes[0, 2].set_ylabel('Média')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Variância explicada PCA
    axes[1, 0].bar(range(1, len(pca_info['explained_variance_ratio']) + 1),
                   pca_info['explained_variance_ratio'], color='purple', alpha=0.7)
    axes[1, 0].set_title('Variância Explicada por Componente PCA')
    axes[1, 0].set_xlabel('Componente')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variância acumulada PCA
    axes[1, 1].plot(range(1, len(pca_info['cumulative_variance']) + 1),
                   pca_info['cumulative_variance'], marker='o', color='purple')
    axes[1, 1].axhline(y=0.95, color='red', linestyle='--', label='95%')
    axes[1, 1].axhline(y=0.90, color='orange', linestyle='--', label='90%')
    axes[1, 1].set_title('Variância Acumulada PCA')
    axes[1, 1].set_xlabel('Nº Componentes')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Distribuição resíduos
    axes[1, 2].hist(df_features['residual'], bins=15, alpha=0.7, color='orange')
    axes[1, 2].set_title('Distribuição dos Resíduos Mensais')
    axes[1, 2].set_xlabel('Resíduo')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =====================================================================

# 8. Pipeline de execução principal - desenvolvimento do modelo e sua validação

def complete_pipeline_monthly():
    """Pipeline completa com agregação MENSAL"""
    print("INICIANDO PIPELINE COMPLETA - DADOS MENSAlS")
    print("=" * 60)
    
    # 1. Carregamento de dados
    print("Carregando dados...")
    df = pd.read_excel(config.EXCEL_PATH, sheet_name=config.SHEET_NAME)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors='coerce')
    df = df.dropna(subset=[config.DATE_COL, config.TARGET_COL])
    
    # 2. Agregação mensal
    df_monthly = aggregate_monthly_data(df, config.DATE_COL, config.TARGET_COL)
    
    print(f"Dados mensais: {len(df_monthly)} meses")
    config.explain_temporal_settings()
    
    # 3. Aplicação LOESS
    print("\nAplicando LOESS...")
    loess_processor = LoessPreprocessor(frac=config.LOESS_FRAC, it=config.LOESS_IT)
    df_features, loess_info = loess_processor.create_enhanced_features(
        df_monthly, config.TARGET_COL, config.DATE_COL, lags=12
    )
    
    # 4. Aplicação PCA
    print("\nAplicando PCA...")
    pca_manager = PCAManager(n_components=config.PCA_COMPONENTS)
    pca_df, pca_info = pca_manager.apply_pca(df_features)
    
    # 5. Visualizações
    plot_monthly_results(df_features, pca_info, loess_info)
    
    # 6. Preparação dos dados para LSTM
    print("\nPreparando dados para LSTM...")
    pca_columns = [col for col in pca_df.columns if col.startswith('PC')]
    X = pca_df[pca_columns].values
    y = pca_df['loess_smooth'].values
    
    print(f"   Features PCA: {X.shape}")
    print(f"   Target: {y.shape}")
    
    # 7. GRID SEARCH COM CROSS VALIDATION
    print("\nExecutando Grid Search com Cross Validation...")
    
    param_grid = {
        'units': config.NEURONS_LIST,
        'dropout_rate': config.DROPOUT_RATES,
        'learning_rate': config.LEARNING_RATES
    }
    
    all_results = []
    
    for lookback in config.LOOKBACK_LIST:
        print(f"\n\tTestando lookback = {lookback} meses")
        
        cv = TimeSeriesCrossValidator(n_splits=config.N_SPLITS)
        results = cv.nested_cross_validation(
            X=X, y=y,
            model_fn=create_multi_step_lstm,
            param_grid=param_grid,
            lookback=lookback,
            forecast_horizon=config.FORECAST_HORIZON
        )
        
        results['lookback'] = lookback
        all_results.append(results)
    
    # 8. Análise de resultados
    print("\nAnálise dos resultados...")
    best_config = analyze_monthly_results(all_results, pca_df, config.FORECAST_HORIZON)
    
    # 9. Treinamento do modelo final
    print("\nTreinamento do modelo final com melhor configuração...")
    final_model = train_final_model_monthly(best_config, X, y, pca_df, config.FORECAST_HORIZON)
    
    return final_model, best_config, all_results, df_features, pca_df

def analyze_monthly_results(all_results, pca_df, forecast_horizon):
    """Analisa resultados do Grid Search CV para dados mensais"""
    print("\n" + "="*50)
    print("RESULTADOS DO GRID SEARCH COM CV - DADOS MENSAlS")
    print("="*50)
    
    best_result = min(all_results, key=lambda x: x['mean_mape'])
    
    print(f"MELHOR CONFIGURAÇÃO:")
    print(f"   Lookback: {best_result['lookback']} meses")
    print(f"   Parâmetros: {best_result['best_params']}")
    print(f"   MAPE: {best_result['mean_mape']:.2f}%")
    print(f"   MAE: {best_result['mean_mae']:.2f}")
    print(f"   RMSE: {best_result['mean_rmse']:.2f}")
    
    # Plot performance por horizonte (meses)
    best_fold = best_result['folds_details'][best_result['best_fold'] - 1]
    horizon_metrics = best_fold['metrics']['horizon_metrics']
    
    horizons = list(range(1, forecast_horizon + 1))
    mapes = [horizon_metrics[f'month_{h}']['MAPE'] for h in horizons]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(horizons, mapes, marker='o', linewidth=2, color='blue')
    plt.title('MAPE por Horizonte de Previsão (Meses)')
    plt.xlabel('Meses à Frente')
    plt.ylabel('MAPE (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot comparação de lookbacks
    plt.subplot(1, 2, 2)
    lookbacks = [r['lookback'] for r in all_results]
    mean_mapes = [r['mean_mape'] for r in all_results]
    
    plt.bar(range(len(lookbacks)), mean_mapes, color='skyblue', alpha=0.7)
    plt.xticks(range(len(lookbacks)), [f'{lb}m' for lb in lookbacks])
    plt.title('Performance por Lookback (Meses)')
    plt.xlabel('Lookback (meses)')
    plt.ylabel('MAPE Médio (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'lookback': best_result['lookback'],
        'params': best_result['best_params'],
        'pca_components': pca_df.shape[1] - 3,
        'forecast_horizon': forecast_horizon
    }
    
def create_sequences_multi_step(features, target, lookback, forecast_horizon):
    """
    Cria sequências para forecast multi-step
    """
    X, y = [], []
    for i in range(lookback, len(features) - forecast_horizon):
        X.append(features[i-lookback:i])
        y.append(target[i:i + forecast_horizon])
    return np.array(X), np.array(y)

def train_final_model_monthly(best_config, X, y, pca_df, forecast_horizon):
    """Treina modelo final com melhor configuração - MENSAL"""
    print("\nTreinando modelo final...")
    
    lookback = best_config['lookback']
    params = best_config['params']
    
    # Criar sequências completas
    X_seq, y_seq = create_sequences_multi_step(X, y, lookback, forecast_horizon)
    n_features = X_seq.shape[2]
    
    # Modelo final
    model = create_multi_step_lstm(
        units=params['units'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate'],
        lookback=lookback,
        n_features=n_features,
        forecast_horizon=forecast_horizon
    )
    
    # Split final treino/teste
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
    ]
    
    print(f"   Dados de treino: {X_train.shape}")
    print(f"   Dados de teste: {X_test.shape}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Avaliação final
    y_pred = model.predict(X_test, verbose=0)
    y_true = y_test
    
    overall_mape = np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / 
                                np.maximum(np.abs(y_true.flatten()), 1))) * 100
    
    print(f"\n🎉 MODELO FINAL TREINADO!")
    print(f"   MAPE Final: {overall_mape:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss do Modelo Final (Mensal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE do Modelo Final (Mensal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model, history
    
# =====================================================================

# 9. Execução principal

# Configurações adicionais
def setup_environment():
    """Configura ambiente para execução"""
    # Limpar sessão Keras
    tf.keras.backend.clear_session()
    
    # Configurar reproduibilidade
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Ambiente configurado!")

# Execução principal
def main():
    """Função principal de execução"""
    setup_environment()
    
    try:
        final_model, best_config, all_results, df_features, pca_df = complete_pipeline_monthly()
        print("\nPIPELINE CONCLUÍDO COM SUCESSO!")
        return final_model, best_config, all_results, df_features, pca_df
    except Exception as e:
        print(f"Erro na execução: {e}")
        return None, None, None, None, None

# Executar
if __name__ == "__main__":
    final_model, best_config, all_results, df_features, pca_df = main()

    # 1. Reconstruir X (features), y (alvo) e datas a partir do pca_df
    pca_columns = [c for c in pca_df.columns if c.startswith('PC')]
    X_all = pca_df[pca_columns].values
    y_all = pca_df['loess_smooth'].values
    dates_all = pd.to_datetime(pca_df['Data'])

    lookback = best_config['lookback']
    forecast_horizon = best_config['forecast_horizon']

    print("Shape original de X_all:", X_all.shape)
    print("Lookback:", lookback, " - Horizonte:", forecast_horizon)

    # 2. Criar sequências para o dataset inteiro
    X_seq_all, y_seq_all = create_sequences_multi_step(
        X_all, y_all, lookback, forecast_horizon
    )

    print("Shape das sequências:")
    print("  X_seq_all:", X_seq_all.shape)
    print("  y_seq_all:", y_seq_all.shape)

    # 3. Garantir que temos o modelo (não a tupla model, history)
    if isinstance(final_model, tuple):
        model_for_pred = final_model[0]
    else:
        model_for_pred = final_model

    # 4. Prever (multi-step) usando o modelo final
    y_pred_seq = model_for_pred.predict(X_seq_all)

    # y_pred_seq tem shape (n_amostras, forecast_horizon, 1)

    # 5. Espalhar as previsões no eixo do tempo
    N = len(y_all)
    full_pred = np.full(N, np.nan, dtype=float)
    counts = np.zeros(N, dtype=int)

    # Índice base de cada sequência (i no create_sequences_multi_step)
    # range(lookback, len(features) - forecast_horizon)
    base_indices = np.arange(lookback, len(y_all) - forecast_horizon)

    for seq_idx, i in enumerate(base_indices):
        for h in range(forecast_horizon):
            t = i + h  # índice temporal correspondente à previsão h
            if t < N:
                pred_value = y_pred_seq[seq_idx, h, 0]
                if np.isfinite(pred_value):
                    if np.isnan(full_pred[t]):
                        full_pred[t] = 0.0
                    full_pred[t] += pred_value
                    counts[t] += 1

    # Média quando houver mais de uma previsão para o mesmo ponto
    mask_valid = counts > 0
    full_pred[mask_valid] = full_pred[mask_valid] / counts[mask_valid]

    # 6. Montar DataFrame com série inteira
    df_all = pd.DataFrame({
        'Data': dates_all,
        'Real': y_all,
        'Previsto': full_pred
    })

    # 7. Filtrar apenas 2025 (onde há previsão)
    mask_2025 = (df_all['Data'].dt.year == 2025) & df_all['Previsto'].notna()
    df_2025 = df_all[mask_2025].copy()

    print("\nPontos previstos em 2025:")
    print(df_2025)

    # 8. Gráfico completo: Real vs Previsto (toda a série)
    plt.figure(figsize=(14, 6))
    plt.plot(df_all['Data'], df_all['Real'], label='Real (LOESS)', linewidth=2)
    plt.plot(df_all['Data'], df_all['Previsto'], label='Previsto (multi-step médio)', linestyle='--')
    plt.axvspan(pd.Timestamp('2025-01-01'),
                pd.Timestamp('2025-12-31'),
                alpha=0.1, label='Período 2025')
    plt.title('Série completa - Real vs Previsto (multi-step)')
    plt.xlabel('Data')
    plt.ylabel('Quantidade (LOESS suavizada)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9. Gráfico só de 2025
    if not df_2025.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(df_2025['Data'], df_2025['Real'], marker='o', label='Real 2025')
        plt.plot(df_2025['Data'], df_2025['Previsto'], marker='s', linestyle='--', label='Previsto 2025')
        plt.title('Previsão para 2025 - Real vs Previsto')
        plt.xlabel('Data')
        plt.ylabel('Quantidade (LOESS suavizada)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Não há pontos de 2025 com previsão (verifique se a base chega em 2025).")

