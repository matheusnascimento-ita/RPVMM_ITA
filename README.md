# RPVMM_ITA
Projeto da matéria Resolução de Problemas via Modelagem Matemática

# Estimativa de Tiragem de Materiais Didáticos: Um Modelo Integrado para o Poliedro Educação

Este repositório contém o código-fonte e a documentação do projeto de modelagem matemática desenvolvido para a disciplina de **Resolução de Problemas via Modelagem Matemática (RPVMM)**. O objetivo é otimizar a previsão de tiragem de materiais didáticos utilizando técnicas avançadas de ciência de dados.

## Visão Geral

O setor editorial educacional enfrenta o desafio de sincronizar a produção com o calendário escolar rígido. Este projeto propõe um **modelo híbrido** que combina métodos estatísticos clássicos e aprendizado profundo para prever a demanda (tiragem) de coleções didáticas.

### Metodologia Híbrida
O modelo baseia-se em três pilares principais:
1.  **LOESS (Locally Estimated Scatterplot Smoothing):** Utilizado para decompor e suavizar o ruído da série temporal original, facilitando a identificação de tendências.
2.  **PCA (Principal Component Analysis):** Redução de dimensionalidade para tratar a correlação entre diferentes segmentos de prospecção e fidelização.
3.  **LSTM (Long Short-Term Memory):** Redes neurais recorrentes capazes de capturar dependências temporais de longo prazo e a sazonalidade do ciclo escolar.

## Arquitetura do Modelo
O modelo final utiliza uma arquitetura Encoder-Decoder para previsões multi-passo:

1. **Input:** Janela temporal (Lookback) processada via PCA.
2. **Processamento:** Camadas LSTM para capturar sazonalidade e tendência.
3. **Output:** Previsão da tiragem para os próximos meses de 2025.

## Estrutura do Repositório

* `Modelo LSTM Base.py`: Implementação inicial da rede LSTM para séries temporais univariadas.
* `Modelo LSTM com PCA.py`: Evolução do modelo integrando componentes principais como variáveis exógenas.
* `Modelo LSTM Multi-step com PCA e LOESS.py`: O modelo final e mais robusto, realizando previsões de múltiplos passos à frente com dados suavizados.
* `Tabelas de validacao dos Modelos.py`: Script para geração automática das métricas de performance (MAE, RMSE, MAPE) e comparação com baselines.
* `poliedro_modelo_atual_overleaf.pdf`: Artigo científico completo detalhando a fundamentação teórica e os resultados obtidos.

### Pré-requisitos
* Python 3.9+
* Bibliotecas: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `statsmodels`.

### Instalação
1. Clone o repositório:
   ```bash
   git clone [https://github.com/matheusnascimento-ita/RPVMM_ITA.git](https://github.com/matheusnascimento-ita/RPVMM_ITA.git)

Principais Resultados
O modelo integrado apresentou ganhos consistentes de acurácia em comparação com métodos tradicionais de média móvel e regressão linear simples, reduzindo significativamente o erro percentual médio (MAPE) nas coleções de maior volume.

# Autores:
Davi de Souza Santos (UNIFESP)

Eduardo Cipolaro (UNIFESP)

Lucas Mieri (ITA)

Matheus Nascimento (ITA)

Thiago Cortez Cursino dos Santos (UNIFESP)
