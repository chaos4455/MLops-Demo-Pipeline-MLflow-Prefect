# Relatório de Teste da API de Modelo MLflow (2025-09-01 19:31:58)

Este relatório resume o teste de consumo da API de modelo MLflow para previsão de vendas Chevrolet.

## 1. Configurações do Teste
- **URL da API de Modelo:** `http://127.0.0.1:8780/predict`
- **Número de Amostras Testadas:** `100`

## 2. Estatísticas das Previsões
- **Mínimo de Vendas Previstas:** `3.40`
- **Máximo de Vendas Previstas:** `29.70`
- **Média de Vendas Previstas:** `15.99`
- **Desvio Padrão das Vendas Previstas:** `6.01`

## 3. Amostra de Previsões
Aqui estão as primeiras 5 amostras de entrada com suas respectivas previsões:
| store_id   | model     |   month |   day_of_week |   predicted_sales |
|:-----------|:----------|--------:|--------------:|------------------:|
| Loja_09    | Montana   |      11 |             5 |             27.26 |
| Loja_20    | Equinox   |      10 |             4 |             18.02 |
| Loja_01    | Montana   |      12 |             0 |             24.46 |
| Loja_12    | Montana   |       3 |             5 |             21.32 |
| Loja_12    | Onix Plus |       3 |             6 |              6.58 |

## 4. Visualização
Um gráfico da distribuição das previsões e média por modelo foi gerado:
![Gráfico de Previsões](model_test_report_overview.png)

---