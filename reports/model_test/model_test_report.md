# Relat�rio de Teste da API de Modelo MLflow (2025-09-01 19:31:58)

Este relat�rio resume o teste de consumo da API de modelo MLflow para previs�o de vendas Chevrolet.

## 1. Configura��es do Teste
- **URL da API de Modelo:** `http://127.0.0.1:8780/predict`
- **N�mero de Amostras Testadas:** `100`

## 2. Estat�sticas das Previs�es
- **M�nimo de Vendas Previstas:** `3.40`
- **M�ximo de Vendas Previstas:** `29.70`
- **M�dia de Vendas Previstas:** `15.99`
- **Desvio Padr�o das Vendas Previstas:** `6.01`

## 3. Amostra de Previs�es
Aqui est�o as primeiras 5 amostras de entrada com suas respectivas previs�es:
| store_id   | model     |   month |   day_of_week |   predicted_sales |
|:-----------|:----------|--------:|--------------:|------------------:|
| Loja_09    | Montana   |      11 |             5 |             27.26 |
| Loja_20    | Equinox   |      10 |             4 |             18.02 |
| Loja_01    | Montana   |      12 |             0 |             24.46 |
| Loja_12    | Montana   |       3 |             5 |             21.32 |
| Loja_12    | Onix Plus |       3 |             6 |              6.58 |

## 4. Visualiza��o
Um gr�fico da distribui��o das previs�es e m�dia por modelo foi gerado:
![Gr�fico de Previs�es](model_test_report_overview.png)

---