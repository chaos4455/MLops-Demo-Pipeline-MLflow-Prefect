import pandas as pd
import numpy as np
from datetime import date, timedelta
from flask import Flask, jsonify
from flask_cors import CORS
import random

# --- 1. Geração de Dados de Vendas Fictícios e Realistas ---

def generate_sales_data(num_stores=20, year=2024):
    """
    Gera dados de vendas realistas para uma concessionária Chevrolet no Brasil.

    Args:
        num_stores (int): Número de lojas.
        year (int): Ano para o qual os dados serão gerados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados de vendas.
    """
    sales_data = []
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    # Modelos Chevrolet populares no Brasil e seus preços médios fictícios (em BRL)
    # Baseado em informações de mercado de 2024 [1, 2, 3, 4]
    models = {
        "Onix": 90000,
        "Onix Plus": 95000,
        "Tracker": 130000,
        "Montana": 140000,
        "Spin": 120000,
        "S10": 250000,
        "Equinox": 200000,
        "Cobalt": 70000 # Retorno em 2024, preço ajustado para cenário fictício [2]
    }

    # Fatores de sazonalidade mensal (ajustados para simular tendências de vendas no Brasil)
    # Dezembro e Outubro são geralmente meses fortes [3, 8]
    # Janeiro e Julho podem ser um pouco mais baixos
    monthly_seasonality = {
        1: 0.85,  # Jan
        2: 0.90,  # Fev
        3: 1.05,  # Mar
        4: 1.00,  # Abr
        5: 1.05,  # Mai
        6: 0.95,  # Jun
        7: 0.90,  # Jul
        8: 1.05,  # Ago
        9: 1.00,  # Set
        10: 1.15, # Out
        11: 1.10, # Nov
        12: 1.20  # Dez
    }

    current_date = start_date
    while current_date <= end_date:
        month = current_date.month
        day_of_week = current_date.weekday() # 0=Segunda, 5=Sábado, 6=Domingo

        # Ajuste base para vendas diárias (ex: 20 a 40 carros por loja/dia)
        # Sábados geralmente têm mais vendas, domingos menos ou nenhum
        base_daily_sales = random.randint(20, 40)
        if day_of_week == 5: # Sábado
            base_daily_sales = int(base_daily_sales * 1.3)
        elif day_of_week == 6: # Domingo
            base_daily_sales = int(base_daily_sales * 0.3) # Vendas mínimas ou fechado

        for store_id in range(1, num_stores + 1):
            # Aplica sazonalidade mensal
            adjusted_daily_sales = int(base_daily_sales * monthly_seasonality[month] * (1 + np.random.normal(0, 0.1))) # Adiciona ruído
            
            # Garante que não haja vendas negativas
            if adjusted_daily_sales < 0:
                adjusted_daily_sales = 0

            cars_sold_today = max(0, adjusted_daily_sales)

            for _ in range(cars_sold_today):
                model_name = random.choice(list(models.keys()))
                price = models[model_name]
                
                # Adiciona alguma variação de preço para realismo (ex: pacotes, acessórios)
                final_price = price * (1 + random.uniform(-0.05, 0.05))

                sales_data.append({
                    "store_id": f"Loja_{store_id:02d}",
                    "date": current_date.isoformat(),
                    "model": model_name,
                    "sales_count": 1, # Cada linha é uma venda de um carro
                    "revenue": round(final_price, 2)
                })
        current_date += timedelta(days=1)
    
    return pd.DataFrame(sales_data)

# Gera os dados uma vez ao iniciar a aplicação
print("Gerando dados de vendas realistas para 2024...")
df_sales = generate_sales_data()
print(f"Dados gerados: {len(df_sales)} registros.")

# --- 2. Criação da API Flask com CORS ---

app = Flask(__name__)
CORS(app) # Habilita CORS para todas as origens

@app.route('/sales_data', methods=['GET'])
def get_sales_data():
    """
    Endpoint que retorna os dados de vendas gerados.
    """
    # Retorna o DataFrame como JSON
    return jsonify(df_sales.to_dict(orient='records'))

@app.route('/')
def home():
    """
    Endpoint inicial para verificar se a API está funcionando.
    """
    return "API de Dados de Vendas Chevrolet 2024 está rodando! Acesse /sales_data para obter os dados."

if __name__ == '__main__':
    # A API será executada na porta 8777
    print("Iniciando a API Flask na porta 8777...")
    app.run(port=8777, debug=True)