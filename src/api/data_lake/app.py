import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sqlite3

# --- Configurações da API de Storage ---
FLASK_APP_PORT = 8778
DATABASE_DIR = "../../data" # Diretório para armazenar o arquivo do banco de dados
DATABASE_PATH = os.path.join(DATABASE_DIR, "sales_data_lake.db") # Caminho completo para o DB SQLite
TABLE_NAME = "raw_sales" # Nome da tabela dentro do SQLite DB

app = Flask(__name__)
CORS(app) # Habilita CORS para todas as origens na nova API

# --- Funções de Inicialização do Banco de Dados ---

def initialize_database():
    """
    Garante que o diretório do banco de dados exista e cria a tabela inicial
    no SQLite se ela ainda não existir.
    """
    os.makedirs(DATABASE_DIR, exist_ok=True)
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        # Cria a tabela com um esquema básico.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                store_id TEXT,
                date TEXT,
                model TEXT,
                sales_count INTEGER,
                revenue REAL
            );
        """)
        conn.commit()
        conn.close()
        print(f"Diretório '{DATABASE_DIR}' e tabela '{TABLE_NAME}' no '{DATABASE_PATH}' garantidos.")
    except Exception as e:
        print(f"\033[1;31mERRO ao inicializar o banco de dados: {e}\033[0m")

# --- Endpoints da Nova API Flask ---

@app.route('/store_data', methods=['POST'])
def store_data_endpoint():
    """
    Endpoint para receber dados via POST e armazená-los no banco de dados SQLite.
    Espera uma lista de objetos JSON no corpo da requisição.
    """
    if not request.is_json:
        print("\033[1;31mERRO: Requisição /store_data deve ser JSON.\033[0m")
        return jsonify({"message": "Content-Type deve ser application/json"}), 400

    data = request.get_json()
    if not isinstance(data, list):
        print("\033[1;31mERRO: Requisição /store_data espera uma lista de objetos JSON.\033[0m")
        return jsonify({"message": "O corpo da requisição deve ser uma lista de objetos JSON"}), 400

    if not data:
        print("\033[33mAVISO: Requisição /store_data recebida, mas com dados vazios. Nada será armazenado.\033[0m")
        return jsonify({"message": "Nenhum dado recebido para armazenar."}), 200

    try:
        df = pd.DataFrame(data)
        
        conn = sqlite3.connect(DATABASE_PATH)
        # Substitui o conteúdo existente. Em um cenário real, você pode querer anexar ou fazer um upsert.
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"\033[1;32mSucesso:\033[0m {len(df)} registros recebidos e armazenados em '{DATABASE_PATH}' (tabela: {TABLE_NAME}).")
        return jsonify({"message": f"{len(df)} registros armazenados com sucesso!", "status": "ok"}), 200
    except Exception as e:
        print(f"\033[1;31mERRO ao armazenar dados no banco de dados: {e}\033[0m")
        return jsonify({"error": f"Falha ao armazenar dados: {e}"}), 500

@app.route('/datasource', methods=['GET'])
def get_datasource():
    """
    Endpoint que serve os dados lidos do banco de dados SQLite local,
    simulando uma fonte de dados de um "data lake" ou storage.
    """
    print(f"Requisição recebida para /datasource. Servindo dados de '{DATABASE_PATH}' (tabela: {TABLE_NAME}).")
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        
        if df.empty:
            print("\033[33mAVISO: O banco de dados está vazio. Nenhuma venda disponível.\033[0m")
            return jsonify({"message": "Nenhum dado disponível na fonte de dados.", "data": []}), 200
            
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        print(f"\033[1;31mERRO: Não foi possível ler dados do banco de dados: {e}\033[0m")
        return jsonify({"error": f"Falha ao recuperar dados da fonte: {e}"}), 500

@app.route('/')
def home():
    """
    Endpoint inicial para verificar se a API está funcionando.
    """
    return (
        f"<h1>API de Data Storage (Passiva) Chevrolet 2024</h1>"
        f"Esta API está rodando na porta \033[1m{FLASK_APP_PORT}\033[0m.<br>"
        f"<ul>"
        f"<li>Envie dados via \033[1mPOST\033[0m para <a href='http://127.0.0.1:{FLASK_APP_PORT}/store_data'>/store_data</a> (corpo JSON).</li>"
        f"<li>Acesse <a href='http://127.0.0.1:{FLASK_APP_PORT}/datasource'>/datasource</a> para obter os dados armazenados.</li>"
        f"</ul>"
    )

if __name__ == '__main__':
    print("╔═════════════════════════════════════════════════════════╗")
    print(f"║   Iniciando API de Data Storage (Passiva) na porta {FLASK_APP_PORT}  ║")
    print("╚═════════════════════════════════════════════════════════╝")

    initialize_database() # Garante que o diretório do DB e a tabela existam
    
    print(f"\nAPI Flask de Data Storage (Passiva) pronta. Acesse \033[1;34mhttp://127.0.0.1:{FLASK_APP_PORT}\033[0m")
    print("\033[33mAtenção: Esta API NÃO busca dados ativamente. Use o Prefect Flow para ingestão.\033[0m")
    app.run(port=FLASK_APP_PORT, debug=True)