# NOME DO ARQUIVO: ML-FLOW-MODEL-API-SERVING.PY
# TAMANHO: Aproximadamente 4.5 KB
# NÚMERO DE LINHAS: Aproximadamente 130 (com a correção)

import mlflow
import pandas as pd
import numpy as np # Importar numpy para os tipos de dados
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

# --- Configurações da API e MLflow ---
FLASK_APP_PORT = 8780 # Porta para esta API de serviço de modelo
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001" # Seu MLflow Tracking Server
MLFLOW_MODEL_NAME = "ChevroletSalesPredictor" # Nome do modelo registrado no MLflow
MLFLOW_MODEL_VERSION = "latest" # Use "latest" para a última versão, ou um número específico (e.g., "1")

app = Flask(__name__)
CORS(app) # Habilita CORS para todas as origens

# Variável global para armazenar o modelo carregado
loaded_model = None

# --- Funções de Inicialização e Carregamento do Modelo ---

def load_mlflow_model():
    """
    Carrega a versão mais recente (ou especificada) do modelo do MLflow Registry.
    Esta função é chamada uma vez na inicialização da API.
    """
    global loaded_model
    print("╔═════════════════════════════════════════════════════════╗")
    print(f"║   Iniciando carregamento do Modelo MLflow '{MLFLOW_MODEL_NAME}'  ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print(f"Conectando ao MLflow Tracking Server: \033[1;34m{MLFLOW_TRACKING_URI}\033[0m")
    print(f"Tentando carregar a versão '{MLFLOW_MODEL_VERSION}' do modelo '{MLFLOW_MODEL_NAME}'...")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # O URI para carregar um modelo do Model Registry é "models:/<model_name>/<version_or_stage>"
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        print(f"\033[1;32mSucesso:\033[0m Modelo '{MLFLOW_MODEL_NAME}' versão '{MLFLOW_MODEL_VERSION}' carregado com sucesso!")
        print(f"Pronto para fazer previsões na porta \033[1m{FLASK_APP_PORT}\033[0m.")
        return True
    except mlflow.exceptions.RestException as e:
        print(f"\033[1;31mERRO MLflow:\033[0m Não foi possível carregar o modelo '{MLFLOW_MODEL_NAME}' do MLflow Registry.")
        print(f"Detalhes do erro: {e}")
        print("Certifique-se de que:")
        print(f"  - O MLflow Tracking Server esteja rodando em \033[1;34m{MLFLOW_TRACKING_URI}\033[0m.")
        print(f"  - O modelo '{MLFLOW_MODEL_NAME}' esteja registrado no MLflow Model Registry.")
        print(f"  - A versão '{MLFLOW_MODEL_VERSION}' do modelo exista ou esteja na fase correta.")
        return False
    except Exception as e:
        print(f"\033[1;31mERRO inesperado ao carregar o modelo: {e}\033[0m")
        return False

# --- Endpoints da API Flask ---

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint para receber dados via POST e fazer previsões usando o modelo MLflow carregado.
    Espera uma lista de objetos JSON no corpo da requisição.
    Cada objeto JSON deve conter as features: 'store_id', 'model', 'month', 'day_of_week'.
    """
    if loaded_model is None:
        print("\033[1;31mERRO: O modelo ainda não foi carregado ou falhou ao carregar.\033[0m")
        return jsonify({"error": "Modelo não disponível. Por favor, verifique os logs da API."}), 503 # Service Unavailable

    if not request.is_json:
        print("\033[1;31mERRO: Requisição /predict deve ser JSON.\033[0m")
        return jsonify({"message": "Content-Type deve ser application/json"}), 400

    data = request.get_json()
    if not isinstance(data, list):
        print("\033[1;31mERRO: Requisição /predict espera uma lista de objetos JSON.\033[0m")
        return jsonify({"message": "O corpo da requisição deve ser uma lista de objetos JSON"}), 400

    if not data:
        print("\033[33mAVISO: Requisição /predict recebida, mas com dados vazios. Retornando lista vazia de previsões.\033[0m")
        return jsonify({"predictions": []}), 200

    try:
        # As features esperadas pelo modelo são: 'store_id', 'model', 'month', 'day_of_week'
        # O modelo pyfunc espera um DataFrame com essas colunas exatas.
        input_df = pd.DataFrame(data)
        
        # Validar se todas as colunas esperadas estão presentes
        expected_features = ['store_id', 'model', 'month', 'day_of_week']
        missing_features = [f for f in expected_features if f not in input_df.columns]
        if missing_features:
            print(f"\033[1;31mERRO: Features ausentes na requisição de previsão: {missing_features}.\033[0m")
            return jsonify({"error": f"Features obrigatórias ausentes: {', '.join(missing_features)}. Esperado: {', '.join(expected_features)}"}), 400

        # Garantir a ordem das colunas para o modelo
        input_df = input_df[expected_features]

        # --- CORREÇÃO: Converter tipos de dados para int32 para colunas numéricas ---
        # Garante que o DataFrame de entrada corresponda ao schema do modelo registrado
        # Pandas por padrão pode usar int64, mas o MLflow Model Signature espera int32.
        for col in ['month', 'day_of_week']:
            if col in input_df.columns:
                # Usa .loc para evitar SettingWithCopyWarning e garantir que a modificação seja no df original
                input_df.loc[:, col] = input_df[col].astype(np.int32)
        # -------------------------------------------------------------------------

        # Fazer as previsões
        predictions = loaded_model.predict(input_df)
        
        # Convertendo as previsões (que são um array numpy) para uma lista Python para jsonify
        predictions_list = predictions.flatten().tolist()

        print(f"\033[1;32mSucesso:\033[0m {len(predictions_list)} previsões geradas.")
        return jsonify({"predictions": predictions_list}), 200
    except KeyError as e:
        print(f"\033[1;31mERRO de coluna: {e}. Verifique se o nome das colunas nos dados de entrada corresponde às features do modelo.\033[0m")
        return jsonify({"error": f"Erro nas colunas de entrada: {e}. Certifique-se de que os dados de entrada contenham as colunas 'store_id', 'model', 'month', 'day_of_week'."}), 400
    except Exception as e:
        print(f"\033[1;31mERRO inesperado ao fazer previsão: {e}\033[0m")
        import traceback
        traceback.print_exc() # Imprime o stack trace para depuração
        return jsonify({"error": f"Falha ao gerar previsões: {e}"}), 500

@app.route('/model_status', methods=['GET'])
def model_status_endpoint():
    """
    Endpoint para verificar o status do carregamento do modelo.
    """
    if loaded_model:
        return jsonify({"status": "ok", "model_loaded": True, "model_name": MLFLOW_MODEL_NAME, "model_version": MLFLOW_MODEL_VERSION}), 200
    else:
        return jsonify({"status": "error", "model_loaded": False, "message": "Modelo não carregado ou falhou ao carregar."}), 503

@app.route('/')
def home():
    """
    Endpoint inicial para verificar se a API está funcionando.
    """
    status_message = "Pronto para previsões." if loaded_model else "Modelo não carregado. Verifique os logs."
    return (
        f"<h1>API de Serviço de Modelo MLflow - Previsão de Vendas Chevrolet</h1>"
        f"Esta API está rodando na porta \033[1m{FLASK_APP_PORT}\033[0m.<br>"
        f"Status do Modelo: \033[1m{status_message}\033[0m<br>"
        f"<ul>"
        f"<li>\033[1mPOST\033[0m para <a href='http://127.0.0.1:{FLASK_APP_PORT}/predict'>/predict</a> para obter previsões (corpo JSON).</li>"
        f"<li>\033[1mGET\033[0m para <a href='http://127.0.0.1:{FLASK_APP_PORT}/model_status'>/model_status</a> para verificar o status do modelo.</li>"
        f"</ul>"
    )

if __name__ == '__main__':
    # Tenta carregar o modelo ao iniciar a aplicação.
    if load_mlflow_model():
        print(f"\nAPI Flask de Serviço de Modelo pronta. Acesse \033[1;34mhttp://127.0.0.1:{FLASK_APP_PORT}\033[0m")
        print("\033[33mCertifique-se de que o MLflow Tracking Server esteja rodando na porta 5001.\033[0m")
        app.run(port=FLASK_APP_PORT, debug=True)
    else:
        print("\033[1;31mNÃO foi possível iniciar a API de serviço de modelo devido a falha no carregamento do modelo MLflow.\033[0m")
        sys.exit(1)