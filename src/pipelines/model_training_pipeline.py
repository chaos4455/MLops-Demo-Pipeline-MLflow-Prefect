import mlflow
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests
import os
import shutil
from datetime import datetime
import sys # Para obter a versão do Python no ambiente Conda

# Importações para ML
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib # Para salvar e carregar o preprocessor
import yaml # Para criar o ambiente Conda
import sklearn # Para obter a versão do scikit-learn

# Fix: Use mlflow.types.DataType e importe ModelSignature, Schema, ColSpec diretamente
from mlflow.models.signature import ModelSignature, Schema, ColSpec
from mlflow.types import DataType # Importa DataType de mlflow.types

# --- Configurações dos Endpoints e MLflow ---
DATALAKE_API_URL = "http://127.0.0.1:8778/datasource" # Sua API de Data Lake (passiva)
ARTIFACT_STORAGE_UPLOAD_URL = "http://127.0.0.1:8779/upload_artifact" # Sua API de Storage de Artefatos
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001" # Seu MLflow Tracking Server

# Diretório temporário para salvar artefatos localmente antes de logar/upload
TEMP_ARTIFACTS_DIR = "temp_pipeline_artifacts"

# Configura o MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Garante que o TensorFlow não consuma toda a GPU VRAM de uma vez (se houver GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(f"Erro ao configurar memória da GPU: {e}")

# --- Definição do Modelo Customizado MLflow Pyfunc ---
# Este modelo encapsula o pré-processador e o modelo Keras
class SalesPredictionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Carrega o pré-processador e o modelo Keras a partir dos artefatos.
        """
        import tensorflow as tf
        import joblib
        # É importante importar todas as classes usadas para deserializar o ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder # Necessário para deserialização
        from sklearn.compose import ColumnTransformer # Necessário para deserialização

        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        self.model = tf.keras.models.load_model(context.artifacts["keras_model"])

    def predict(self, context, model_input):
        """
        Pré-processa a entrada e faz previsões usando o modelo Keras.
        `model_input` é esperado como um DataFrame pandas com as colunas brutas.
        """
        # Pré-processa os dados de entrada usando o pré-processador carregado
        processed_input = self.preprocessor.transform(model_input)
        
        # Faz as previsões com o modelo Keras
        predictions = self.model.predict(processed_input)
        
        # Garante que as previsões não sejam negativas
        predictions[predictions < 0] = 0

        # Retorna as previsões como um array 1D
        return predictions.flatten()

# --- Funções Auxiliares do Pipeline ---

def fetch_data(url: str) -> pd.DataFrame:
    """
    Busca os dados de vendas da API de Data Lake.
    """
    print(f"Buscando dados da Data Lake API: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Levanta um HTTPError para respostas de erro (4xx ou 5xx)
        data = response.json()
        
        if not data:
            print("\033[33mAVISO: Nenhum dado recebido da Data Lake API. Retornando DataFrame vazio.\033[0m")
            return pd.DataFrame()
        
        if not isinstance(data, list):
            print(f"\033[1;31mERRO: Formato de dados inesperado da Data Lake API.\033[0m")
            print(f"Esperava uma lista (JSON array), mas recebeu tipo: {type(data)}. Conteúdo (primeiros 200 chars): {str(data)[:200]}...")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        print(f"\033[1;32mSucesso:\033[0m {len(df)} registros recebidos da Data Lake API.")
        return df
    except requests.exceptions.ConnectionError:
        print(f"\033[1;31mERRO: Não foi possível conectar à Data Lake API em {url}.\033[0m")
        print("Certifique-se de que sua API de Storage (api_datasource_storage_passive.py) esteja rodando na porta 8778.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"\033[1;31mERRO HTTP ao buscar dados da Data Lake API: {e}.\033[0m Resposta: {e.response.text}")
        raise
    except Exception as e:
        print(f"\033[1;31mERRO inesperado ao buscar dados da Data Lake API: {e}\033[0m")
        raise

def preprocess_data_for_ml(df: pd.DataFrame):
    """
    Pré-processa os dados para o treinamento do modelo de ML.
    Agrega os dados de vendas para obter 'total_sales' por categoria relevante.
    Cria um pipeline de pré-processamento para features categóricas.
    """
    print("Pré-processando e agregando dados para treinamento do modelo ML...")
    if df.empty:
        raise ValueError("DataFrame vazio para pré-processamento.")

    # Converter 'date' para datetime e extrair features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek # Segunda=0, Domingo=6

    # Definir features que serão usadas para agregação e como features do modelo
    grouping_features = ['store_id', 'model', 'month', 'day_of_week']
    target = 'total_sales' # O novo target será o total de vendas agregadas

    # Agregação: Somar 'sales_count' por grupo
    # A coluna original 'sales_count' em 'df' é 1 por transação.
    # Precisamos somar estas para obter o total de vendas para cada combinação de features.
    df_agg = df.groupby(grouping_features).agg(
        total_sales=('sales_count', 'sum') # Agrega 'sales_count' para criar o novo target
    ).reset_index()

    # Remover linhas com valores ausentes no target ou features críticas
    df_agg = df_agg.dropna(subset=[target] + grouping_features)

    # Garantir que 'total_sales' seja numérico e não negativo
    df_agg[target] = pd.to_numeric(df_agg[target], errors='coerce').fillna(0).astype(int)
    df_agg = df_agg[df_agg[target] >= 0]

    if df_agg.empty:
        raise ValueError("DataFrame vazio após agregação, pré-processamento e limpeza.")

    X = df_agg[grouping_features]
    y = df_agg[target]

    categorical_features = ['store_id', 'model', 'month', 'day_of_week']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    return X, y, preprocessor, grouping_features # Retornar grouping_features como a lista de features que o modelo espera

def train_keras_model(X_train_processed: np.array, y_train: pd.Series, input_shape: tuple, epochs=50, batch_size=32):
    """
    Define e treina um modelo Keras.
    """
    print("Iniciando treinamento do modelo Keras...")

    # Definir o modelo Keras
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.2), # Regularização para evitar overfitting
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='relu') # Saída para regressão (sales_count >= 0)
    ])

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Mean Squared Error, Mean Absolute Error

    # Treinar o modelo
    history = model.fit(
        X_train_processed, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2, # Usar parte dos dados de treinamento para validação
        verbose=1 # AGORA MOSTRARÁ O PROGRESSO DE CADA ÉPOCA
    )
    print(f"\033[1;32mTreinamento do modelo Keras concluído em {epochs} epochs.\033[0m")
    
    # Obter o valor da última época para a métrica de validação
    val_mae = history.history['val_mae'][-1]
    val_loss = history.history['val_loss'][-1]
    print(f"Última Validação MAE: {val_mae:.2f}")
    print(f"Última Validação Loss (MSE): {val_loss:.2f}")

    return model, history

def evaluate_model(model: keras.Model, X_test_processed: np.array, y_test: pd.Series):
    """
    Avalia o modelo treinado e retorna as métricas.
    """
    print("Avaliando o modelo...")
    y_pred = model.predict(X_test_processed).flatten()
    
    # Garante que as previsões não sejam negativas
    y_pred[y_pred < 0] = 0

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"MSE (Mean Squared Error): {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2_score": r2, "y_pred": y_pred}

def predict_top_car_for_month(
    model: keras.Model,
    preprocessor: ColumnTransformer,
    features_list: list, # A lista de features usadas para treinar o modelo
    all_models: list, # Lista de todos os modelos de carro conhecidos
    all_stores: list, # Lista de todas as lojas conhecidas
    target_month: int,
    target_day_of_week: int = 2 # Exemplo: assume uma quarta-feira para a predição
):
    """
    Prevê o carro com maior propensão de vendas para um mês específico.
    Cria um dataset sintético para o mês e preve para todos os modelos e lojas.
    """
    print(f"Prevenindo o carro de maior propensão para o mês: {target_month} (usando quarta-feira {target_day_of_week})...")
    
    prediction_data = []
    for model_name in all_models:
        for store_id in all_stores:
            prediction_data.append({
                'store_id': store_id,
                'model': model_name,
                'month': target_month,
                'day_of_week': target_day_of_week
            })
    
    synthetic_df = pd.DataFrame(prediction_data)
    
    # Garante que o DataFrame de entrada para o preprocessor tenha as mesmas colunas na mesma ordem
    synthetic_df = synthetic_df[features_list] # Use features_list para garantir a ordem e seleção
    
    X_synthetic_processed = preprocessor.transform(synthetic_df)
    
    predictions = model.predict(X_synthetic_processed).flatten()
    predictions[predictions < 0] = 0

    synthetic_df['predicted_sales'] = predictions

    # Agrupar por modelo para encontrar a soma de vendas preditas em todas as lojas para este mês
    model_predictions = synthetic_df.groupby('model')['predicted_sales'].sum().reset_index()
    
    top_car = model_predictions.loc[model_predictions['predicted_sales'].idxmax()]
    
    print(f"\033[1;36mCarro com maior propensão de vendas para o mês {target_month}: {top_car['model']} (Previsão Total: {top_car['predicted_sales']:.2f}).\033[0m")
    
    return top_car, model_predictions

def generate_ml_report(
    metrics: dict,
    top_car_prediction: pd.Series,
    all_model_predictions: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.array,
    target_month_for_prediction: int, # Adicionado para incluir no relatório
    report_filename="ml_model_report.md",
    plot_filename="ml_model_predictions_plot.png"
) -> tuple[str, str]:
    """
    Gera um relatório Markdown e um gráfico PNG de desempenho do modelo e previsões.
    Retorna os caminhos dos arquivos gerados.
    """
    os.makedirs(TEMP_ARTIFACTS_DIR, exist_ok=True)
    report_path = os.path.join(TEMP_ARTIFACTS_DIR, report_filename)
    plot_path = os.path.join(TEMP_ARTIFACTS_DIR, plot_filename)

    # --- Gerar Relatório Markdown ---
    print(f"Gerando relatório Markdown em: '{report_path}'...")
    with open(report_path, "w") as f:
        f.write(f"# Relatório do Modelo de Previsão de Vendas Chevrolet ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
        f.write("## 1. Métricas de Avaliação do Modelo\n")
        f.write(f"- **MAE (Mean Absolute Error):** `{metrics['mae']:.2f}`\n")
        f.write(f"- **MSE (Mean Squared Error):** `{metrics['mse']:.2f}`\n")
        f.write(f"- **RMSE (Root Mean Squared Error):** `{metrics['rmse']:.2f}`\n")
        f.write(f"- **R2 Score:** `{metrics['r2_score']:.2f}`\n\n")

        f.write(f"## 2. Previsão de Carro de Maior Propensão para o Mês {target_month_for_prediction}\n")
        f.write(f"Para o mês de previsão (`{target_month_for_prediction}`), o carro com maior propensão de vendas é:\n")
        f.write(f"- **Modelo:** `{top_car_prediction['model']}`\n")
        f.write(f"- **Previsão de Vendas (Total para o mês):** `{top_car_prediction['predicted_sales']:.2f}`\n\n")
        
        f.write("### Top 5 Previsões de Modelos para o Mês\n")
        top_5_pred = all_model_predictions.nlargest(5, 'predicted_sales')
        f.write(top_5_pred.to_markdown(index=False))
        f.write("\n\n")


    # --- Gerar Gráfico PNG ---
    print(f"Gerando gráfico de previsões em: '{plot_path}'...")
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, label='Dados de Teste')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha Ideal (y_pred = y_test)')
    plt.title('Vendas Reais vs. Previsões do Modelo', fontsize=16)
    plt.xlabel('Vendas Reais', fontsize=12)
    plt.ylabel('Vendas Previstas', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\033[1;32mRelatório Markdown salvo em: '{report_path}'.\033[0m")
    print(f"\033[1;32mGráfico de Previsões salvo em: '{plot_path}'.\033[0m")
    return report_path, plot_path

def upload_artifact_to_api(local_filepath: str, upload_url: str, mime_type: str = 'application/octet-stream'):
    """
    Faz o upload de um arquivo local para a API de Storage de Artefatos.
    """
    print(f"Fazendo upload do artefato '{os.path.basename(local_filepath)}' para a Storage API: {upload_url}")
    if not os.path.exists(local_filepath):
        print(f"\033[1;31mERRO: Arquivo local '{local_filepath}' não encontrado para upload.\033[0m")
        return None

    try:
        with open(local_filepath, 'rb') as f:
            files = {'file': (os.path.basename(local_filepath), f, mime_type)}
            response = requests.post(upload_url, files=files)
            response.raise_for_status() # Levanta um HTTPError para respostas de erro
            
            upload_info = response.json()
            print(f"\033[1;32mSucesso:\033[0m Artefato carregado. Resposta da API: {upload_info}")
            return upload_info.get("access_url") # Retorna a URL de acesso do artefato na API de storage
    except requests.exceptions.ConnectionError:
        print(f"\033[1;31mERRO: Não foi possível conectar à API de Storage de Artefatos em {upload_url}.\033[0m")
        print("Certifique-se de que sua API de Storage de Artefatos (api_artifact_storage.py) esteja rodando na porta 8779.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"\033[1;31mERRO HTTP ao fazer upload para a API de Storage de Artefatos: {e}.\033[0m Resposta: {e.response.text}")
        raise
    except Exception as e:
        print(f"\033[1;31mOcorreu um erro inesperado ao fazer upload do artefato: {e}\033[0m")
        raise

# --- MLflow Pipeline Principal ---

def run_mlflow_model_training_pipeline(
    test_month_for_prediction: int = 10 # Mês para o qual queremos prever a propensão (ex: 10 para Outubro)
):
    """
    Orquestra o pipeline completo de treinamento de modelo de ML, avaliação,
    predição e geração de relatórios usando MLflow para rastreamento.
    """
    # Inicia um novo run do MLflow
    with mlflow.start_run(run_name="Chevrolet_Sales_Prediction_Model_Training", tags={"mlflow_pipeline_name": "SalesModelTraining"}):
        mlflow.log_param("data_lake_api_url", DATALAKE_API_URL)
        mlflow.log_param("artifact_storage_api_url", ARTIFACT_STORAGE_UPLOAD_URL)
        mlflow.log_param("prediction_month", test_month_for_prediction)

        print("\n" + "="*80)
        print("         Iniciando MLflow Pipeline: Treinamento e Predição de Vendas Chevrolet         ")
        print("="*80 + "\n")

        try:
            # 1. Buscar dados da API de Data Lake
            sales_df = fetch_data(DATALAKE_API_URL)
            if sales_df.empty:
                raise ValueError("Nenhum dado de vendas foi buscado da Data Lake API. Abortando pipeline.")
            mlflow.log_metric("num_records_fetched", len(sales_df))

            # 2. Pré-processar dados para ML (agora com agregação)
            X, y, preprocessor, features_for_model = preprocess_data_for_ml(sales_df) 

            # Obter todos os modelos e lojas únicos para uso na predição (do DF AGRUPADO X)
            all_car_models = X['model'].unique().tolist()
            all_store_ids = X['store_id'].unique().tolist()

            # Dividir os dados em conjuntos de treinamento e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Ajustar o pré-processador APENAS nos dados de treinamento e transformar
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # 3. Treinar o Modelo Keras
            epochs = 50
            batch_size = 32
            model, history = train_keras_model(
                X_train_processed, y_train,
                input_shape=(X_train_processed.shape[1],),
                epochs=epochs,
                batch_size=batch_size
            )
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # 4. Avaliar o Modelo
            eval_metrics = evaluate_model(model, X_test_processed, y_test)
            for metric_name, value in eval_metrics.items():
                if metric_name != "y_pred": # Não logar o array y_pred diretamente como métrica
                    mlflow.log_metric(f"test_{metric_name}", value)

            # 5. Registrar o Modelo Completo (Pré-processador + Modelo Keras) no MLflow como pyfunc
            os.makedirs(TEMP_ARTIFACTS_DIR, exist_ok=True)
            preprocessor_artifact_path = os.path.join(TEMP_ARTIFACTS_DIR, "preprocessor.joblib")
            
            keras_model_artifact_path = os.path.join(TEMP_ARTIFACTS_DIR, "keras_model.keras") 

            joblib.dump(preprocessor, preprocessor_artifact_path) # Salva o preprocessor
            model.save(keras_model_artifact_path) # Salva o modelo Keras no formato nativo .keras

            # Definir o ambiente Conda para o modelo customizado
            conda_env = {
                "channels": ["defaults", "conda-forge"],
                "dependencies": [
                    f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "pip",
                    {
                        "pip": [
                            f"mlflow=={mlflow.__version__}", # Fixa a versão do mlflow
                            f"tensorflow=={tf.__version__}", # Fixa a versão do tensorflow
                            f"scikit-learn=={sklearn.__version__}", # Fixa a versão do scikit-learn
                            f"pandas=={pd.__version__}",
                            f"joblib=={joblib.__version__}",
                            f"matplotlib=={matplotlib.__version__}",
                            f"requests=={requests.__version__}"
                        ]
                    },
                ],
                "name": "mlflow-env"
            }
            conda_env_path = os.path.join(TEMP_ARTIFACTS_DIR, "conda.yaml")
            with open(conda_env_path, "w") as f:
                yaml.dump(conda_env, f, default_flow_style=False)

            print("\n\033[1;34mLogando o modelo como artefato e registrando no MLflow Model Registry...\033[0m")
            
            # Loga o modelo como artefato do run e OBTÉM o ModelInfo
            model_info = mlflow.pyfunc.log_model(
                python_model=SalesPredictionModel(), # Instância da sua classe Pyfunc
                artifacts={
                    "preprocessor": preprocessor_artifact_path,
                    "keras_model": keras_model_artifact_path,
                },
                conda_env=conda_env_path,
                artifact_path="sales_prediction_pyfunc_model", # Subpasta no MLflow UI para este modelo
                signature=ModelSignature(
                    inputs=Schema([
                        ColSpec(type=DataType.string, name="store_id"),
                        ColSpec(type=DataType.string, name="model"),
                        ColSpec(type=DataType.integer, name="month"),
                        ColSpec(type=DataType.integer, name="day_of_week")
                    ]),
                    outputs=Schema([
                        ColSpec(type=DataType.double)
                    ])
                )
            )
            
            # AQUI ESTÁ A CORREÇÃO: REGISTRAR O MODELO NO REGISTRY
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri, # Usa o URI do modelo logado
                name="ChevroletSalesPredictor" # Nome que aparecerá na aba "Models"
            )
            
            print(f"\033[1;32mModelo '{registered_model.name}' versão {registered_model.version} registrado no MLflow Model Registry.\033[0m")
            print(f"Você pode ver o modelo em: http://127.0.0.1:5001/#/models/{registered_model.name}/versions/{registered_model.version}")

            # 6. Predição de Carro com Maior Propensão para um Mês Específico
            top_car, all_model_predictions_df = predict_top_car_for_month(
                model,
                preprocessor,
                features_for_model,
                all_car_models,
                all_store_ids,
                test_month_for_prediction
            )
            mlflow.log_param("predicted_top_car_model", top_car['model'])
            mlflow.log_metric("predicted_top_car_sales", top_car['predicted_sales'])

            # 7. Gerar Relatório Markdown e Gráfico PNG
            report_filename = "ml_model_report.md"
            plot_filename = "ml_model_predictions_plot.png"
            report_path, plot_path = generate_ml_report(
                eval_metrics,
                top_car,
                all_model_predictions_df,
                y_test,
                eval_metrics['y_pred'],
                test_month_for_prediction,
                report_filename=report_filename,
                plot_filename=plot_filename
            )

            # Logar o relatório e o gráfico como artefatos no MLflow
            mlflow.log_artifact(report_path, artifact_path="model_report")
            mlflow.log_artifact(plot_path, artifact_path="visualizations")

            # 8. Fazer upload do relatório e do gráfico para a API de Storage de Artefatos
            report_access_url = upload_artifact_to_api(report_path, ARTIFACT_STORAGE_UPLOAD_URL, mime_type='text/markdown')
            plot_access_url = upload_artifact_to_api(plot_path, ARTIFACT_STORAGE_UPLOAD_URL, mime_type='image/png')
            
            if report_access_url:
                mlflow.log_param("model_report_external_url", report_access_url)
            if plot_access_url:
                mlflow.log_param("predictions_plot_external_url", plot_access_url)

            print("\n" + "="*80)
            print("  MLflow Pipeline: Treinamento e Predição de Vendas Chevrolet CONCLUÍDO  ")
            print("="*80 + "\n")

            mlflow.log_param("pipeline_status", "Completed")

        except Exception as e:
            print(f"\033[1;31mMLflow Pipeline FALHOU: {e}\033[0m")
            mlflow.log_param("pipeline_status", "Failed")
            mlflow.log_param("error_message", str(e))
            import traceback
            mlflow.log_text(traceback.format_exc(), "error_trace.txt")
            raise # Re-levanta a exceção para que o Python saia com erro, se executado de fora

        finally:
            # Limpa o diretório de artefatos temporários
            if os.path.exists(TEMP_ARTIFACTS_DIR):
                shutil.rmtree(TEMP_ARTIFACTS_DIR)
                print(f"Diretório temporário '{TEMP_ARTIFACTS_DIR}' limpo.")

if __name__ == "__main__":
    print(f"\nMLflow Tracking UI configurada para: \033[1;34m{MLFLOW_TRACKING_URI}\033[0m")
    target_month_for_prediction = 10 
    run_mlflow_model_training_pipeline(test_month_for_prediction=target_month_for_prediction)