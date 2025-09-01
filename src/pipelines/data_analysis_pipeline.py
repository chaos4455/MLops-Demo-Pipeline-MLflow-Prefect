import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import os
import shutil # Para limpar diretórios temporários

# --- Configurações dos Endpoints e MLflow ---
DATALAKE_API_URL = "http://127.0.0.1:8778/datasource" # Sua API de Data Lake (passiva)
ARTIFACT_STORAGE_UPLOAD_URL = "http://127.0.0.1:8779/upload_artifact" # Sua API de Storage de Artefatos
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001" # Seu MLflow Tracking Server

# Diretório temporário para salvar artefatos localmente antes de logar/upload
TEMP_ARTIFACTS_DIR = "temp_pipeline_artifacts"

# Configura o MLflow Tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Funções do Pipeline ---

def fetch_data(url: str) -> pd.DataFrame:
    """
    Busca os dados de vendas da API de Data Lake.
    """
    print(f"Buscando dados da Data Lake API: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Levanta um HTTPError para respostas de erro (4xx ou 5xx)
        data = response.json()
        
        # --- CORREÇÃO IMPLEMENTADA AQUI ---
        # A API de storage retorna uma LISTA de dicionários diretamente.
        # Portanto, não devemos tentar usar `.get('data')` nela.
        
        if not data: # Se a lista estiver vazia
            print("\033[33mAVISO: Nenhum dado recebido da Data Lake API. Retornando DataFrame vazio.\033[0m")
            return pd.DataFrame()
        
        if not isinstance(data, list):
            # Caso a API mude o formato ou haja um erro, é bom ter um aviso
            print(f"\033[1;31mERRO: Formato de dados inesperado da Data Lake API.\033[0m")
            print(f"Esperava uma lista (JSON array), mas recebeu tipo: {type(data)}. Conteúdo (primeiros 200 chars): {str(data)[:200]}...")
            return pd.DataFrame()

        # Se for uma lista, cria o DataFrame diretamente a partir dela
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

def process_data_for_top_models(df: pd.DataFrame) -> pd.Series:
    """
    Processa o DataFrame para identificar os 10 modelos mais vendidos.
    """
    print("Processando dados para encontrar os 10 modelos mais vendidos...")
    if df.empty:
        print("\033[33mDataFrame vazio, impossível processar.\033[0m")
        return pd.Series()
    
    # Garante que 'sales_count' seja numérico, tratando possíveis erros
    df['sales_count'] = pd.to_numeric(df['sales_count'], errors='coerce').fillna(0)
    
    top_10_models = df.groupby('model')['sales_count'].sum().nlargest(10)
    print("Top 10 modelos calculados:")
    print(top_10_models.to_string())
    return top_10_models

def generate_top_models_plot(top_models_data: pd.Series, plot_filename: str) -> str:
    """
    Gera um gráfico de barras dos 10 modelos mais vendidos e o salva localmente.
    Retorna o caminho do arquivo salvo.
    """
    print(f"Gerando gráfico: {plot_filename}")
    if top_models_data.empty or top_models_data.sum() == 0:
        print("\033[33mNenhum dado de top modelos válido para plotar. Ignorando a geração do gráfico.\033[0m")
        return ""

    plt.figure(figsize=(12, 7))
    top_models_data.sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Modelos Chevrolet Mais Vendidos (2024)', fontsize=16)
    plt.xlabel('Número de Vendas', fontsize=12)
    plt.ylabel('Modelo do Carro', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Salva o gráfico em um arquivo temporário local
    os.makedirs(TEMP_ARTIFACTS_DIR, exist_ok=True)
    local_plot_path = os.path.join(TEMP_ARTIFACTS_DIR, plot_filename)
    plt.savefig(local_plot_path)
    plt.close() # Fecha o plot para liberar memória
    
    print(f"\033[1;32mGráfico salvo localmente em: '{local_plot_path}'.\033[0m")
    return local_plot_path

def upload_artifact_to_api(local_filepath: str, upload_url: str):
    """
    Faz o upload de um arquivo local para a API de Storage de Artefatos.
    """
    print(f"Fazendo upload do artefato '{os.path.basename(local_filepath)}' para a Storage API: {upload_url}")
    if not os.path.exists(local_filepath):
        print(f"\033[1;31mERRO: Arquivo local '{local_filepath}' não encontrado para upload.\033[0m")
        return None

    try:
        with open(local_filepath, 'rb') as f:
            # 'files' espera um tupla: ('nome_do_campo', (nome_do_arquivo, objeto_arquivo_aberto, tipo_mime))
            files = {'file': (os.path.basename(local_filepath), f, 'image/png')}
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

def run_mlflow_pipeline():
    """
    Orquestra o pipeline completo de processamento de dados e geração de artefatos
    usando MLflow para rastreamento.
    """
    # Inicia um novo run do MLflow
    # Adicione a tag 'mlflow_pipeline_name' para identificar este pipeline no MLflow UI
    with mlflow.start_run(run_name="Chevrolet_Top_Sales_Report", tags={"mlflow_pipeline_name": "SalesAnalysis"}):
        mlflow.log_param("data_lake_api_url", DATALAKE_API_URL)
        mlflow.log_param("artifact_storage_api_url", ARTIFACT_STORAGE_UPLOAD_URL)
        
        print("\n" + "="*70)
        print("              Iniciando MLflow Pipeline: Relatório de Vendas Chevrolet             ")
        print("="*70 + "\n")

        try:
            # 1. Buscar dados da API de Data Lake
            sales_df = fetch_data(DATALAKE_API_URL)
            if sales_df.empty:
                raise ValueError("Nenhum dado de vendas foi buscado da Data Lake API. Abortando pipeline.")
            mlflow.log_metric("num_records_fetched", len(sales_df))

            # 2. Processar dados para encontrar os 10 modelos mais vendidos
            top_10_models = process_data_for_top_models(sales_df)
            if top_10_models.empty:
                raise ValueError("Nenhum Top 10 modelos encontrado após o processamento. Abortando pipeline.")
            mlflow.log_metric("num_top_models_found", len(top_10_models))
            
            # Logar os top modelos como um artefato de texto no MLflow
            os.makedirs(TEMP_ARTIFACTS_DIR, exist_ok=True) # Garante que o diretório temp exista
            top_models_csv_path = os.path.join(TEMP_ARTIFACTS_DIR, "top_10_models.csv")
            top_10_models.to_csv(top_models_csv_path)
            mlflow.log_artifact(top_models_csv_path, artifact_path="processed_data")
            
            # 3. Gerar o gráfico dos top 10 modelos e salvá-lo localmente
            plot_filename = "top_10_chevrolet_sales_plot.png"
            local_plot_path = generate_top_models_plot(top_10_models, plot_filename)
            
            if not local_plot_path:
                print("\033[33mAVISO: O gráfico não foi gerado. Pulando upload e log.\033[0m")
            else:
                # Logar o gráfico como um artefato no MLflow
                mlflow.log_artifact(local_plot_path, artifact_path="visualizations")
                    
                # 4. Fazer upload do gráfico para a API de Storage de Artefatos
                artifact_access_url = upload_artifact_to_api(local_plot_path, ARTIFACT_STORAGE_UPLOAD_URL)
                if artifact_access_url:
                    mlflow.log_param("top_sales_plot_external_url", artifact_access_url)
                    print(f"URL externa do gráfico logada no MLflow: {artifact_access_url}")
                else:
                    print("\033[33mAVISO: Falha ao obter a URL externa do gráfico da API de Storage.\033[0m")

            print("\n" + "="*70)
            print("       MLflow Pipeline: Relatório de Vendas Chevrolet CONCLUÍDO       ")
            print("="*70 + "\n")

            mlflow.log_param("status", "Completed")

        except Exception as e:
            print(f"\033[1;31mMLflow Pipeline FALHOU: {e}\033[0m")
            mlflow.log_param("status", "Failed")
            mlflow.log_param("error_message", str(e))
            # Opcional: registrar detalhes do erro para depuração
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
    run_mlflow_pipeline()