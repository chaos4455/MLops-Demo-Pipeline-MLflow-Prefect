from prefect import flow, task
import requests
import pandas as pd
import os
import time

# --- Configurações dos Endpoints ---
SALES_API_URL = "http://127.0.0.1:8777/sales_data" # Sua API de vendas original
STORAGE_API_URL = "http://127.0.0.1:8778/store_data" # O novo endpoint POST da API de storage

@task(name="Buscar Dados da API de Vendas")
def fetch_data_from_sales_api(api_url: str) -> list:
    """
    Tarefa Prefect para buscar dados da API de vendas original.
    Retorna uma lista de dicionários (JSON).
    """
    print(f"Buscando dados da API de vendas: {api_url}")
    try:
        response = requests.get(api_url)
        response.raise_for_status() # Levanta um HTTPError para respostas de erro
        data = response.json()
        print(f"\033[1;32mSucesso:\033[0m {len(data)} registros recebidos da API de vendas.")
        return data
    except requests.exceptions.ConnectionError:
        print(f"\033[1;31mERRO: Não foi possível conectar à API de vendas em {api_url}.\033[0m")
        print("Certifique-se de que sua API Flask (api_vendas.py) esteja rodando na porta 8777.")
        raise # Re-levanta a exceção para que o Prefect marque a tarefa como falha
    except requests.exceptions.HTTPError as e:
        print(f"\033[1;31mERRO HTTP ao buscar dados da API de vendas: {e}.\033[0m Resposta: {e.response.text}")
        raise
    except Exception as e:
        print(f"\033[1;31mERRO inesperado ao buscar dados da API de vendas: {e}\033[0m")
        raise

@task(name="Enviar Dados para API de Storage")
def push_data_to_storage_api(data: list, storage_api_url: str):
    """
    Tarefa Prefect para enviar os dados para a API de storage.
    """
    if not data:
        print("\033[33mAVISO: Nenhum dado para enviar para a API de storage. Pulando push.\033[0m")
        return

    print(f"Enviando {len(data)} registros para a API de storage: {storage_api_url}")
    try:
        response = requests.post(storage_api_url, json=data) # Envia como JSON
        response.raise_for_status() # Levanta um HTTPError para respostas de erro
        print(f"\033[1;32mSucesso:\033[0m Dados enviados para a API de storage. Resposta: {response.json()}")
    except requests.exceptions.ConnectionError:
        print(f"\033[1;31mERRO: Não foi possível conectar à API de storage em {storage_api_url}.\033[0m")
        print("Certifique-se de que sua API de Storage (api_datasource_storage_passive.py) esteja rodando na porta 8778.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"\033[1;31mERRO HTTP ao enviar dados para a API de storage: {e}.\033[0m Resposta: {e.response.text}")
        raise
    except Exception as e:
        print(f"\033[1;31mERRO inesperado ao enviar dados para a API de storage: {e}\033[0m")
        raise

@flow(name="Data Ingestion Flow - Chevrolet Sales", log_prints=True)
def data_ingestion_flow():
    """
    Flow Prefect para orquestrar a ingestão de dados da API de vendas
    para a API de storage passiva.
    """
    print("╔═════════════════════════════════════════════════════════╗")
    print("║        Iniciando Prefect Data Ingestion Flow            ║")
    print("╚═════════════════════════════════════════════════════════╝")
    
    # 1. Buscar dados da API de vendas
    raw_sales_data = fetch_data_from_sales_api(SALES_API_URL)
    
    # 2. Enviar os dados para a API de storage
    if raw_sales_data:
        push_data_to_storage_api(raw_sales_data, STORAGE_API_URL)
    else:
        print("\033[33mNenhum dado recebido da API de vendas. Nenhuma ação de push realizada.\033[0m")

    print("\nPrefect Data Ingestion Flow concluído.")

if __name__ == "__main__":
    # Para executar este flow e registrá-lo no Prefect Server:
    # 1. Certifique-se de que o Prefect Server (`start_prefect_server.py`) esteja rodando.
    # 2. Execute este script: `python data_ingestion_flow.py`
    # Isso registrará o flow no servidor e o executará imediatamente.
    # Você poderá ver o run na UI do Prefect (http://localhost:4200).
    data_ingestion_flow()