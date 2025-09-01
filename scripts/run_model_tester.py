# NOME DO ARQUIVO: ML-FLOW-MODEL-TESTER.PY
# TAMANHO: Aproximadamente 4.0 KB
# NÚMERO DE LINHAS: Aproximadamente 120

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# --- Configurações do Tester ---
MODEL_API_URL = "http://127.0.0.1:8780/predict" # URL do endpoint de previsão da sua API de modelo
MODEL_STATUS_URL = "http://127.0.0.1:8780/model_status" # URL do endpoint de status da sua API de modelo
TEST_OUTPUT_DIR = "reports/model_test" # Diretório para salvar os relatórios de teste

# --- Dados de Referência para Geração de Teste (do seu gerador original) ---
# Usamos uma lista pequena para evitar sobrecarga, mas pode ser expandida
ALL_STORE_IDS = [f"Loja_{i:02d}" for i in range(1, 21)] # De Loja_01 a Loja_20
ALL_MODELS = ["Onix", "Onix Plus", "Tracker", "Montana", "Spin", "S10", "Equinox", "Cobalt"]
ALL_MONTHS = list(range(1, 13)) # De Janeiro (1) a Dezembro (12)
ALL_DAYS_OF_WEEK = list(range(7)) # Segunda (0) a Domingo (6)

# --- Funções Auxiliares ---

def check_model_api_status(status_url: str) -> bool:
    """
    Verifica o status da API de serviço do modelo.
    """
    print(f"\nVerificando status da API de Modelo em: {status_url}")
    try:
        response = requests.get(status_url)
        response.raise_for_status()
        status_info = response.json()
        if status_info.get("model_loaded"):
            print(f"\033[1;32mSucesso:\033[0m API de Modelo OK. Modelo '{status_info.get('model_name')}' versão '{status_info.get('model_version')}' carregado.")
            return True
        else:
            print(f"\033[1;31mERRO:\033[0m API de Modelo respondendo, mas modelo não carregado. Mensagem: {status_info.get('message')}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"\033[1;31mERRO:\033[0m Não foi possível conectar à API de Modelo em {status_url}.")
        print("Certifique-se de que a API (ML-FLOW-MODEL-API-SERVING.PY) esteja rodando na porta 8780.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"\033[1;31mERRO HTTP:\033[0m ao verificar status da API de Modelo: {e}. Resposta: {e.response.text}")
        return False
    except Exception as e:
        print(f"\033[1;31mERRO inesperado:\033[0m ao verificar status da API de Modelo: {e}")
        return False

def generate_synthetic_test_data(num_samples: int = 50) -> pd.DataFrame:
    """
    Gera um DataFrame com dados de entrada sintéticos para testar o modelo.
    """
    print(f"Gerando {num_samples} amostras de dados sintéticos para teste...")
    test_data = []
    for _ in range(num_samples):
        test_data.append({
            "store_id": np.random.choice(ALL_STORE_IDS),
            "model": np.random.choice(ALL_MODELS),
            "month": np.random.choice(ALL_MONTHS),
            "day_of_week": np.random.choice(ALL_DAYS_OF_WEEK)
        })
    df_test = pd.DataFrame(test_data)
    
    # --- CORREÇÃO: Converter tipos de dados para int32 ---
    # O MLflow signature esperava int32 para 'month' e 'day_of_week'.
    # Pandas pode usar int64 por padrão, causando um erro de schema enforcement.
    df_test['month'] = df_test['month'].astype(np.int32)
    df_test['day_of_week'] = df_test['day_of_week'].astype(np.int32)
    # ---------------------------------------------------

    print(f"\033[1;32mSucesso:\033[0m Dados sintéticos gerados e tipos ajustados.")
    return df_test

def get_predictions_from_api(data_df: pd.DataFrame, api_url: str) -> list:
    """
    Envia os dados sintéticos para a API de modelo e retorna as previsões.
    """
    print(f"Enviando {len(data_df)} amostras para a API de Modelo em: {api_url}...")
    headers = {'Content-Type': 'application/json'}
    payload = data_df.to_dict(orient='records') # Os dados já estarão com int32 por causa da função generate_synthetic_test_data

    try:
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        predictions = response.json().get('predictions', [])
        print(f"\033[1;32mSucesso:\033[0m {len(predictions)} previsões recebidas da API.")
        return predictions
    except requests.exceptions.ConnectionError:
        print(f"\033[1;31mERRO:\033[0m Não foi possível conectar à API de Modelo em {api_url}.")
        print("Certifique-se de que a API (ML-FLOW-MODEL-API-SERVING.PY) esteja rodando na porta 8780.")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"\033[1;31mERRO HTTP:\033[0m ao obter previsões da API de Modelo: {e}. Resposta: {e.response.text}")
        raise
    except Exception as e:
        print(f"\033[1;31mERRO inesperado:\033[0m ao obter previsões da API de Modelo: {e}")
        raise

def generate_prediction_plot(
    data_df: pd.DataFrame,
    predictions: list,
    output_path: str,
    plot_filename="model_predictions_overview.png"
) -> str:
    """
    Gera um gráfico com a distribuição das previsões e salva localmente.
    """
    print(f"Gerando gráfico de previsões em: '{output_path}'...")
    if not predictions:
        print("\033[33mAVISO: Nenhuma previsão para plotar. Pulando a geração do gráfico.\033[0m")
        return ""

    predictions_series = pd.Series(predictions)
    
    # Adicionar previsões ao DataFrame para análise mais fácil
    data_df_with_predictions = data_df.copy()
    data_df_with_predictions['predicted_sales'] = predictions_series.round(2)

    # Plot 1: Histograma da distribuição das previsões
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].hist(predictions_series, bins=20, edgecolor='black', color='lightgreen')
    axes[0].set_title('Distribuição das Previsões de Vendas', fontsize=14)
    axes[0].set_xlabel('Vendas Previstas', fontsize=12)
    axes[0].set_ylabel('Frequência', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Top N Modelos mais previstos em média
    avg_predictions_by_model = data_df_with_predictions.groupby('model')['predicted_sales'].mean().nlargest(10)
    avg_predictions_by_model.sort_values(ascending=True).plot(kind='barh', ax=axes[1], color='lightcoral')
    axes[1].set_title('Top 10 Modelos com Média de Vendas Previstas (Aleatório)', fontsize=14)
    axes[1].set_xlabel('Média de Vendas Previstas', fontsize=12)
    axes[1].set_ylabel('Modelo do Carro', fontsize=12)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    plot_filepath = os.path.join(output_path, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    print(f"\033[1;32mSucesso:\033[0m Gráfico salvo em: '{plot_filepath}'.\033[0m")
    return plot_filepath

def generate_test_report(
    data_df: pd.DataFrame,
    predictions: list,
    output_path: str,
    report_filename="model_test_report.md"
) -> str:
    """
    Gera um relatório Markdown com o resumo do teste e salva localmente.
    """
    print(f"Gerando relatório Markdown em: '{output_path}'...")
    os.makedirs(output_path, exist_ok=True)
    report_filepath = os.path.join(output_path, report_filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_filepath, "w") as f:
        f.write(f"# Relatório de Teste da API de Modelo MLflow ({timestamp})\n\n")
        f.write("Este relatório resume o teste de consumo da API de modelo MLflow para previsão de vendas Chevrolet.\n\n")
        f.write("## 1. Configurações do Teste\n")
        f.write(f"- **URL da API de Modelo:** `{MODEL_API_URL}`\n")
        f.write(f"- **Número de Amostras Testadas:** `{len(data_df)}`\n\n")

        f.write("## 2. Estatísticas das Previsões\n")
        if predictions:
            predictions_series = pd.Series(predictions)
            f.write(f"- **Mínimo de Vendas Previstas:** `{predictions_series.min():.2f}`\n")
            f.write(f"- **Máximo de Vendas Previstas:** `{predictions_series.max():.2f}`\n")
            f.write(f"- **Média de Vendas Previstas:** `{predictions_series.mean():.2f}`\n")
            f.write(f"- **Desvio Padrão das Vendas Previstas:** `{predictions_series.std():.2f}`\n\n")
        else:
            f.write("- Nenhuma previsão recebida para calcular estatísticas.\n\n")

        f.write("## 3. Amostra de Previsões\n")
        if predictions:
            # Combinar entradas com previsões para uma amostra
            sample_data = data_df.head(5).copy()
            sample_data['predicted_sales'] = pd.Series(predictions[:5]).round(2)
            f.write("Aqui estão as primeiras 5 amostras de entrada com suas respectivas previsões:\n")
            f.write(sample_data.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("- Nenhuma previsão recebida para exibir amostra.\n\n")

        f.write("## 4. Visualização\n")
        f.write(f"Um gráfico da distribuição das previsões e média por modelo foi gerado:\n")
        f.write(f"![Gráfico de Previsões]({os.path.basename(report_filepath).replace('.md', '_overview.png')})\n\n")
        f.write("---")

    print(f"\033[1;32mSucesso:\033[0m Relatório salvo em: '{report_filepath}'.\033[0m")
    return report_filepath

# --- Pipeline Principal de Teste ---

def run_model_tester_pipeline():
    """
    Orquestra o pipeline completo de teste do modelo via API.
    """
    print("╔═════════════════════════════════════════════════════════╗")
    print("║          Iniciando MLflow Model Tester Pipeline         ║")
    print("╚═════════════════════════════════════════════════════════╝")

    try:
        # 0. Verificar o status da API de modelo
        if not check_model_api_status(MODEL_STATUS_URL):
            print("\033[1;31mFalha crítica:\033[0m A API de modelo não está pronta ou o modelo não foi carregado. Abortando teste.")
            return

        # 1. Gerar dados sintéticos para teste
        test_input_df = generate_synthetic_test_data(num_samples=100) # Gerar 100 amostras

        # 2. Obter previsões da API de modelo
        predictions = get_predictions_from_api(test_input_df, MODEL_API_URL)

        # 3. Gerar e salvar o gráfico de previsões
        plot_filepath = generate_prediction_plot(test_input_df, predictions, TEST_OUTPUT_DIR)

        # 4. Gerar e salvar o relatório Markdown
        report_filepath = generate_test_report(test_input_df, predictions, TEST_OUTPUT_DIR)

        print("\n" + "="*80)
        print("          MLflow Model Tester Pipeline CONCLUÍDO          ")
        print(f"Relatórios disponíveis em: \033[1;34m{os.path.abspath(TEST_OUTPUT_DIR)}\033[0m")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\033[1;31mMLflow Model Tester Pipeline FALHOU: {e}\033[0m")
        import traceback
        traceback.print_exc() # Imprime o stack trace para depuração
        print("\033[1;31mVerifique se todas as APIs de dependência estão rodando e acessíveis.\033[0m")

if __name__ == "__main__":
    run_model_tester_pipeline()