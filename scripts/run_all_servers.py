import subprocess
import os
import time
import sys

# Define o diretório raiz do projeto. Este script está em 'scripts/', então subimos um nível.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_command_in_new_terminal(command_parts, name, port=None):
    """Executa um comando Python em um novo terminal dependendo do SO."""
    # Garante que o interpretador Python correto seja usado
    python_executable = sys.executable

    print(f"Iniciando {{name}} (porta: {{port if port else 'N/A'}}) em novo terminal...")

    # Construir o comando a ser executado
    full_command_list = [python_executable] + command_parts
    
    if sys.platform.startswith('win'):
        # No Windows, usamos 'cmd /k' para abrir um novo terminal e manter a janela aberta.
        # Usamos subprocess.CREATE_NEW_CONSOLE para garantir uma nova janela de console.
        # A string de comando é construída para 'cmd /k' poder executá-la.
        # `cd /d` muda de diretório de forma confiável no Windows.
        # list2cmdline é usado para converter a lista de partes do comando Python em uma única string
        command_str_for_cmd = f'cd /d "{PROJECT_ROOT}" && {subprocess.list2cmdline(full_command_list)}'
        print(f"  Windows: cmd /k '{{command_str_for_cmd}}'")
        try:
            subprocess.Popen(
                ['cmd', '/k', command_str_for_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=PROJECT_ROOT # Define o diretório de trabalho, embora o 'cd /d' já faça isso
            )
        except Exception as e:
            print(f"  ERRO ao iniciar {{name}} no Windows: {{e}}")
            print(f"  Tente executar manualmente: cd /d "{{PROJECT_ROOT}}" && {subprocess.list2cmdline(full_command_list)}")

    elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        # No Linux/macOS, tenta diferentes emuladores de terminal
        terminal_commands_options = [
            ['gnome-terminal', '--'],
            ['konsole', '-e'],
            ['xterm', '-e']
        ]
        
        launched = False
        for term_cmd_prefix in terminal_commands_options:
            try:
                print(f"  Linux/macOS (tentando {{term_cmd_prefix[0]}}): {{' '.join(term_cmd_prefix + full_command_list)}}")
                # Popen espera uma lista de argumentos para o comando e seus parâmetros.
                # full_command_list já é [python_executable, script_path, ...]
                subprocess.Popen(term_cmd_prefix + full_command_list, cwd=PROJECT_ROOT)
                launched = True
                break
            except FileNotFoundError:
                continue
        
        if not launched:
            print(f"  AVISO: Nenhum emulador de terminal gráfico comum encontrado para {{name}}.")
            print(f"  Tente abrir um novo terminal manualmente e execute:")
            print(f"  cd "{{PROJECT_ROOT}}" && {subprocess.list2cmdline(full_command_list)}")
    else:
        print(f"ERRO: Sistema operacional não suportado para iniciar {{name}} em novo terminal.")
        print(f"Por favor, execute manualmente: cd "{{PROJECT_ROOT}}" && {subprocess.list2cmdline(full_command_list)}")

def main():
    print("╔═════════════════════════════════════════════════════════╗")
    print("║         Iniciando TODOS os Servidores e APIs            ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print("\nCertifique-se de que o ambiente Python esteja ativado e as dependências instaladas.")
    print("Os logs de cada serviço aparecerão em suas respectivas janelas de terminal.")
    print("Pressione Ctrl+C em cada terminal para encerrar os serviços individualmente.")
    print("-" * 70)

    # Definir scripts com seus caminhos relativos à raiz do projeto
    mlflow_server_script = os.path.join("scripts", "start_mlflow_server.py")
    prefect_server_script = os.path.join("scripts", "start_prefect_server.py")
    
    data_gen_api_script = os.path.join("src", "api", "data_generator", "app.py")
    data_lake_api_script = os.path.join("src", "api", "data_lake", "app.py")
    artifact_store_api_script = os.path.join("src", "api", "artifact_store", "app.py")
    model_serving_api_script = os.path.join("src", "api", "model_serving", "app.py")

    # Iniciar MLflow Server
    run_command_in_new_terminal([mlflow_server_script], "MLflow Tracking Server", port=5001)
    time.sleep(5) # Dar tempo para o MLflow iniciar

    # Iniciar Prefect Server
    run_command_in_new_terminal([prefect_server_script], "Prefect Server", port=4200)
    time.sleep(5) # Dar tempo para o Prefect iniciar

    # Iniciar API de Geração de Dados
    run_command_in_new_terminal([data_gen_api_script], "Data Generator API", port=8777)
    time.sleep(2)

    # Iniciar API de Data Lake
    run_command_in_new_terminal([data_lake_api_script], "Data Lake API", port=8778)
    time.sleep(2)

    # Iniciar API de Storage de Artefatos
    run_command_in_new_terminal([artifact_store_api_script], "Artifact Store API", port=8779)
    time.sleep(2)

    # Iniciar API de Serviço de Modelo
    run_command_in_new_terminal([model_serving_api_script], "Model Serving API", port=8780)
    time.sleep(2)

    print("\n" + "="*70)
    print("Todos os serviços foram iniciados (verifique os terminais abertos).")
    print("Agora você pode executar os Prefect Flows e MLflow Pipelines em novos terminais.")
    print("Ex: python src/flows/data_ingestion_flow.py")
    print("Ex: python src/pipelines/data_analysis_pipeline.py")
    print("Ex: python src/pipelines/model_training_pipeline.py")
    print("Ex: python scripts/run_model_tester.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()