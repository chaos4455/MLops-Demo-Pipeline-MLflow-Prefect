import os
import shutil
import re
import fileinput
import sys
import subprocess # Usado para subprocess.CREATE_NEW_CONSOLE e list2cmdline

# Define o caminho base do projeto como o diretório onde este script está sendo executado.
# Certifique-se de executar este script a partir da raiz do seu projeto.
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------------------------------------
# 1. Definição da Nova Estrutura de Diretórios
# -------------------------------------------------------------------------------------------------
NEW_DIRS = [
    "scripts",
    "src",
    "src/api",
    "src/api/data_generator",
    "src/api/data_lake",
    "src/api/artifact_store",
    "src/api/model_serving",
    "src/flows",
    "src/pipelines",
    "data",
    "artifacts/storage",
    "mlruns", # Será o local padrão para MLflow Tracking e artefatos
    "reports/model_test",
    "docs"
]

# -------------------------------------------------------------------------------------------------
# 2. Mapeamento de Arquivos Antigos para Novos Caminhos e Nomes
# -------------------------------------------------------------------------------------------------
FILE_MAPPING = {
    # APIs
    "API-DATA-SERVER-V1.PY": "src/api/data_generator/app.py",
    "API-OBJECT-STORAGE-SERVER-V1.PY": "src/api/data_lake/app.py",
    "API-SERVER-DATA-STORAGE-OUTPUT-V1.PY": "src/api/artifact_store/app.py",
    "API-SERVER-MODEL-LOAD-FROM-ML-FLOW--PREDICT-SERVICE-V1.PY": "src/api/model_serving/app.py",

    # Prefect Flows
    "PREFECT-DATA-INJECTION-FLOW-V1.PY": "src/flows/data_ingestion_flow.py",

    # MLflow Pipelines
    "ML-FLOW-DATA-PIPELINE-GRAPH-GEN-V1.PY": "src/pipelines/data_analysis_pipeline.py",
    "ML-FLOW-DATA-PIPELINE-MODEL-TRAINER-GRAPH-REPORT-EXPORTER.PY": "src/pipelines/model_training_pipeline.py",

    # Server/Tester Scripts
    "ML-FLOW-SERVER-V1.PY": "scripts/start_mlflow_server.py",
    "PREFECT-SERVER-V1.PY": "scripts/start_prefect_server.py",
    "API-SERVER-MODEL-TESTER-V1.PY": "scripts/run_model_tester.py",

    # Documentação
    "PROJECT-DOCUMENTATION-V1.HTML": "docs/project_documentation.html",
    
    # Arquivos a serem ignorados ou deletados (defina como None)
    "lista-diretorio-projeto.py": None 
}

# -------------------------------------------------------------------------------------------------
# 3. Replacements de Conteúdo em Arquivos (regex_antigo, string_nova, regex_para_caminho_novo_do_arquivo)
# -------------------------------------------------------------------------------------------------
CONTENT_REPLACEMENTS = [
    # src/api/data_lake/app.py
    (r'DATABASE_DIR = "data"', r'DATABASE_DIR = "../../data"', r'src/api/data_lake/app\.py'),
    # src/api/artifact_store/app.py
    (r'ARTIFACTS_DIR = "artifacts_storage"', r'ARTIFACTS_DIR = "../../artifacts/storage"', r'src/api/artifact_store/app\.py'),
    # scripts/start_mlflow_server.py
    (r'backend_uri="sqlite:///mlruns.db"', r'backend_uri="sqlite:///data/mlflow_backend.db"', r'scripts/start_mlflow_server\.py'),
    (r'artifact_root="./mlartifacts"', r'artifact_root="./mlruns"', r'scripts/start_mlflow_server\.py'),
    # scripts/run_model_tester.py
    (r'TEST_OUTPUT_DIR = "test_data_reports"', r'TEST_OUTPUT_DIR = "reports/model_test"', r'scripts/run_model_tester\.py'),
    # src/pipelines/data_analysis_pipeline.py e src/pipelines/model_training_pipeline.py
    # O diretório temporário será unificado para evitar confusão.
    (r'TEMP_ARTIFACTS_DIR = "temp_mlflow_artifacts"', r'TEMP_ARTIFACTS_DIR = "temp_pipeline_artifacts"', r'src/pipelines/data_analysis_pipeline\.py'),
    (r'TEMP_ARTIFACTS_DIR = "temp_mlflow_model_artifacts"', r'TEMP_ARTIFACTS_DIR = "temp_pipeline_artifacts"', r'src/pipelines/model_training_pipeline\.py'),
]

# -------------------------------------------------------------------------------------------------
# 4. Definição de Movimentações de Conteúdo de Diretórios Antigos para Novos
#    NOTA: Removido "data" -> "data" para evitar auto-movimentação problemática.
# -------------------------------------------------------------------------------------------------
# (old_path_relative_to_base, new_path_relative_to_base)
CONTENT_MOVES_DIRS = [
    ("artifacts_storage", "artifacts/storage"),
    ("test_data_reports", "reports/model_test"),
]

# Diretórios relacionados ao MLflow que precisam ser consolidados na nova pasta 'mlruns'
MLFLOW_OLD_DIRS_TO_CONSOLIDATE = ["mlartifacts", "mlruns"] # O 'mlruns' antigo será mesclado no novo

# -------------------------------------------------------------------------------------------------
# Funções Auxiliares para o Logging
# -------------------------------------------------------------------------------------------------
def print_status(message, color_code="0"):
    """Imprime uma mensagem colorida no console."""
    colors = {
        "0": "\033[0m",   # Reset
        "32": "\033[1;32m", # Green
        "33": "\033[1;33m", # Yellow
        "31": "\033[1;31m", # Red
        "34": "\033[1;34m", # Blue
        "36": "\033[1;36m", # Cyan
    }
    print(f"{colors.get(color_code, colors['0'])}{message}{colors['0']}")

# -------------------------------------------------------------------------------------------------
# Funções Auxiliares para a Refatoração
# -------------------------------------------------------------------------------------------------

def _merge_directories_content(src_dir, dst_dir):
    """
    Mescla o conteúdo de src_dir em dst_dir recursivamente.
    Se um item existe em ambos e é um diretório, mescla. Se é um arquivo, move/sobrescreve.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    for item in os.listdir(src_dir):
        src_item_path = os.path.join(src_dir, item)
        dst_item_path = os.path.join(dst_dir, item)

        try:
            if os.path.isdir(src_item_path):
                # Se ambos são diretórios, mescla recursivamente
                _merge_directories_content(src_item_path, dst_item_path)
                # Remove o diretório de origem se ele estiver vazio após a mesclagem
                if not os.listdir(src_item_path):
                    shutil.rmtree(src_item_path)
            elif os.path.isfile(src_item_path):
                # Se a origem é um arquivo
                if os.path.exists(dst_item_path):
                    if os.path.isdir(dst_item_path):
                        # Conflito: arquivo vs diretório. Remove o diretório para mover o arquivo
                        print_status(f"    AVISO: Conflito de tipo. Removendo diretório '{dst_item_path}' para mover arquivo.", "33")
                        shutil.rmtree(dst_item_path)
                    else:
                        # Ambos são arquivos, sobrescreve o destino
                        print_status(f"    Sobrescrevendo arquivo: '{dst_item_path}'", "33")
                        os.remove(dst_item_path) # Remover antes de mover para garantir
                shutil.move(src_item_path, dst_item_path)
            else:
                print_status(f"    AVISO: Item '{src_item_path}' é de um tipo desconhecido, pulando.", "33")
        except shutil.Error as e:
            print_status(f"    ERRO de shutil ao mesclar '{src_item_path}' para '{dst_item_path}': {e}", "31")
        except OSError as e:
            print_status(f"    ERRO de OS ao mesclar '{src_item_path}' para '{dst_item_path}': {e}", "31")
        except Exception as e:
            print_status(f"    ERRO inesperado ao mesclar '{src_item_path}' para '{dst_item_path}': {e}", "31")


def create_new_directories():
    print_status("1. Criando a nova estrutura de diretórios...", "34")
    for d in NEW_DIRS:
        path = os.path.join(BASE_PATH, d)
        os.makedirs(path, exist_ok=True)
        print_status(f"  Criado: {path}", "36")

def create_init_files():
    print_status("\n2. Criando arquivos __init__.py em diretórios de pacotes Python...", "34")
    python_dirs = [
        "src/api", "src/api/data_generator", "src/api/data_lake",
        "src/api/artifact_store", "src/api/model_serving",
        "src/flows", "src/pipelines"
    ]
    for p_dir in python_dirs:
        init_path = os.path.join(BASE_PATH, p_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding='utf-8') as f:
                f.write("") # Cria um arquivo __init__.py vazio
            print_status(f"  Criado: {init_path}", "36")
        else:
            print_status(f"  Pulado (já existe): {init_path}", "33")

def move_and_rename_files():
    print_status("\n3. Movendo e renomeando arquivos...", "34")
    for old_name, new_rel_path in FILE_MAPPING.items():
        old_path = os.path.join(BASE_PATH, old_name)
        
        if new_rel_path is None: # Arquivos para serem ignorados/deletados
            if os.path.exists(old_path):
                print_status(f"  Deletando (ignorado no mapeamento): {old_name}", "33")
                try:
                    os.remove(old_path)
                except OSError as e:
                    print_status(f"  ERRO ao deletar '{old_name}': {e}", "31")
            else:
                print_status(f"  Pulado (não encontrado para deletar): {old_name}", "33")
            continue

        new_path = os.path.join(BASE_PATH, new_rel_path)
        
        if not os.path.exists(old_path):
            if os.path.exists(new_path):
                print_status(f"  Pulado (arquivo '{old_name}' já movido para '{new_rel_path}').", "33")
            else:
                print_status(f"  Pulado (arquivo '{old_name}' não encontrado).", "33")
            continue
        
        # Se o arquivo de origem existe e o de destino também
        if os.path.exists(new_path):
            try:
                if os.path.samefile(old_path, new_path):
                    print_status(f"  Pulado (arquivo '{old_name}' já está no destino correto '{new_rel_path}').", "33")
                    # Se eles são o mesmo arquivo, remove a referência antiga para limpar
                    os.remove(old_path)
                    continue
            except FileNotFoundError:
                # Se um dos caminhos não existe mais, não é o mesmo arquivo
                pass 
            
            # Se o destino já existe mas não é o mesmo arquivo, remove o destino para sobrescrever
            print_status(f"  AVISO: Destino '{new_rel_path}' já existe. Sobrescrevendo com '{old_name}'.", "33")
            try:
                if os.path.isdir(new_path): shutil.rmtree(new_path)
                else: os.remove(new_path)
            except OSError as e:
                print_status(f"  ERRO ao remover destino existente '{new_path}': {e}", "31")
                continue # Não podemos mover se o destino não pode ser limpo
        
        try:
            # Garante que o diretório pai do destino exista antes de mover
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
            print_status(f"  Movido: {old_name} -> {new_rel_path}", "32")
        except Exception as e:
            print_status(f"  ERRO ao mover '{old_name}' para '{new_rel_path}': {e}", "31")

def apply_content_replacements():
    print_status("\n4. Aplicando substituições de conteúdo em arquivos...", "34")
    for old_regex, new_string, file_pattern_regex in CONTENT_REPLACEMENTS:
        # Itera sobre todos os novos caminhos de arquivos para encontrar os que correspondem ao padrão
        for new_original_filename, new_rel_path in FILE_MAPPING.items():
            if new_rel_path and re.match(file_pattern_regex, new_rel_path):
                file_path = os.path.join(BASE_PATH, new_rel_path)
                if os.path.exists(file_path):
                    print_status(f"  Atualizando: {new_rel_path}", "36")
                    # Usar fileinput.FileInput(inplace=True) para edição segura e in-place
                    with fileinput.FileInput(file_path, inplace=True, encoding="utf-8") as file:
                        for line in file:
                            # Re.sub faz a substituição baseada em regex
                            print(re.sub(old_regex, new_string, line), end='')
                else:
                    print_status(f"  Pulado atualização de conteúdo: {new_rel_path} (arquivo não encontrado)", "33")

def consolidate_and_move_data_directories():
    print_status("\n5. Consolidando e movendo diretórios de dados e artefatos...", "34")

    # Primeiro, move o arquivo mlruns.db
    old_mlruns_db_path = os.path.join(BASE_PATH, "mlruns.db")
    new_mlruns_db_dir = os.path.join(BASE_PATH, "data")
    new_mlruns_db_path = os.path.join(new_mlruns_db_dir, "mlflow_backend.db")
    
    os.makedirs(new_mlruns_db_dir, exist_ok=True) # Garante que o diretório 'data' exista
    
    if os.path.exists(old_mlruns_db_path):
        if os.path.exists(new_mlruns_db_path):
            # Se o novo DB já existe, e o antigo ainda está lá, assume que o antigo é redundante.
            print_status(f"  AVISO: '{os.path.basename(new_mlruns_db_path)}' já existe. Removendo DB antigo: '{os.path.basename(old_mlruns_db_path)}'.", "33")
            try:
                os.remove(old_mlruns_db_path)
            except OSError as e:
                print_status(f"  ERRO ao remover '{old_mlruns_db_path}': {e}", "31")
        else:
            try:
                shutil.move(old_mlruns_db_path, new_mlruns_db_path)
                print_status(f"  Movido DB de backend do MLflow: mlruns.db -> data/mlflow_backend.db", "32")
            except Exception as e:
                print_status(f"  ERRO ao mover '{old_mlruns_db_path}': {e}", "31")
    else:
        if os.path.exists(new_mlruns_db_path):
            print_status(f"  Pulado (mlruns.db não encontrado, mas '{os.path.basename(new_mlruns_db_path)}' já existe).", "33")
        else:
            print_status(f"  Pulado (mlruns.db não encontrado).", "33")

    # Move os conteúdos de outros diretórios especificados (ex: artifacts_storage, test_data_reports)
    for old_rel_path, new_rel_path in CONTENT_MOVES_DIRS:
        old_path = os.path.join(BASE_PATH, old_rel_path)
        new_path = os.path.join(BASE_PATH, new_rel_path)

        if not os.path.exists(old_path):
            if os.path.exists(new_path):
                print_status(f"  Pulado (diretório '{old_rel_path}' não encontrado, mas '{new_rel_path}' já existe).", "33")
            else:
                print_status(f"  Diretório de origem não encontrado, pulando: {old_rel_path}.", "33")
            continue
        if not os.path.isdir(old_path):
            print_status(f"  Origem '{old_rel_path}' não é um diretório, pulando movimento de conteúdo.", "33")
            continue
        if old_path == new_path: # Evita tentar mover diretório para ele mesmo
            print_status(f"  Pulado (origem e destino são o mesmo diretório): {old_rel_path}", "33")
            continue

        print_status(f"  Mesclando e movendo conteúdo de '{old_rel_path}' para '{new_path}'...", "36")
        _merge_directories_content(old_path, new_path)
        
        if not os.listdir(old_path): # Se o diretório de origem ficou vazio, remove
            try:
                shutil.rmtree(old_path)
                print_status(f"  Diretório de origem vazio removido: {old_rel_path}", "32")
            except OSError as e:
                print_status(f"  ERRO ao remover diretório vazio '{old_path}': {e}", "31")
        else:
            print_status(f"  Diretório de origem '{old_rel_path}' não está vazio após a movimentação (pode haver itens restantes ou erros).", "33")


    # Lida com a consolidação dos diretórios raiz do MLflow (mlartifacts e mlruns)
    target_mlruns_path = os.path.join(BASE_PATH, "mlruns")
    os.makedirs(target_mlruns_path, exist_ok=True) # Garante que o novo diretório 'mlruns' exista

    for old_mlflow_dir_name in MLFLOW_OLD_DIRS_TO_CONSOLIDATE:
        old_mlflow_path = os.path.join(BASE_PATH, old_mlflow_dir_name)
        if os.path.exists(old_mlflow_path) and os.path.isdir(old_mlflow_path):
            print_status(f"  Consolidando dados do MLflow de '{old_mlflow_dir_name}' para '{os.path.basename(target_mlruns_path)}'...", "36")
            _merge_directories_content(old_mlflow_path, target_mlruns_path)
            
            if not os.listdir(old_mlflow_path):
                try:
                    shutil.rmtree(old_mlflow_path)
                    print_status(f"  Diretório de origem MLflow vazio removido: {old_mlflow_dir_name}", "32")
                except OSError as e:
                    print_status(f"  ERRO ao remover diretório vazio '{old_mlflow_path}': {e}", "31")
            else:
                print_status(f"  Diretório de origem MLflow '{old_mlflow_dir_name}' não está vazio após a movimentação (pode haver itens restantes ou erros).", "33")
        else:
            print_status(f"  Diretório MLflow de origem '{old_mlflow_dir_name}' não encontrado, pulando.", "33")

def create_gitignore():
    print_status("\n6. Criando arquivo .gitignore...", "34")
    gitignore_content = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env

# Cache
.pytest_cache/
*.egg-info/
.ipynb_checkpoints/
tmp/
temp_pipeline_artifacts/

# Prefect
.prefect/

# MLflow
mlruns/
# Os artefatos do MLflow geralmente ficam dentro de 'mlruns/', então 'mlartifacts/' não deveria ser necessário como diretório de nível superior.
# No entanto, se o seu setup anterior criou um diretório 'mlartifacts' separado,
# ele será movido para 'mlruns' e este padrão garantirá que futuros logs não criem um novo.

# Dados sensíveis/grandes
data/*.db    # Para bancos de dados locais como mlflow_backend.db e sales_data_lake.db

# Relatórios e logs (podem ser grandes ou transitórios)
reports/
*.log

# IDEs
.idea/
.vscode/

# Build artifacts
dist/
build/
    """
    with open(os.path.join(BASE_PATH, ".gitignore"), "w", encoding='utf-8') as f:
        f.write(gitignore_content.strip())
    print_status("  .gitignore criado.", "32")

def create_requirements_txt():
    print_status("\n7. Criando requirements.txt (placeholder)...", "34")
    requirements_content = """
flask
flask-cors
werkzeug
requests
pandas
numpy
matplotlib
scikit-learn
tensorflow
mlflow
prefect>=2.0
joblib
pyyaml
# Observação: sqlite3 é built-in no Python e não precisa ser listado.
# pip install --upgrade "protobuf<3.20" # Descomente se tiver problemas com TensorFlow/protobuf
    """
    with open(os.path.join(BASE_PATH, "requirements.txt"), "w", encoding='utf-8') as f:
        f.write(requirements_content.strip())
    print_status("  requirements.txt criado (placeholder). Recomenda-se executar 'pip freeze > requirements.txt' no seu ambiente final para gerar a lista exata de dependências.", "33")

def create_run_all_servers_script():
    print_status("\n8. Criando script 'run_all_servers.py'...", "34")
    # Define o caminho do script AQUI antes de usar
    run_all_servers_script_path = os.path.join(BASE_PATH, "scripts", "run_all_servers.py")

    # Usamos uma string literal tripla-aspas para o conteúdo do script `run_all_servers.py`.
    # As f-strings dentro deste conteúdo (ex: f"Iniciando {name}...") são escritas com
    # aspas simples ou duplas internas e os placeholders {} são escapados como {{}}
    # para que o Python interprete-os corretamente NO ARQUIVO GERADO.
    run_all_servers_content = """
import subprocess
import os
import time
import sys

# Define o diretório raiz do projeto. Este script está em 'scripts/', então subimos um nível.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_command_in_new_terminal(command_parts, name, port=None):
    \"\"\"Executa um comando Python em um novo terminal dependendo do SO.\"\"\"
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
            print(f"  Tente executar manualmente: cd /d \"{{PROJECT_ROOT}}\" && {subprocess.list2cmdline(full_command_list)}")

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
            print(f"  cd \"{{PROJECT_ROOT}}\" && {subprocess.list2cmdline(full_command_list)}")
    else:
        print(f"ERRO: Sistema operacional não suportado para iniciar {{name}} em novo terminal.")
        print(f"Por favor, execute manualmente: cd \"{{PROJECT_ROOT}}\" && {subprocess.list2cmdline(full_command_list)}")

def main():
    print("╔═════════════════════════════════════════════════════════╗")
    print("║         Iniciando TODOS os Servidores e APIs            ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print("\\nCertifique-se de que o ambiente Python esteja ativado e as dependências instaladas.")
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

    print("\\n" + "="*70)
    print("Todos os serviços foram iniciados (verifique os terminais abertos).")
    print("Agora você pode executar os Prefect Flows e MLflow Pipelines em novos terminais.")
    print("Ex: python src/flows/data_ingestion_flow.py")
    print("Ex: python src/pipelines/data_analysis_pipeline.py")
    print("Ex: python src/pipelines/model_training_pipeline.py")
    print("Ex: python scripts/run_model_tester.py")
    print("="*70 + "\\n")

if __name__ == "__main__":
    main()
"""
    with open(run_all_servers_script_path, "w", encoding='utf-8') as f:
        f.write(run_all_servers_content.strip())
    print_status(f"  scripts/run_all_servers.py criado/atualizado. Ajustes podem ser necessários para seu emulador de terminal no Linux/macOS.", "32")


# -------------------------------------------------------------------------------------------------
# Função Principal de Refatoração
# -------------------------------------------------------------------------------------------------
def main():
    print_status("Iniciando a refatoração do projeto para uma estrutura MLOps profissional...", "32")
    
    # 0. Verifica se o ambiente está limpo de pastas temporárias antes de começar
    temp_dirs = ["temp_mlflow_artifacts", "temp_mlflow_model_artifacts", "temp_pipeline_artifacts"]
    for t_dir in temp_dirs:
        full_temp_path = os.path.join(BASE_PATH, t_dir)
        if os.path.exists(full_temp_path):
            print_status(f"  Removendo diretório temporário existente: {t_dir}", "33")
            try:
                shutil.rmtree(full_temp_path)
            except OSError as e:
                print_status(f"  AVISO: Não foi possível remover o diretório temporário '{t_dir}': {e}. Pode estar em uso.", "33")


    create_new_directories()
    create_init_files()
    move_and_rename_files()
    apply_content_replacements()
    consolidate_and_move_data_directories()
    create_gitignore()
    create_requirements_txt()
    create_run_all_servers_script() # Cria o script para iniciar tudo

    print_status("\n------------------------------------------------------------------------------------------", "32")
    print_status("Refatoração completa! Seu projeto agora segue uma estrutura MLOps profissional.", "32")
    print_status("------------------------------------------------------------------------------------------", "32")
    print_status("\nPróximos passos recomendados:", "36")
    print_status("1. Verifique a nova estrutura de diretórios e arquivos.", "36")
    print_status("2. **Crie e ative um ambiente virtual** (venv ou conda) para o projeto. Exemplo:", "36")
    print_status("   `python -m venv .venv`", "36")
    print_status("   `source .venv/bin/activate` (Linux/macOS) ou `.venv\\Scripts\\activate` (Windows)", "36")
    print_status("3. **Instale as dependências**:", "36")
    print_status("   `pip install -r requirements.txt`", "36")
    print_status("   (Opcional, mas recomendado para fixar versões: `pip freeze > requirements.txt` depois da instalação)", "36")
    print_status("4. Edite o novo `README.md` para descrever seu projeto, configuração e como executá-lo de forma mais detalhada.", "36")
    print_status("5. Para **iniciar todos os serviços** (APIs e Servidores), execute:", "36")
    print_status(f"   `python {os.path.join('scripts', 'run_all_servers.py')}`", "36")
    print_status("   (Isso abrirá vários terminais, um para cada serviço. Deixe-os rodando.)", "36")
    print_status("6. Depois, em **novos terminais** separados, execute seus pipelines e flows:", "36")
    print_status(f"   `python {os.path.join('src', 'flows', 'data_ingestion_flow.py')}`", "36")
    print_status(f"   `python {os.path.join('src', 'pipelines', 'data_analysis_pipeline.py')}`", "36")
    print_status(f"   `python {os.path.join('src', 'pipelines', 'model_training_pipeline.py')}`", "36")
    print_status(f"   `python {os.path.join('scripts', 'run_model_tester.py')}`", "36")
    print_status("\nOBS: Arquivos de backup `.bak` gerados durante a substituição de conteúdo são normais e serão ignorados/limpos pelo `.gitignore`.", "33")

if __name__ == "__main__":
    # Crie um README.md básico se não existir
    if not os.path.exists(os.path.join(BASE_PATH, "README.md")):
        with open(os.path.join(BASE_PATH, "README.md"), "w", encoding='utf-8') as f:
            f.write("# Projeto MLOps de Previsão de Vendas Chevrolet\n\n")
            f.write("Este projeto demonstra um pipeline de MLOps completo para previsão de vendas de carros Chevrolet, utilizando Prefect para orquestração de dados e MLflow para rastreamento de experimentos, registro de modelos e serving.\n\n")
            f.write("## Estrutura do Projeto\n\n")
            f.write("A estrutura do projeto foi organizada seguindo as melhores práticas de MLOps para clareza, modularidade e escalabilidade.\n\n")
            f.write("```\n")
            f.write(f"{os.path.basename(BASE_PATH)}/\n")
            for d in NEW_DIRS:
                f.write(f"├── {d}/\n")
            f.write("├── .gitignore\n")
            f.write("├── README.md\n")
            f.write("└── requirements.txt\n")
            f.write("```\n\n")
            f.write("## Configuração e Execução\n\n")
            f.write("Para instruções detalhadas sobre como configurar o ambiente e executar o projeto, consulte a documentação completa em `docs/project_documentation.html` e os comentários nos scripts dentro da pasta `scripts/`.\n")
        print_status("  Criado um README.md básico.", "32")

    main()