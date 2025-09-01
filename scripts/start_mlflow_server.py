import subprocess
import sys
import os

def start_mlflow_server(host="127.0.0.1", port=5001, backend_uri="sqlite:///data/mlflow_backend.db", artifact_root="./mlruns"):
    """
    Inicia o MLflow Tracking Server localmente.

    Args:
        host (str): O host em que o servidor MLflow será executado.
        port (int): A porta em que o servidor MLflow será executado.
        backend_uri (str): O URI para o backend store (e.g., "sqlite:///mlruns.db" para SQLite local).
        artifact_root (str): O caminho para o diretório raiz de artefatos (e.g., "./mlartifacts").
    """
    print("╔═════════════════════════════════════════════════════════╗")
    print("║           Iniciando o MLflow Tracking Server            ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print(f"\nO MLflow Server usará:")
    print(f"  - Backend Store (metadados): \033[33m{backend_uri}\033[0m")
    print(f"  - Default Artifact Root (modelos/arquivos): \033[33m{artifact_root}\033[0m")
    print(f"Você poderá acessar a UI em: \033[1;34mhttp://{host}:{port}\033[0m")
    print("\nPressione \033[1mCtrl+C\033[0m neste terminal para encerrar o servidor.\n")
    print("-----------------------------------------------------------\n")

    # Garante que o diretório de artefatos exista
    os.makedirs(artifact_root, exist_ok=True)

    try:
        # Comando para iniciar o servidor MLflow.
        # Estamos usando sys.executable -m mlflow para garantir que o CLI
        # do MLflow do ambiente Python correto seja usado.
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--host", str(host),
            "--port", str(port),
            "--backend-store-uri", backend_uri,
            "--default-artifact-root", artifact_root
        ]
        
        print(f"Executando comando: {' '.join(command)}\n")

        # subprocess.run() irá bloquear o script até que o servidor seja parado (e.g., Ctrl+C)
        # stdout e stderr são passados diretamente para o terminal atual.
        subprocess.run(
            command,
            check=True,  # Levanta CalledProcessError se o comando retornar um código de erro
            text=True,   # Decodifica stdout/stderr como texto
            shell=False, # Não usa o shell, o que é mais seguro
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print("\nMLflow Tracking Server process finished.")

    except FileNotFoundError:
        print("\n\033[1;31mERRO:\033[0m O comando 'mlflow' ou 'python' não foi encontrado.")
        print("Verifique se o Python e o MLflow estão instalados e no PATH.")
        print("Tente instalar o MLflow: \033[33mpip install mlflow\033[0m")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n\033[1;31mERRO:\033[0m ao iniciar o MLflow Tracking Server. Código de saída: {e.returncode}")
        print("Verifique se a porta 5001 está disponível ou tente uma porta diferente.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n\033[33mMLflow Tracking Server parado pelo usuário (Ctrl+C)...\033[0m")
    except Exception as e:
        print(f"\n\033[1;31mOcorreu um erro inesperado: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    start_mlflow_server()