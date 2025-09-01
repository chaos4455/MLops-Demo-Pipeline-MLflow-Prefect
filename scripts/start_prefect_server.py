# start_prefect_server.py (CORRIGIDO)

import subprocess
import sys
import time
import os

def start_prefect_server():
    print("╔═════════════════════════════════════════════════════════╗")
    print("║          Iniciando o Servidor Prefect (Local)           ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print("\nIsso inclui o banco de dados e a interface web (UI).")
    print("Aguarde alguns segundos para o servidor carregar...")
    print("Você poderá acessar a UI em: \033[1;34mhttp://localhost:4200\033[0m")
    print("\nPressione \033[1mCtrl+C\033[0m neste terminal para encerrar o servidor.")
    print("-----------------------------------------------------------\n")

    try:
        # COMANDO CORRIGIDO: Use "prefect" como o módulo principal, não "prefect.cli"
        command = [sys.executable, "-m", "prefect", "server", "start"]
        
        print(f"Executando comando: {' '.join(command)}\n")

        # `subprocess.run` irá executar o comando e bloquear o script Python
        # até que o servidor seja encerrado (geralmente por Ctrl+C).
        # check=True: levanta um CalledProcessError se o comando retornar um código de saída não-zero
        # text=True: decodifica stdout/stderr como texto
        # stdout=sys.stdout, stderr=sys.stderr: Redireciona a saída do processo filho
        # diretamente para a saída padrão e de erro do script pai.
        subprocess.run(command, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\n\033[1;31mERRO:\033[0m ao iniciar o servidor Prefect. Código de saída: {e.returncode}")
        # e.stdout e e.stderr podem não estar preenchidos se não foram capturados explicitamente
        # Mas com stdout=sys.stdout e stderr=sys.stderr, a saída já terá sido mostrada.
        print("Verifique se o Prefect está instalado corretamente (`pip install prefect`).")
        sys.exit(1)
    except FileNotFoundError:
        print("\n\033[1;31mERRO:\033[0m O executável Python ou o comando 'prefect' não foi encontrado.")
        print("Certifique-se de que o Python esteja no PATH e que o Prefect esteja instalado no ambiente correto.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n\033[33mEncerrando o servidor Prefect pelo usuário (Ctrl+C)...\033[0m")
    except Exception as e:
        print(f"\n\033[1;31mOcorreu um erro inesperado: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    start_prefect_server()