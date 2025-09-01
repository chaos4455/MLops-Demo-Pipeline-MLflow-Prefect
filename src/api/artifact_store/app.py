import os
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename # Para sanitizar nomes de arquivos

# --- Configurações da API de Storage de Artefatos ---
FLASK_APP_PORT = 8779 # Porta para esta API (diferente das outras)
ARTIFACTS_DIR = "../../artifacts/storage" # Diretório onde os artefatos serão salvos

app = Flask(__name__)
CORS(app) # Habilita CORS para todas as origens

# Garante que o diretório de artefatos exista ao iniciar a aplicação
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f"Diretório de armazenamento de artefatos garantido em: '{ARTIFACTS_DIR}'")

# --- Funções Auxiliares ---
def allowed_file(filename):
    """
    Verifica se a extensão do arquivo é permitida (opcional, para segurança).
    Aqui, permitiremos quase todos os tipos para flexibilidade.
    """
    # Exemplo: Apenas alguns tipos. Para permitir "qualquer tipo", pode-se remover esta checagem.
    # return '.' in filename and \
    #        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'txt', 'pdf', 'xlsx', 'csv', 'html', 'json', 'pkl', 'joblib', 'onnx', 'tf', 'h5'}
    return True # Para permitir qualquer tipo de arquivo conforme a descrição

# --- Endpoints da API de Storage de Artefatos ---

@app.route('/upload_artifact', methods=['POST'])
def upload_artifact():
    """
    Endpoint para fazer upload de um artefato.
    Espera um arquivo via `multipart/form-data` no campo 'file'.
    """
    if 'file' not in request.files:
        print("\033[1;31mERRO: Nenhuma parte de arquivo na requisição de upload.\033[0m")
        return jsonify({"message": "Nenhuma parte de arquivo 'file' na requisição."}), 400

    file = request.files['file']

    if file.filename == '':
        print("\033[1;31mERRO: Nenhum arquivo selecionado para upload.\033[0m")
        return jsonify({"message": "Nenhum arquivo selecionado."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) # Limpa o nome do arquivo para segurança
        filepath = os.path.join(ARTIFACTS_DIR, filename)
        try:
            file.save(filepath)
            print(f"\033[1;32mSucesso:\033[0m Artefato '{filename}' salvo em '{filepath}'.")
            return jsonify({
                "message": f"Artefato '{filename}' carregado com sucesso!",
                "filename": filename,
                "access_url": f"http://{request.host}/artifacts/{filename}"
            }), 201 # 201 Created
        except Exception as e:
            print(f"\033[1;31mERRO ao salvar artefato '{filename}': {e}\033[0m")
            return jsonify({"message": f"Falha ao salvar artefato: {e}"}), 500
    else:
        print(f"\033[1;31mERRO: Tipo de arquivo não permitido para '{file.filename}'.\033[0m")
        return jsonify({"message": "Tipo de arquivo não permitido."}), 400

@app.route('/artifacts/<filename>', methods=['GET'])
def download_artifact(filename):
    """
    Endpoint para servir (fazer download) um artefato pelo seu nome.
    """
    try:
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, filename)):
            print(f"\033[1;33mAVISO: Artefato '{filename}' não encontrado.\033[0m")
            abort(404) # Not Found
        
        print(f"Servindo artefato: '{filename}' de '{ARTIFACTS_DIR}'.")
        return send_from_directory(ARTIFACTS_DIR, filename, as_attachment=False) # as_attachment=False para exibir no navegador se possível
    except Exception as e:
        print(f"\033[1;31mERRO ao servir artefato '{filename}': {e}\033[0m")
        return jsonify({"message": f"Falha ao recuperar artefato: {e}"}), 500

@app.route('/list_artifacts', methods=['GET'])
def list_artifacts():
    """
    Endpoint para listar todos os artefatos atualmente armazenados.
    """
    try:
        artifacts = os.listdir(ARTIFACTS_DIR)
        artifact_list = []
        for artifact_name in artifacts:
            # Pular arquivos ocultos do sistema
            if not artifact_name.startswith('.'):
                artifact_list.append({
                    "filename": artifact_name,
                    "access_url": f"http://{request.host}/artifacts/{artifact_name}",
                    "size_bytes": os.path.getsize(os.path.join(ARTIFACTS_DIR, artifact_name)),
                    "created_at": os.path.getctime(os.path.join(ARTIFACTS_DIR, artifact_name)) # Tempo de criação
                })
        print(f"Listando {len(artifact_list)} artefatos.")
        return jsonify({"artifacts": artifact_list}), 200
    except Exception as e:
        print(f"\033[1;31mERRO ao listar artefatos: {e}\033[0m")
        return jsonify({"message": f"Falha ao listar artefatos: {e}"}), 500

@app.route('/')
def home():
    """
    Endpoint inicial para verificar se a API está funcionando.
    """
    return (
        f"<h1>API de Storage de Artefatos MLflow</h1>"
        f"Esta API está rodando na porta \033[1m{FLASK_APP_PORT}\033[0m e armazena arquivos em '{ARTIFACTS_DIR}'.<br>"
        f"<ul>"
        f"<li>\033[1mPOST\033[0m para <a href='http://127.0.0.1:{FLASK_APP_PORT}/upload_artifact'>/upload_artifact</a> para carregar um arquivo (via `multipart/form-data`).</li>"
        f"<li>\033[1mGET\033[0m para <a href='http://127.0.0.1:{FLASK_APP_PORT}/artifacts/&lt;filename&gt;'>/artifacts/&lt;filename&gt;</a> para baixar um artefato.</li>"
        f"<li>\033[1mGET\033[0m para <a href='http://127.0.0.1:{FLASK_APP_PORT}/list_artifacts'>/list_artifacts</a> para ver a lista de artefatos.</li>"
        f"</ul>"
    )

if __name__ == '__main__':
    print("╔═════════════════════════════════════════════════════════╗")
    print(f"║   Iniciando API de Storage de Artefatos na porta {FLASK_APP_PORT}  ║")
    print("╚═════════════════════════════════════════════════════════╝")
    
    print(f"\nAPI Flask de Storage de Artefatos pronta. Acesse \033[1;34mhttp://127.0.0.1:{FLASK_APP_PORT}\033[0m")
    app.run(port=FLASK_APP_PORT, debug=True)