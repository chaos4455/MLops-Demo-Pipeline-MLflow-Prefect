# ✨ Projeto de demonstração - Tech Demo Showcase Pipeline de dados modular MLops✨

[![MLOps](https://img.shields.io/badge/MLOps-Architecture-blueviolet?style=for-the-badge&logo=apacheairflow&logoColor=white)](https://ml-ops.org/)
[![Prefect](https://img.shields.io/badge/Orchestration-Prefect%202.x-8a00b6?style=for-the-badge&logo=prefect&logoColor=white)](https://www.prefect.io/)
[![MLflow](https://img.shields.io/badge/ML%20Platform-MLflow-blue?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Language-Python%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Microservices](https://img.shields.io/badge/Architecture-Microservices-purple?style=for-the-badge&logo=nginx&logoColor=white)](https://microservices.io/)
[![Data Pipeline](https://img.shields.io/badge/Pipeline-Data%20Driven-007ACC?style=for-the-badge&logo=apachekafka&logoColor=white)](https://en.wikipedia.org/wiki/Data_pipeline)

---

## 👨‍💻 Sobre o Projeto: Construindo um Pipeline de MLOps para Previsão de Vendas com APIs, MLflow e Prefect

Olá! Sou **Elias Andrade**, um especialista em IA, Dados e MLOps, e este projeto é uma demonstração prática e modular de como construo pipelines de Machine Learning de ponta a ponta, focado em **escalabilidade, desacoplamento e observabilidade**. Aqui, apresento uma **Prova de Conceito (POC)** de um sistema de previsão de vendas de veículos Chevrolet, simulando um ambiente de produção realista com princípios de **arquitetura de microserviços** e orquestração robusta.

O objetivo principal é demonstrar minha capacidade de projetar, implementar e gerenciar pipelines de ML que entregam valor de negócio contínuo, desde a ingestão de dados até o monitoramento do modelo em produção.

🔗 [Conecte-se comigo no LinkedIn!](https://www.linkedin.com/in/itilmgf/)

---

## 🎯 O Desafio: Prever Vendas em um Ecossistema Complexo

No cenário atual, concessionárias buscam otimizar estoque, campanhas de marketing e alocação de recursos. A previsão precisa de vendas é crucial. No entanto, os dados podem vir de múltiplas fontes, precisar de pré-processamento complexo, e os modelos de ML exigem gerenciamento rigoroso, desde o treinamento até a implantação e monitoramento.

Este projeto aborda esses desafios através de:
*   **Fontes de Dados Desacopladas:** Simulação de APIs de dados.
*   **Injeção de Dados Confiável:** Orquestração para garantir a movimentação de dados.
*   **Centralização de Dados:** Um "Data Lake" simplificado para consumo por pipelines.
*   **Pipeline de ML Completo:** Do pré-processamento à avaliação e registro do modelo.
*   **Serviço de Modelo em Tempo Real:** Disponibilização de previsões via API.
*   **Gerenciamento de Artefatos:** Armazenamento externo de relatórios e gráficos.
*   **Observabilidade:** Monitoramento do ciclo de vida do ML com MLflow e Prefect.

---

# Relatório de Teste da API de Modelo MLflow (2025-09-01 19:31:58)

Este relatório resume o teste de consumo da API de modelo MLflow para previsão de vendas Chevrolet.

## 1. Configurações do Teste
- **URL da API de Modelo:** `http://127.0.0.1:8780/predict`
- **Número de Amostras Testadas:** `100`

## 2. Estatísticas das Previsões
- **Mínimo de Vendas Previstas:** `3.40`
- **Máximo de Vendas Previstas:** `29.70`
- **Média de Vendas Previstas:** `15.99`
- **Desvio Padrão das Vendas Previstas:** `6.01`

## 3. Amostra de Previsões
Aqui estão as primeiras 5 amostras de entrada com suas respectivas previsões:
| store_id   | model     |   month |   day_of_week |   predicted_sales |
|:-----------|:----------|--------:|--------------:|------------------:|
| Loja_09    | Montana   |      11 |             5 |             27.26 |
| Loja_20    | Equinox   |      10 |             4 |             18.02 |
| Loja_01    | Montana   |      12 |             0 |             24.46 |
| Loja_12    | Montana   |       3 |             5 |             21.32 |
| Loja_12    | Onix Plus |       3 |             6 |              6.58 |

## 4. Visualização
Um gráfico da distribuição das previsões e média por modelo foi gerado:
![Gráfico de Previsões](model_test_report_overview.png)

<img width="1200" height="600" alt="ml_model_predictions_plot" src="https://github.com/user-attachments/assets/f9ca790a-b2d7-4ca7-8c58-9495e599da10" />

---



## 🚀 Visão Geral da Arquitetura: Microserviços e Orquestração

Minha abordagem para este pipeline é fundamentalmente baseada em **microserviços**, onde cada componente executa uma função específica e se comunica via APIs bem definidas. Essa modularidade garante **flexibilidade**, **manutenibilidade** e **escalabilidade** independente de cada parte.
---

## ⚙️ Detalhando os Componentes & Fluxo de Dados

Cada peça deste quebra-cabeça foi cuidadosamente projetada para uma função específica, demonstrando um pipeline completo de MLOps.

### 1. 🚰 Fontes e Ingestão de Dados (APIs + Prefect)

A jornada dos dados começa com a geração e ingestão:

*   **`src/api/data_generator/app.py` (API de Dados de Vendas - Porta 8777)**
    *   **Função:** Simula a **fonte de dados primária** – uma API RESTful que gera dados de vendas de veículos Chevrolet para 20 lojas ao longo de um ano. Inclui modelos populares, preços médios e sazonalidade diária/mensal, tornando os dados mais realistas.
    *   **Tecnologia:** Flask, Pandas, Numpy, Random.
    *   **Endpoint:** `/sales_data` (GET)
    *   **Minha Visão:** Este é um exemplo de uma fonte de dados *ativa*. No mundo real, poderia ser um CRM, ERP, ou um sistema de PDV, provendo dados brutos que precisam ser coletados.

*   **`src/flows/data_ingestion_flow.py` (Prefect Data Ingestion Flow)**
    *   **Função:** O cérebro da orquestração de dados. Este **Prefect Flow** é responsável por buscar ativamente os dados brutos da `data_generator` e enviá-los para a `data_lake` de forma confiável. Utiliza **tasks** para cada etapa (fetch, push) e inclui tratamento de erros (`try-except`) para resiliência.
    *   **Tecnologia:** Prefect 2.x, `requests`.
    *   **Minha Visão:** Demonstra como orquestro a movimentação de dados. Prefect garante que o processo seja executado de forma agendada, monitorada, com retentativas e notificação de falhas, crucial para pipelines de dados de produção.

*   **`src/api/data_lake/app.py` (API de Data Lake - Porta 8778)**
    *   **Função:** Atua como um **repositório centralizado e passivo** para os dados ingeridos. Ele não busca dados ativamente; apenas recebe (`POST /store_data`) e serve (`GET /datasource`) o que lhe é enviado, armazenando-o em um banco de dados SQLite (`sales_data_lake.db`).
    *   **Tecnologia:** Flask, Pandas, SQLite.
    *   **Endpoints:** `/store_data` (POST), `/datasource` (GET).
    *   **Minha Visão:** Este microserviço simula a camada de ingestão para um "Data Lake" ou "Data Warehouse" simplificado. O desacoplamento aqui é fundamental: a fonte de dados (data_generator) não precisa saber onde os dados serão armazenados, e os consumidores (ML Pipelines) só precisam saber de onde lê-los.

### 2. 🧠 Pipeline de Machine Learning (MLflow)

Com os dados no Data Lake, entra em cena a magia do ML:

*   **`src/pipelines/model_training_pipeline.py` (MLflow Model Training Pipeline)**
    *   **Função:** Este é o coração do projeto. Um **pipeline completo de ML** que:
        1.  Busca os dados agregados da `data_lake`.
        2.  Realiza **pré-processamento avançado**: engenharia de features (mês, dia da semana), One-Hot Encoding para categóricas, e cria um `ColumnTransformer` robusto.
        3.  Treina um modelo de **Deep Learning (TensorFlow/Keras)** para prever as vendas.
        4.  **Avalia** o modelo usando métricas como MAE, MSE, RMSE e R2 Score.
        5.  Realiza uma **predição de "Top Car"** para um mês específico, demonstrando o valor de negócio imediato.
        6.  Gera **relatórios detalhados (Markdown)** e **gráficos (PNG)** do desempenho do modelo e das previsões.
        7.  **Rastreia todo o experimento com MLflow**: Registra parâmetros, métricas, artefatos (pré-processador, modelo Keras), e o **modelo completo no MLflow Model Registry** como um `pyfunc` customizado, que encapsula tanto o pré-processador quanto o modelo TensorFlow.
        8.  Faz o **upload dos relatórios e gráficos** para a `artifact_store` externa, garantindo a acessibilidade.
    *   **Tecnologia:** MLflow, TensorFlow/Keras, Scikit-learn, Pandas, Matplotlib, `requests`.
    *   **Minha Visão:** Este pipeline é um claro exemplo de **MLOps em ação**. Desde o rastreamento de experimentos com MLflow até a modularização do pré-processamento e a encapsulação do modelo (Pyfunc), tudo visa reprodutibilidade, governança e deploy facilitado. O upload para a `artifact_store` ilustra o desacoplamento do MLflow para fins de distribuição e arquivamento de relatórios.

*   **`src/pipelines/data_analysis_pipeline.py` (MLflow Data Analysis Pipeline)**
    *   **Função:** Um pipeline auxiliar focado na **análise exploratória de dados (EDA)** e geração de insights. Ele busca os dados do `data_lake`, identifica e visualiza os **10 modelos mais vendidos**.
    *   **Tecnologia:** MLflow, Pandas, Matplotlib, `requests`.
    *   **Minha Visão:** Destaca a importância da fase de exploração de dados no ciclo de vida do ML, além de demonstrar a capacidade de gerar e persistir relatórios visuais como artefatos valiosos, tanto no MLflow quanto externamente na `artifact_store`.

### 3. 📦 Armazenamento de Artefatos e Model Serving

A persistência e o consumo das saídas são cruciais:

*   **`src/api/artifact_store/app.py` (API de Storage de Artefatos - Porta 8779)**
    *   **Função:** Um microserviço simples para **armazenamento genérico de arquivos (artefatos)**. Ele recebe uploads (`POST /upload_artifact`) e serve arquivos (`GET /artifacts/<filename>`).
    *   **Tecnologia:** Flask, `werkzeug.utils` (para nomes de arquivos seguros).
    *   **Endpoints:** `/upload_artifact` (POST), `/artifacts/<filename>` (GET), `/list_artifacts` (GET).
    *   **Minha Visão:** Este serviço atua como um "S3 local" para relatórios, gráficos, e outros artefatos que não são necessariamente gerenciados pelo MLflow, mas que precisam ser persistidos e acessíveis. Ele ilustra a flexibilidade de desacoplar o armazenamento de artefatos não-MLflow do próprio MLflow Tracking Server.

*   **`src/api/model_serving/app.py` (API de Serviço de Modelo - Porta 8780)**
    *   **Função:** A interface para o consumo do modelo de ML em produção. Carrega o modelo treinado e registrado no **MLflow Model Registry** (`ChevroletSalesPredictor:latest`) e expõe um endpoint de `predict` em tempo real. Garante que os dados de entrada correspondam à `ModelSignature` definida.
    *   **Tecnologia:** MLflow, Flask, Pandas, Numpy.
    *   **Endpoints:** `/predict` (POST), `/model_status` (GET).
    *   **Minha Visão:** Demonstra o deploy de modelos de ML como **microserviços desacoplados**. A API de serving é agnóstica ao pipeline de treinamento, buscando a "última versão" do modelo no Registry, o que facilita rollouts e rollbacks. A validação de schema é uma prática de MLOps crucial para garantir a integridade das previsões.

*   **`scripts/run_model_tester.py` (MLflow Model Tester)**
    *   **Função:** Um script de **validação pós-deploy** essencial para MLOps. Ele gera dados sintéticos, envia-os para a `model_serving` API, coleta as previsões e gera um relatório de teste (Markdown) e um gráfico de distribuição das previsões.
    *   **Tecnologia:** `requests`, Pandas, Matplotlib.
    *   **Minha Visão:** Destaca a importância de testar e validar modelos continuamente após a implantação, garantindo que a API de serving esteja funcional e o modelo entregando previsões dentro do esperado.

### 4. 🛠️ Orquestração de Infraestrutura (Scripts Utilitários)

Para gerenciar o ambiente de desenvolvimento e produção:

*   **`scripts/start_mlflow_server.py`:** Inicia o MLflow Tracking Server e UI (`http://127.0.0.1:5001`), usando `data/mlflow_backend.db` para metadados e `mlruns/` para artefatos. Essencial para rastreamento e registro de modelos.
*   **`scripts/start_prefect_server.py`:** Inicia o Prefect Server e UI (`http://localhost:4200`), a plataforma de orquestração de fluxos de dados.
*   **`scripts/run_all_servers.py`:** Um script de conveniência para iniciar *todas* as APIs e servidores (MLflow, Prefect, Data Generator, Data Lake, Artifact Store, Model Serving) em terminais separados, simplificando a configuração do ambiente.

---

## 📊 Fluxo de Dados de Ponta a Ponta: Uma Jornada do Dado

1.  **Geração/Coleta de Dados Brutos (API de Geração de Dados)**: Dados simulados de vendas são criados.
2.  **Injeção Orquestrada (Prefect Data Ingestion Flow)**: O Prefect busca os dados da API de Geração e os envia para a API de Data Lake.
3.  **Armazenamento no Data Lake (API de Data Lake)**: Os dados são persistidos no `sales_data_lake.db`, servindo como fonte de verdade.
4.  **Análise e Visualização (MLflow Data Analysis Pipeline)**: O pipeline de análise lê do Data Lake, calcula top modelos e gera gráficos que são logados no MLflow e enviados para a API de Storage de Artefatos.
5.  **Treinamento e Registro do Modelo (MLflow Model Training Pipeline)**:
    *   Lê dados do Data Lake.
    *   Pré-processa (engenharia de features, One-Hot Encoding).
    *   Treina um modelo TensorFlow/Keras.
    *   Avalia o modelo.
    *   Gera relatórios e gráficos do desempenho.
    *   **Loga tudo no MLflow Tracking Server**.
    *   **Registra o modelo (junto com seu pré-processador)** no MLflow Model Registry, com `ModelSignature` e ambiente Conda.
    *   Envia relatórios e gráficos para a API de Storage de Artefatos.
6.  **Serviço de Modelo (API de Serviço de Modelo)**: A API de Serving carrega a "versão mais recente" do modelo do MLflow Model Registry, tornando-o disponível para inferência em tempo real.
7.  **Validação Pós-Deploy (MLflow Model Tester)**: O tester gera dados sintéticos, testa a API de Serving e gera relatórios de validação para garantir a integridade do serviço.
8.  **Monitoramento e Governanças (MLflow UI & Prefect UI)**: Todas as execuções de pipelines, parâmetros, métricas, artefatos e modelos são visíveis e gerenciáveis via MLflow UI. A orquestração e o status dos fluxos são monitorados via Prefect UI.

---

## 🛠️ Tecnologias Principais Utilizadas

Este projeto demonstra proficiência com as seguintes ferramentas e frameworks:

*   **Python 3.11:** Linguagem de programação principal.
*   **Flask:** Framework para construção de APIs RESTful.
*   **Prefect 2.x:** Plataforma de orquestração de dataflows e workflows, com UI para monitoramento.
*   **MLflow:** Plataforma para o ciclo de vida de Machine Learning (Tracking, Models, Registry, Projects, Serving).
*   **TensorFlow/Keras:** Framework de Deep Learning para construção e treinamento do modelo de previsão.
*   **Scikit-learn:** Ferramentas para pré-processamento de dados e avaliação de modelos.
*   **Pandas:** Biblioteca para manipulação e análise de dados.
*   **Numpy:** Biblioteca para computação numérica.
*   **Matplotlib:** Para geração de visualizações de dados e relatórios.
*   **Requests:** Para comunicação HTTP entre os microserviços.
*   **SQLite:** Banco de dados leve para persistência de dados local.
*   **`subprocess` & `os`:** Para gerenciamento de processos e sistema de arquivos.

---

## 🌍 Aplicações no Mundo Real e Valor de Negócio

Este pipeline modular e desacoplado não é apenas uma demonstração técnica; ele reflete a forma como soluções de IA/ML são construídas e mantidas em ambientes de produção. As aplicações e o valor de negócio são vastos:

*   **Previsão de Demanda e Vendas:** Otimização de estoque, planejamento de produção, alocação de equipes de vendas.
*   **Marketing Otimizado:** Identificação de tendências de modelos para campanhas direcionadas.
*   **Detecção de Anomalias:** Adaptação do pipeline para identificar padrões incomuns em dados financeiros, de segurança, etc.
*   **Sistemas de Recomendação:** Com ajustes, o mesmo padrão de ingestão e serving pode ser usado para sistemas que recomendam produtos ou serviços.
*   **Manutenção Preditiva:** Prever falhas de equipamentos com base em dados de sensores.

Minha experiência em arquitetar e implementar tal infraestrutura significa que posso traduzir requisitos de negócio em soluções de ML escaláveis, robustas e de alto impacto.

---

## 🚀 Como Começar (Setup & Execução)

Para explorar este projeto, siga estas etapas. Garanto uma experiência suave e um ambiente de trabalho bem organizado!

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/SeuUsuario/mlops-sales-prediction.git
    cd mlops-sales-prediction
    ```
2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv .venv
    # No Linux/macOS
    source .venv/bin/activate
    # No Windows
    .venv\Scripts\activate
    ```
3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    # Para garantir que todas as dependências estejam fixadas:
    pip freeze > requirements.txt
    ```
4.  **Execute o Script de Refatoração (Apenas na primeira vez ou se a estrutura estiver bagunçada):**
    Caso você tenha a estrutura antiga ou queira reconfirmar a organização, execute:
    ```bash
    python refactor_project.py
    ```
    Este script organizará todos os arquivos na estrutura MLOps profissional.
5.  **Inicie Todos os Serviços (APIs e Servidores):**
    Abra um terminal na raiz do projeto e execute:
    ```bash
    python scripts/run_all_servers.py
    ```
    Isso abrirá várias janelas de terminal, cada uma executando um serviço (MLflow UI, Prefect UI, APIs de dados/modelos). Deixe-as rodando.

6.  **Execute os Pipelines de Dados e ML (em NOVOS terminais):**
    Com os serviços base rodando, abra *novos terminais* na raiz do projeto (e ative seu ambiente virtual em cada um).

    *   **1. Ingestão de Dados (Prefect Flow):**
        ```bash
        python src/flows/data_ingestion_flow.py
        ```
        (Isso enviará dados da API de Geração para a API de Data Lake)

    *   **2. Análise de Dados (MLflow Pipeline):**
        ```bash
        python src/pipelines/data_analysis_pipeline.py
        ```
        (Isso gerará um relatório de top modelos)

    *   **3. Treinamento e Registro do Modelo (MLflow Pipeline):**
        ```bash
        python src/pipelines/model_training_pipeline.py
        ```
        (Isso treinará, registrará o modelo MLflow, e gerará relatórios)

    *   **4. Teste de Modelo Pós-Deploy:**
        ```bash
        python scripts/run_model_tester.py
        ```
        (Isso testará a API de Modelo e gerará um relatório de teste)

7.  **Explore as UIs:**
    *   **MLflow UI:** `http://127.0.0.1:5001` (para ver experimentos, modelos, artefatos)
    *   **Prefect UI:** `http://localhost:4200` (para ver o status e logs dos flows)
    *   **APIs:** Verifique as páginas iniciais das APIs (ex: `http://127.0.0.1:8777`, `8778`, `8779`, `8780`)

---

## 📈 Melhorias Futuras e Roadmap

*   **CI/CD Automatizado:** Implementar GitHub Actions (ou Jenkins, GitLab CI) para automação de testes, build, deploy de serviços e triggers de pipelines de ML.
*   **Monitoramento de Drift de Dados e Modelo:** Integração com ferramentas como Evidently AI ou Fiddler para detectar quando a performance do modelo se degrada ou os dados de entrada mudam.
*   **Infraestrutura Cloud-Native:** Migrar os serviços para plataformas como AWS (ECS/EKS, S3, RDS, SageMaker), GCP (Cloud Run, GCS, Vertex AI) ou Azure (Container Apps, Blob Storage, Azure ML).
*   **Modelos Mais Complexos:** Experimentar com modelos de séries temporais ou ensemble para previsões mais avançadas.
*   **UI/Dashboard para Previsões:** Construir uma interface de usuário simples (Streamlit, Dash) para consumir a API de serviço de modelo e visualizar previsões.
*   **A/B Testing de Modelos:** Implementar estratégias para testar novas versões de modelos em produção.
*   **Segurança e Autenticação:** Adicionar autenticação e autorização para as APIs.

---

## 🤝 Contato e Colaboração

Sou apaixonado por construir soluções de IA robustas e escaláveis. Se você tem um projeto desafiador ou quer discutir as melhores práticas de MLOps, adoraria conectar!

**Elias Andrade**  
[LinkedIn: https://www.linkedin.com/in/itilmgf/](https://www.linkedin.com/in/itilmgf/)  
[GitHub: https://github.com/SeuUsuario](https://github.com/SeuUsuario) <!-- Substitua 'SeuUsuario' pelo seu real username do GitHub -->

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---
