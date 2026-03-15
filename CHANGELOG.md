# Changelog

Todas as modificações notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Adicionado
- Estrutura inicial do projeto para o módulo de Machine Learning em Python.
- Arquivo `.gitignore` configurado para ignorar arquivos temporários, de ambiente virtual (como `.venv`) e compilações no Python.
- Configuração local de dependências com `requirements.txt`.
- Diretório centralizado de logs da execução (`logs/`).
- Módulo principal da aplicação (`src/index.py`).
- Arquivo central de configurações (`src/config.py`).
- Módulo de manipulação de dados (`src/data/`):
  - `dataset.py`: rotinas e abstrações para carregamento de dados.
  - `transforms.py`: funções para pré-processamento e transformações nos dados.
- Módulo de definições dos modelos de inteligência artificial (`src/model/`):
  - `model.py`: estrutura do modelo de ML e predições base.
- Módulo para fluxos e pipelines completos de processamento e treinamento (`src/pipeline/`):
  - `baseline.py`: criação e teste de modelos base comparativos.
  - `compare.py`: relatórios e métricas de comparação de modelos.
  - `crossval.py`: estratégias de validação cruzada para garantir a confiabilidade.
  - `evaluate.py`: fluxo dedicado de avaliação em dados de teste.
  - `export.py`: geração de exportações (artefatos) dos modelos prontos.
  - `train.py`: fluxo principal de execução de treinamento.
