# sentience-train

Treino determinístico e exportação imutável de um classificador FER2013 de quatro classes para ESP32-S3-CAM. Os únicos workflows públicos são `train` e `export`; a seleção cria exclusivamente candidatos DS-CNN.

## Pré-requisitos e instalação

- Python **3.11.15** e uv **0.11.32**, fixados em `mise.toml`;
- `mise` para provisionar as versões;
- CSV FER2013 canônico e seus dados locais em `dataset/`;
- CPU para validação host. CUDA é opcional para treino.

A instalação é bloqueada pelo lockfile:

```bash
mise install
uv sync --locked --group dev --group export
```

O `uv.lock` deve ser usado sem alterações locais. O grupo `dev` inclui pytest, Ruff, mypy e stubs de YAML; `export` inclui ONNX, ONNX Runtime, Matplotlib/Agg e LiteRT/AI Edge.

## Dataset canônico

O arquivo é um CSV UTF-8 com cabeçalho exato `emotion,pixels,Usage`. Cada registro possui 2.304 pixels inteiros entre 0 e 255, e `Usage` é `Training`, `PublicTest` ou `PrivateTest`. O SHA-256 de aceitação é:

```text
3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b
```

O mapeamento estável de emoções para a saída do modelo é `0 -> angry`, `3 -> happy`, `6 -> neutral`, `4 -> sad`. Os dados  não selecionados continuam na proveniência, mas somente as quatro classes canônicas entram no treino. `Training` é otimização; `PublicTest` é validação, early stopping, calibração e seleção; `PrivateTest` só pode ser lido uma vez pelo export completo de um run canônico explicitamente fornecido.

## Configuração e seleção

`configs/default.yaml` é a configuração canônica. O schema é estrito: chaves removidas como `model.name`, `training.seed` e toda a seção `distillation` são rejeitadas. A CLI permite apenas sobrescrever `training.epochs` entre 1 e 100 e `training.device` entre `auto`, `cpu` e `cuda`.

A seleção percorre widths `0.25`, `0.35`, `0.50`, nessa ordem, e seeds `42`, `43`, `44`. Ela para no primeiro width que passa todos os gates. O representante é a mediana de macro-F1 INT8, com menor seed como desempate. Cada treino usa early stopping por macro-F1, com loss como desempate, e restaura o melhor checkpoint. A perda é CrossEntropy com pesos de classe fixos por frequência inversa da raiz quadrada, normalizados pela média.

PTQ é tentada primeiro. QAT só é acionada pelos gatilhos de conversão PTQ ou perda de acurácia acima do limite; não há teacher, MiniVGG ou distilação. Se todos os DS-CNN falharem, o comando cria um diagnóstico imutável `candidate-comparison.json`/`.md`, não cria release e não acessa `PrivateTest`.

## Treino

```bash
sentience-train train --config configs/default.yaml
sentience-train train --config configs/default.yaml --epochs 10 --device cpu
```

O resultado é um run imutável `runs/<id>/` com `manifest.json` no schema `frozen-run/v1`, configuração resolvida, hashes, índice de offsets do `PublicTest`, checkpoints, histórico JSONL, métricas e comparação de candidatos. Fixtures sintéticas são úteis para testes, mas nunca são evidência de release.

## Exportação

```bash
sentience-train export --run-id <id>
sentience-train export --run-id <id> --format onnx --runs-dir ./runs
sentience-train export --run-id <id> --format tflite --runs-dir ./runs
```

`all` é o padrão e publica `exports/<id>/all/`. `onnx` e `tflite` são handoffs verificáveis derivados do mesmo run congelado. Nenhum formato altera o run; manifests e payloads são hash-checados e publicação/reuso é atômico.

### Handoff ONNX FP32

- dtype `float32`;
- layout NCHW `[N,1,48,48]`;
- batch dinâmico;
- normalização `pixel / 127.5 - 1`;
- saída FP32 `[N,4]`;
- validação de paridade host contra PyTorch e índice exclusivamente `PublicTest`.

### Handoff TFLite INT8

- layout NHWC `[1,48,48,1]`;
- batch fixo;
- escala e zero-point são registrados em `input_contract.json`/`output_contract.json`;
- `labels.txt`, allowlist de operadores e metadados de recursos acompanham o FlatBuffer;
- `emotion_model_data.h` e `emotion_model_data.cc` são gerados do mesmo bytecode, com os mesmos labels e tamanho.

### ESP32-S3-CAM N8R8

Os gates host verificam FlatBuffer até 1 MiB, arena com margem até 1,5 MiB e até 15 milhões de MACs. Medição real de arena e p95 de 200 ms no ESP32 permanecem `pending_external`; não são apresentados como aprovados por uma estimativa host.

## Métricas e relatórios

`sentience_train.metrics` é o único módulo de métricas usado por treino, seleção e release. Cada documento inclui accuracy, macro-precision, macro-recall, macro-F1, loss, precision/recall/F1 por classe, suporte e matriz bruta com ordem `angry,happy,neutral,sad`. Matrizes JSON usam `labels` e `matrices` nomeadas; CSV usa o formato longo `model,actual,predicted,count`; PNG é renderizado com Matplotlib/Agg de forma byte-determinística.

O release mantém evidências `PublicTest` e `PrivateTest` separadas. `PrivateTest` não é salvo como previsões: somente métricas, suporte, matriz e marcador de acesso são persistidos. Status finais são honestamente `accepted` ou `failed`, com gates, hashes e diagnóstico.

## Aceitação real e integridade

A aceitação nunca faz parte da suíte padrão (`pytest` usa `-m 'not acceptance'`). Para executá-la, forneça explicitamente `SENTIENCE_RELEASE_RUN_ID` e, opcionalmente, `SENTIENCE_RELEASE_RUNS_ROOT`/`SENTIENCE_RELEASE_EXPORTS_ROOT`:

```bash
SENTIENCE_RELEASE_RUN_ID=<id> pytest -m acceptance tests/test_release_acceptance.py
```

Sem o run e os insumos reais, `pytest -m acceptance` falha; não há skip nem fallback sintético. O export `all` só autoriza `PrivateTest` após validar schema `frozen-run/v1`, SHA-256 do CSV canônico, índice PublicTest, checkpoint, TFLite, operadores e gates host. Runs congelados anteriores, inclusive manifests que registram teacher, continuam consumíveis; a remoção afeta apenas a criação de novos runs.
