# Treinamento determinístico FER2013

O fluxo novo é independente do pipeline legado:

```bash
uv sync --locked --group dev --group export
sentience-train train --config configs/default.yaml
sentience-train export --run-id <run-id> --format onnx|tflite
```

Somente `Training` participa da otimização. `PublicTest` é usado para métricas,
early stopping e a escolha do checkpoint. `PrivateTest` é validado durante a
leitura do CSV e aparece no manifesto de proveniência, mas nunca é criado como
um `Dataset` ou `DataLoader` pelo motor.

As repetições são garantidas no mesmo `uv.lock`, sistema operacional, hardware
e dispositivo. Em CPU, a ordem e as previsões devem ser iguais e as métricas
são comparadas com tolerância absoluta de `1e-8`. Não há promessa de igualdade
entre CPU e CUDA, nem entre stacks diferentes.

Cada execução finalizada é imutável em `runs/<UTC>-<uuid-curto>/`. Ela contém a
configuração resolvida, checkpoint escolhido, métricas, logs JSONL e manifesto
`frozen-run/v1` com hashes e proveniência. Também contém
`data/public_test.index.json`, com offsets, labels, linhas e hashes das únicas
linhas que exportação pode ler. Uma fixture CSV válida continua explicitamente
marcada como `release_evidence: false`; somente o SHA-256 canônico FER2013
recebe essa marca.
