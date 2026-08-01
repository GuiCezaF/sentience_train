# Release determinístico

O workflow público de release é:

```text
sentience-train export --run-id <run-id>
```

`all` é o padrão e publica `exports/<run-id>/all/`, com handoffs ONNX FP32 e
TFLite INT8, validação de `PublicTest`, gates host e, somente então, a reserva
atômica de `PrivateTest`. Os formatos `onnx` e `tflite` publicam handoffs
individuais e nunca criam o ledger reservado.

ONNX registra NCHW `[N,1,48,48]`, batch dinâmico e normalização
`pixel / 127.5 - 1`. TFLite registra NHWC `[1,48,48,1]`, batch fixo, escala,
zero-point, labels, operadores e arquivos C++ derivados dos mesmos bytes.

A transação completa reserva `PrivateTest` exatamente uma vez em
`exports/<run-id>/.private-test-access/<candidate-identity>/`. Crash, reserva
concorrente ou ledger sem bundle final bloqueia nova tentativa. Bundles
finalizados são imutáveis e só podem ser reutilizados após validação do digest
do manifesto e de todos os payloads.

Os gates host verificam 1 MiB de TFLite, 1,5 MiB de arena com margem e 15
milhões de MACs. Arena medida no ESP32-S3 e p95 de 200 ms permanecem
`pending_external`. Status final é sempre `accepted` ou `failed`; resultados
de `PublicTest` e `PrivateTest` permanecem separados.
