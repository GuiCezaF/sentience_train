# Treinamento determinístico FER2013

O workflow público é `sentience-train train --config PATH`. Ele lê somente
`Training` para otimização e `PublicTest` para validação, early stopping,
calibração e seleção. `PrivateTest` permanece apenas na proveniência do CSV e
não é materializado pelo engine.

A seleção percorre DS-CNN `w025`, `w035`, `w050` e seeds `42,43,44`, parando no
primeiro width aprovado. PTQ precede QAT; QAT só responde aos gatilhos de
conversão ou queda de accuracy. Não existe teacher nem distilação para novos
runs. Se todos os candidatos falharem, o run imutável termina com
`selection_failed`, diagnóstico completo e nenhum acesso a `PrivateTest`.

A entrada é o CSV FER2013 canônico, validado por schema e SHA-256; as classes
de saída são `angry,happy,neutral,sad`, com mapeamento bruto `0,3,6,4`. A
normalização é `pixel / 127.5 - 1`, e os pesos de classe usam frequência
inversa da raiz quadrada, normalizada pela média.

Cada run contém configuração resolvida, índice congelado de `PublicTest`,
checkpoints, histórico, métricas, comparação de candidatos, hashes e manifesto
`frozen-run/v1`. Runs anteriores válidos continuam consumíveis pelo export,
mesmo que registrem metadados históricos de teacher.
