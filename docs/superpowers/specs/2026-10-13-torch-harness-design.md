# Harness de pesquisa em PyTorch para GPU na AWS

Data: 2026-10-13. Estado: rascunho para revisão.

## Objetivo

Encurtar a pesquisa de compressão dos experts. Hoje cada medição de
qualidade (`tools/routing-trace/quality.py`, `codebook.py`) e cada treino
(`dwq.py`) roda em MLX, no Mac de 48 GB, um por vez, em 10 min a horas.
O harness novo roda as mesmas medições e treinos em GPU NVIDIA na AWS, como
jobs que sobem, rodam, gravam no S3 e desligam (SageMaker Training Jobs),
vários em paralelo, com os resultados num MLflow. O motor (`moe-run`,
Metal) fica fora: ele continua sendo medido no Mac.

O que o harness precisa reproduzir, com os mesmos números do MLX onde a
comparação importa:

- a quantização afim do MLX (g64, 2 a 8 bits, escala e bias por grupo), para
  que "q4", "q3", "q2" signifiquem o mesmo que nas tabelas atuais;
- o codebook do `cbq.py` (linhas normalizadas por RMS, Hadamard, k-means
  d4/k1024 ou d8/k65536);
- as regras de uso dos experts do `TieredMoE` (`all`, `rank`, `skip<n>`,
  `rankskip`, `p<frac>`), e a precisão mista por projeção (`down` em q3);
- a métrica: KL(ref ‖ variante) por token, top-1 e razão de perplexidade,
  no corpus de 25 sequências (prompts do `prompts.py` com continuação
  gulosa + 3 arquivos humanos do repo);
- o DWQ: destilação que treina só escalas/biases (afim) ou codebook/escalas
  de linha (VQ), com 1–2 M tokens.

Fora do escopo: roteamento ciente de cache em loop fechado (`cacheaware.py`,
depende do cache do motor), preditores, traces de roteamento, e qualquer
benchmark de SSD.

## Arquitetura

Um projeto Python novo em `research/` (uv, `torch`, `transformers`,
`safetensors`, `datasets`, `mlflow`, `boto3`, `sagemaker`), independente de
`engine/` e `tools/routing-trace/`, que só importa de lá o `prompts.py`
(texto dos prompts e arquivos humanos).

```
research/
  pyproject.toml
  research/
    quant.py        afim g64 (quantize/dequantize bit-exatos com o MLX), Hadamard,
                    codebook (k-means, encode, decode), fake-quant de um tensor
    moe.py          wrapper do bloco MoE do HF (Qwen3-MoE; DeepSeek-V3 depois)
                    com as regras do TieredMoE e cópias por projeção
    corpus.py       prompts + continuação gulosa + arquivos humanos; split
                    treino/teste igual ao sim.split; textos de calibração
    metrics.py      logprobs, KL/top-1/ppl contra uma referência gravada
    tasks/
      ref.py        gera a referência (logprobs top-1024 por token) e grava no S3
      quality.py    aplica variantes (afim, codebook, mista) e mede
      dwq.py        destilação de escalas/biases/codebooks
    run.py          ponto de entrada único: `python -m research.run <task> ...`
                    lê args, resolve caminhos locais/S3, loga no MLflow
  jobs/
    submit.py       cria um SageMaker Training Job por variante (instância,
                    imagem DLC de PyTorch, canais S3, spot opcional, MLflow URI)
    sync.py         baixa resultados do S3 para results/ e traces/
  tests/            quant bit-exato vs MLX (fixtures gravadas no Mac),
                    regras do wrapper num modelo Qwen3-MoE minúsculo na CPU
tools/routing-trace/mlflow_log.py   importa os JSON/CSV existentes como runs
```

### Modelo e memória

O modelo entra em bf16 pelo `transformers` (`Qwen/Qwen3-30B-A3B`, 61 GB;
`Qwen/Qwen3-235B-A22B`, 470 GB) a partir de uma cópia no S3 na mesma
região, com `device_map="auto"` sobre as GPUs do job.

A referência é configurável: `--ref q4` (padrão) quantiza os experts em
afim q4 g64 antes de tudo, igual ao checkpoint `mlx-community/*-4bit`, e as
variantes partem desse q4 como hoje (os números ficam comparáveis às
tabelas); `--ref bf16` usa o modelo cheio.

As variantes modificam os experts **no lugar** (uma cópia do modelo na
memória) quando a regra é `all`. As regras por pick (`rank`, `rankskip`,
`p<frac>`) precisam das duas cópias de cada expert; elas guardam a cópia de
baixa precisão reconstruída em int8 por expert (29 GB no 30B) e só rodam em
jobs com memória para isso (4×L40S para o 30B; 8×H100 para o 235B).

Logits de referência são calculados uma vez por (modelo, corpus) e gravados
no S3 como top-1024 por token (o mesmo truque do `mlx_lm.quant.dwq`); cada
variante custa só a passada do aluno.

### Tamanhos de job

| Tarefa | 30B | 235B |
|---|---|---|
| referência, variante `all` | 1 GPU de 80 GB ou 2×L40S | 8×H100 (p5/p5en) |
| variante por pick | 4×L40S (g6e.12xlarge) | 8×H100 |
| DWQ 1–2 M tokens | 4×L40S, ~1 h | 8×H100, ~2–4 h |

### Execução e resultados

`run.py` roda igual localmente (`--local`, qualquer máquina com GPU) e no
SageMaker (o container chama o mesmo entrypoint; `/opt/ml/input` e
`/opt/ml/output` mapeiam para os canais S3). Cada run grava um JSON no
formato atual de `traces/*.json` (as chaves `kl`, `top1`, `ppl_ratio`,
`bits`, `rule`, `rel_err`) e loga no MLflow: params (modelo, tarefa,
variante, bits, regra, ref, tokens), métricas (as mesmas) e o JSON como
artefato. O MLflow começa local (`mlflow server` com SQLite, artefatos num
bucket S3) e o URI vem de `MLFLOW_TRACKING_URI`; migrar para o MLflow
gerenciado do SageMaker é só trocar o URI.

`jobs/submit.py` recebe uma lista de variantes e abre um job por variante
(ou um job com várias, quando são rápidas), com `max_run` e spot opcionais,
e imprime os nomes. `jobs/sync.py` traz os JSONs de volta para
`tools/routing-trace/traces/` para o README seguir sendo a fonte das
tabelas.

### Tratamento de erros

Jobs do SageMaker falham sozinhos e deixam o log no CloudWatch; `submit.py`
não espera, `sync.py` reporta os que falharam. Falta de cota aparece na
criação do job e é reportada com o nome da cota. Um job nunca grava por
cima de um resultado existente: o nome do run inclui a variante e um
timestamp.

### Testes

- `tests/test_quant.py`: fixtures geradas no Mac com MLX (pesos aleatórios,
  `mx.quantize`/`mx.dequantize` em 2/3/4/8 bits, Hadamard) e comparadas
  bit a bit com o `quant.py` em CPU.
- `tests/test_moe.py`: Qwen3-MoE minúsculo (config aleatória, 2 camadas, 8
  experts) na CPU; a regra `all` com q8 fica perto do original, `skip1` zera
  o pick de menor peso, `rank` usa a cópia certa por pick.
- Validação de ponta a ponta: a variante `vq d4 k1024 had all` no 30B em
  GPU deve dar KL ≈ 0,185 e ppl ≈ ×1,14 como no Mac (mesma referência q4).

## Decisões

- PyTorch + transformers, não vLLM/SGLang: o que se mede é qualidade com
  pesos modificados, e precisa de acesso direto aos experts.
- Referência q4 por padrão, para manter as tabelas comparáveis.
- SageMaker Training Jobs, não EC2 com auto-desligar: cobrança por segundo,
  sem máquina para cuidar, MLflow gerenciado disponível depois.
- O corpus de avaliação continua o do `prompts.py`; a calibração do DWQ em
  escala vem de um dataset público (texto geral + código), não de
  continuações amostradas.
