# routing-trace — traces reais de roteamento MoE

Roda um Qwen3-MoE no MLX, grava o top-k que o roteador de cada camada escolhe
a cada token de decode e salva o trace para o `expert-bench sim --trace`.

```bash
cd tools/routing-trace
uv run capture.py --out traces/qwen3-30b-a3b            # baixa ~17 GB na 1ª vez
```

- `traces/<nome>.bin`: `uint8`, formato `[tokens, camadas, top_k]`, cada linha
  ordenada pelo score do roteador; as sequências ficam concatenadas.
- `traces/<nome>.json`: dimensões do modelo e, por sequência, categoria,
  tokens de prompt/prefill/decode e onde ela começa no `.bin`.

Só o decode entra no trace: é a fase em que os experts vêm do SSD token a token.
O prefill processa muitas posições por vez e toca quase todos os experts.

Os prompts (`prompts.py`) misturam código, chat em PT/EN/ZH, matemática com
thinking, contexto longo, tradução e JSON, para medir localidade dentro de uma
tarefa e também o custo de trocar de tarefa.

O Qwen3-30B-A3B (48 camadas, 128 experts, top-8) é um proxy do 235B (94 camadas,
128 experts, top-8): mesma família e mesmo roteamento, mas não o mesmo modelo.

## Simulação (`sim.py`)

```bash
uv run sim.py traces/qwen3-30b-a3b --json traces/qwen3-30b-a3b.sim.json
```

Treino em metade dos prompts, teste na outra metade (o `chat-en` fica só no
teste, como domínio nunca visto). Cada sequência de teste começa com cache frio.

### Primeira rodada (Qwen3-30B-A3B, 21,7 mil tokens; projeção para o 235B a 14 GB/s)

| cache (% dos experts) | ≈ GiB no 235B | LRU hit | GB/token | tok/s (só SSD) |
|---|---|---|---|---|
| 15% | 18 | 62,5% | 2,99 | 4,7 |
| 20% | 24 | 71,8% | 2,25 | 6,2 |
| 25% | 30 | 79,4% | 1,64 | 8,5 |
| 30% | 36 | 85,3% | 1,18 | 11,9 |

- O roteamento tem forte localidade **temporal**: LRU simples fica a ~12 pontos
  do ótimo teórico (Belady). Fixar os experts mais populares do treino é muito
  pior (44% com 25%): a popularidade global não generaliza, a recência sim.
- Pre-gating (roteador da camada l no estado da camada l-1) acerta 85% dos
  experts da camada seguinte. Com prefetch, o acerto vai a 94% e o stall cai de
  1,64 para 0,45 GB/token, mas as previsões erradas custam +0,5 GB/token de
  leitura. Se compensa depende de quanto do I/O se sobrepõe ao compute: o
  replay real no `expert-bench` decide.
- Aquecer o cache com o roteamento do prompt quase não ajuda (+0,2 ponto).
- Agrupar experts co-ativados no disco reduz o desperdício em relação a grupos
  aleatórios (2,8× contra 3,7× com grupos de 4), mas ainda lê 2,8× mais bytes do
  que precisa: não vale para experts grandes.
- A taxa de acerto é estável entre tarefas (74–89% com 25%), inclusive no
  domínio não visto.
- Os misses carregam quase a mesma fração do peso do roteador que dos picks
  (18,8% contra 20,6% com 25%): pular misses de peso < 5% não economiza nada, e
  < 10% descarta 7% do peso (perda de qualidade a medir antes de considerar).
