# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é

Spike de pesquisa para rodar LLMs MoE grandes (alvo: Qwen3-235B-A22B) num Mac,
lendo os experts do SSD sob demanda. Há quatro partes:

- **`expert-bench`** (Rust, `src/`): mede leituras reais do SSD. `gen` grava um
  arquivo de teste, `run` faz decode sintético com roteador aleatório e sem cache,
  e `replay` aplica um trace de roteamento real a um cache de experts em RAM.
- **`tools/routing-trace`** (Python + MLX, `uv`): captura o roteamento real do
  Qwen3-30B-A3B (um proxy do 235B), simula políticas de cache e mede a qualidade
  (perplexidade/KL) de experts em poucos bits.
- **`engine/`** (Python): `pack.py` converte um modelo MLX em
  `model.safetensors` (tudo menos os experts) + `experts.bin` (um slot de 16 KiB
  alinhado por camada e expert); `engine.py` é a referência do bloco MoE em
  streaming, e `test_engine.py` a compara ao mlx-lm.
- **`engine-rs/`** (Rust sobre mlx-rs, binário `moe-run`): o motor de verdade.
  Gera texto lendo os experts do SSD, com cache LFU por camada, `--bias`
  (roteamento ciente de cache) e prefetch da camada seguinte. Compila contra o
  MLX do pacote Python (`MLX_SYSTEM_DIR=<site-packages>/mlx cargo build
  --release`), sem Xcode; a `mlx-sys` modificada está em `engine-rs/vendor/`.
  Valide mudanças com `cd engine && uv run test_rust.py ~/packs/qwen3-30b`
  (KL contra o mlx-lm dentro do ruído do próprio modelo; texto greedy
  idêntico não é critério, 1 ulp troca tokens). `MOE_PROFILE=1` imprime o
  tempo por token (leitura, GPU, resto). Os experts em RAM ficam numa arena
  (um array por tensor, `[slots, ...]`), e cada camada roda um `gather_qmm`
  por projeção indexado por slot. No 235B com 6 GB de cache (pico de 12 GB):
  ~3 tok/s; com 10 GB (16 GB): ~4 tok/s, `--bias 1`.

O README.md e o `tools/routing-trace/README.md` guardam os resultados de cada
experimento. Quando um experimento novo terminar, acrescente os números lá, no
mesmo estilo: tabelas, pt-BR, vírgula decimal.

## Comandos

```bash
cargo build --release                       # binário em target/release/expert-bench
cargo test                                  # testes unitários (data.rs, replay.rs)
cargo test <nome_do_teste>                  # um teste só

target/release/expert-bench gen --path ~/experts.bin --size-gb 64
target/release/expert-bench run --path ~/experts.bin --method pread --preset qwen3-235b-a22b --tokens 5 --verify
target/release/expert-bench replay --path ~/experts96.bin --trace tools/routing-trace/traces/qwen3-30b-a3b \
  --cache-gb 16 --policy lfu --tiered --skip-last-miss --compute-us 270 --max-tokens 200

scripts/sweep.sh <arquivo> [preset] [tokens]        # run: método × lookahead × compute → results/*.csv
scripts/replay-sweep.sh <arquivo> <trace-base>      # replay com 20/25/30% de cache, com e sem prefetch
scripts/lowram-sweep.sh <arquivo> <trace-base>      # replay com 8/16/24 GB, LRU/LFU/duas resoluções/+skip
scripts/prefetch-sweep.sh <arquivo> <trace-base> [20,25]  # replay: sem prefetch, pre-gate d=1, preditor d=2/3, largura 8/12

cd tools/routing-trace
uv run capture.py --out traces/qwen3-30b-a3b        # baixa ~17 GB do modelo na 1ª vez
uv run sim.py traces/qwen3-30b-a3b --json traces/qwen3-30b-a3b.sim.json
uv run budget.py traces/qwen3-30b-a3b               # política de cache com pouca RAM
uv run tiered.py traces/qwen3-30b-a3b               # cache em duas resoluções (simulado)
uv run quality.py [--only <variantes>]              # qualidade de experts em baixa precisão
uv run basis.py                                     # base compartilhada entre experts
uv run dwq.py                                       # DWQ nas cópias q2 (treina escalas/biases)
uv run capture.py --out traces/qwen3-30b-a3b-h --hidden --hidden-stride 2   # + estados ocultos (~5,7 GB)
uv run predictor.py traces/qwen3-30b-a3b-h --dists 1,2,3 --json traces/predictor.json
uv run cacheaware.py [--budgets 8,16] [--rules ref,bias1,swap2] [--tiered] [--limit n]
```

Os scripts de sweep pedem `sudo` uma vez e rodam `sudo purge` antes de cada
rodada. Nenhum número é medida de SSD se `resident_pct` (impresso no `RESULT`)
não estiver perto de 0: `metal`, `mmap` e até `pread` com `F_NOCACHE` leem do
page cache quando as páginas já estão lá.

## Arquitetura

**Arquivo de teste (`data.rs`)**: uma sequência contínua de blocos de 4 KiB. Cada bloco
começa com o próprio offset (u64 LE) e o resto é incompressível; o `--verify`
usa isso para checar o offset de cada leitura. Todo offset e tamanho é alinhado a
16 KiB (`ALIGN`, a página do Apple Silicon), para que I/O sem cache
(`F_NOCACHE` no macOS, `O_DIRECT` no Linux) nunca caia num bounce buffer.

**`run` (`main.rs` + `backend.rs` + `metal_io.rs`)**: `Backend` abstrai o
método de leitura. `backend.rs` tem um pool de threads sobre `pread` ou `mmap`.
`metal_io.rs` (só macOS, atrás de `#[cfg(target_os = "macos")]`) usa
`MTLIOCommandQueue` via `objc2-metal`. Os presets calculam o tamanho do expert
a partir das dimensões do modelo e de `--bits`. `--lookahead` simula um
preditor perfeito e `--compute-us` faz espera ativa na CPU no lugar da GPU.

**`replay` (`replay.rs`)**: o arquivo de teste faz o papel do modelo-alvo. O
expert (camada, e) fica num offset fixo e tem o tamanho do expert do preset,
então os bytes por miss são os do 235B mesmo com o trace vindo do 30B. O pool de
I/O tem duas filas: demanda (a camada atual espera por ela) antes de prefetch, e
um prefetch ainda na fila é promovido quando vira demanda. O cache tem orçamento
em bytes, com LRU global ou LFU por camada com decaimento e bypass. No modo
`--tiered`, a cópia de baixa precisão é lida como os primeiros `low_bytes` do
slot do expert e serve a metade de menor peso do roteador. `--skip-last-miss`
pula o pick de menor peso quando ele não está residente. `replay.rs` espelha
`tools/routing-trace/tiered.py`: ao mudar a política num lado, mude no outro.

**Formato do trace** (escrito por `capture.py`, lido por `sim.py` e `replay.rs`):
- `<base>.bin`: `uint8 [tokens, camadas, top_k]` com os picks do roteador
  ordenados por score e as sequências concatenadas. Só tem o decode.
- `<base>.pred.bin`: a predição de pre-gating da camada, feita na camada
  anterior (255 = sem predição).
- `<base>.scores.bin`: o peso do roteador de cada pick, em bits f16.
- `<base>.pred<d>.bin` (do `predictor.py`): `uint8 [tokens, camadas, 16]`, os
  16 melhores palpites do preditor treinado para a camada, feitos d camadas
  antes. O `replay` lê com `--prefetch --prefetch-dist d [--prefetch-width m]`
  (d = 1 continua lendo o `.pred.bin` do pre-gating).
- `<base>.hidden.bin` / `.hpicks.bin` (do `capture.py --hidden`): entrada do
  roteador de cada camada (f16 `[linhas, camadas, 2048]`) e os picks dessas
  linhas; por sequência, as linhas de prefill amostradas vêm antes das de
  decode (`hidden_start`/`hidden_prefill`/`hidden_decode` no JSON).
- `<base>.json`: as dimensões do modelo e, para cada sequência, id, categoria,
  início e número de tokens. `replay.rs` lê esse JSON com um scan de chaves e
  não usa crate de JSON; mantenha o formato simples.

Os `.bin`/`.npy` estão no gitignore. Os JSONs de resultado em `traces/` são
versionados.

**Python**: `sim.py` define `Trace`, `split` (metade dos prompts para treino e
metade para teste, com categorias de um só prompt sempre no teste) e `TARGET`
(as dimensões do 235B e 14 GB/s de SSD, usados nas projeções de tok/s).
`budget.py`, `tiered.py` e `predictor.py` importam de `sim`; `tiered.py` importa
de `budget`; `cacheaware.py` importa de `quality`.
Rode os scripts de dentro de `tools/routing-trace`, porque os imports são
locais.

**`quality.py`**: `TieredMoE` envolve o bloco MoE de cada camada e manda parte
das chamadas para uma cópia em poucos bits (`low_bit_copy`), por regra (`all`,
`rank`, `p<frac>`, `skip<n>`, `rankskip[p]`). As cópias são requantizadas a
partir do q4, o que já custa ×1,003 (variante `q4 re-quant floor`). `dwq.py`
e `quality.py` usam o mesmo wrapper. Duas armadilhas já corrigidas: `copy.copy`
de um módulo MLX compartilha o `_no_grad` (descongelar a cópia descongelaria o
original), e os índices do roteador precisam de `stop_gradient` para treinar.

**`predictor.py`**: para cada distância d, um mapa linear por camada de
`h_l` para os experts de l+d, que é o roteador de l+d (congelado) mais uma
correção treinada (`dw`, `b`). A perda é destilação: o alvo é a distribuição
completa do roteador de l+d aplicado a `h_{l+d}`. Não use entropia cruzada
contra o conjunto top-k: ela piora o acerto até no treino, porque é dominada
pelos picks imprevisíveis e os empurra para o top-k de todo mundo.

**`cacheaware.py`**: roteamento que prefere experts residentes, medido em loop
fechado (o roteamento muda o cache e o cache muda o roteamento). O texto entra
token a token com KV cache; cada camada tem um cache LFU com decaimento, em
arrays MLX. A regra é `keep` (os top picks originais que nunca mudam) mais
`bias` (somado ao log-prob dos residentes ao escolher o resto). Os pesos de
mistura continuam os do roteador. Cada variante grava o trace das escolhas em
`traces/cacheaware/`, para o `tiered.py`/`replay` medirem a velocidade da
mesma rodada.

## O que já foi testado

Os números estão nos READMEs. Leia antes de repetir algo.

Funcionou e é o design atual para pouca RAM: LFU por camada com decaimento e
bypass, experts em duas resoluções escolhidas pelo peso do roteador (q4 nos
picks 1–4, q2 nos 5–8) e pular o pick 8 quando é miss. No replay real, o tok/s
no 235B foi de 2,09→3,33 (8 GB), 3,17→4,46 (16 GB) e 4,31→5,92 (24 GB), com
perplexidade ×1,041.

Também funcionou: roteamento ciente de cache (`cacheaware.py`, sem treino). O
`bias<d>` soma d ao log-prob dos experts residentes. Com 16 GB, ele substitui
as duas resoluções: `bias0.5` só com LFU faz 4,70 tok/s com ×1,022 (o q4/q2 +
skip faz 4,51 com ≈ ×1,04), e `bias1` faz 6,38 com ×1,057. Somado ao q2, os
custos de qualidade quase se somam (bias1 + q2 = ×1,088) e o ganho é só +4%.
Com 8 GB, as opções ficam próximas (3,3–3,8 tok/s). O `swap<m>` (trocar os m
picks de menor peso) rende menos que o `bias` e não soma nada ao skip. As
escolhas no replay vieram dos traces de `traces/cacheaware/`, e a base é o
`ref` deles, não as tabelas antigas.

Não funcionou (não repita do mesmo jeito):
- compressão sem perda dos experts de 4 bits (zstd 1,07×, ideal 1,12×);
- q2 ou q3 em tudo, q2 em chamadas aleatórias, ou q3 + q2 por peso;
- rotação de Hadamard antes da quantização afim do MLX (piora);
- base compartilhada entre os experts de uma camada (D2-MoE/MoBE): os experts
  do Qwen3 não compartilham subespaço;
- DWQ com ~20 mil tokens de calibração: overfit e KL pior (precisa de ~2M);
- fixar experts populares, agrupar experts co-ativados no disco, aquecer o
  cache com o prefill, prefetch de 1 camada no replay (com o SSD saturado; no
  `moe-run`, onde o SSD espera a GPU, o mesmo prefetch rende +14%);
- prefetch 2–3 camadas à frente com o preditor treinado (`predictor.py`,
  acerto 83,8% em d=2): o acerto do cache sobe, mas lê 23–108% a mais e o
  tok/s cai (4,67→4,41 com 20% de cache). Com pouca RAM, só rende o que
  reduz os bytes lidos por token.

Próximo passo: ajuste fino só do roteador para chegar ao acerto do `bias1`
(~83% com 16 GB) com perda perto de ×1,02, o que daria ~6,4 tok/s com
qualidade melhor que a do design atual. É o candidato a treino em escala (com
muitos tokens; talvez na AWS). Outras ideias ainda não testadas: layout
Matryoshka (o q2 como prefixo do q4) e kernel Metal real concorrendo com o I/O.

## Convenções

- Docs e resultados em pt-BR; código, comentários e mensagens de commit em inglês.
- Dependências Rust mínimas (`clap`, `libc`, `objc2*`). Evite adicionar crates.
- Projeções de tok/s são sempre para o 235B (94 camadas, top-8, ~10 MiB por
  expert em 4,5 bits), mesmo quando o trace vem do 30B.
