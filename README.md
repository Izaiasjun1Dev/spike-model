# spike-model — benchmark de streaming de experts do SSD

Primeiro spike do engine: medir **quão rápido dá para puxar experts de um MoE
do SSD** num Mac, antes de desenhar o cache de experts e o "roda ou não roda".

O `expert-bench` simula o loop de decode *roteador escolhe top-k experts →
carrega → computa*, **sem cache de experts**. O resultado é o teto que o SSD
impõe; tudo que o cache acertar depois é ganho sobre esse número.

## Métodos comparados

| `--method` | Como lê | Passa pelo page cache? |
|---|---|---|
| `pread` | `pread` em pool de threads, arquivo aberto com `F_NOCACHE` (no Linux, `O_DIRECT`) | Não |
| `mmap` | `mmap` do arquivo inteiro + `memcpy` do expert (`MADV_RANDOM`) | Sim |
| `metal` | `MTLIOCommandQueue` carregando direto num `MTLBuffer` (storage shared). Requer macOS 14+ | Sim (medido: popula e lê do cache) |

Todos os offsets e tamanhos são alinhados a 16 KiB (página do Apple Silicon).

## Como rodar (no Mac)

Requer Rust (`curl https://sh.rustup.rs -sSf | sh`).

```bash
cargo build --release

# 1. Arquivo de teste. Para pread/metal o tamanho importa pouco; para mmap ele
#    precisa ser bem maior que a RAM, senão o page cache serve as leituras.
target/release/expert-bench gen --path ~/experts.bin --size-gb 64

# 2. Uma rodada: Qwen3-235B-A22B (94 camadas MoE, top-8, ~10 MiB por expert em ~4.5 bits)
target/release/expert-bench run --path ~/experts.bin --method pread --preset qwen3-235b-a22b --tokens 5 --verify

# 3. Matriz completa (método × lookahead × compute) → results/*.csv
scripts/sweep.sh ~/experts.bin qwen3-235b-a22b 5
```

Rode `sudo purge` antes de medir: `metal` e `mmap` populam o page cache, e até o
`pread` com `F_NOCACHE` lê do cache quando as páginas já estão lá. O `sweep.sh`
pede o sudo uma vez e faz `purge` antes de cada rodada.

Cada `run` imprime no stderr a fração do arquivo que está no page cache
(`mmap` + `mincore`) antes e depois do loop, e coloca os dois valores no
`RESULT` como `resident_pct` e `resident_end_pct`. Se `resident_pct` não for ~0,
o número de GB/s não é só SSD.

## Replay de traces reais

`expert-bench replay` usa o roteamento gravado por `tools/routing-trace` para
fazer leituras reais do SSD através de um cache LRU de experts em RAM, com
prefetch opcional pela predição de pre-gating. O tamanho do expert vem do
`--preset` (padrão: 235B), mesmo que o trace venha de um modelo menor.

```bash
target/release/expert-bench replay --path ~/experts96.bin \
  --trace tools/routing-trace/traces/qwen3-30b-a3b --cache-pct 25 \
  --compute-us 270 --max-tokens 200 [--prefetch] [--verify]

# 20/25/30% de cache, com e sem prefetch, nas sequências de teste do simulador
scripts/replay-sweep.sh ~/experts96.bin tools/routing-trace/traces/qwen3-30b-a3b
```

O arquivo de teste precisa de `camadas × experts × tamanho do expert` bytes
(~61 GiB para o trace do 30B com experts do 235B). Leituras pendentes de
prefetch são promovidas para a fila de demanda quando a camada precisa delas.

Primeira rodada (M5 Max, 11 sequências de teste × 200 tokens, compute 270 µs
por camada, tok/s projetado para as 94 camadas do 235B):

| cache | ≈ GiB no 235B | acerto | sem prefetch | com prefetch |
|---|---|---|---|---|
| 20% | 24 | 72,1% | 4,7 tok/s | 4,5 tok/s |
| 25% | 30 | 79,8% | 6,1 tok/s | 5,6 tok/s |
| 30% | 36 | 84,7% | 6,7 tok/s | 7,1 tok/s |

O acerto real bate com o `sim.py` (< 1 ponto). O prefetch de 1 camada quase não
ajuda: o compute por camada (~0,27 ms) é menor que uma leitura de expert
(~0,8 ms), e as previsões erradas gastam +27% de leitura. Com só 1–2 misses
por camada, o SSD fica em 8–12 GB/s: o gargalo passa a ser latência.

### Pouca RAM: LFU e experts em duas resoluções

`--cache-gb N` define o orçamento em GB no modelo-alvo. `--policy lfu` usa
LFU com decaimento por camada e bypass: um expert novo só entra se valer mais
por byte do que os que sairiam; senão é lido só para aquela camada.
`--tiered` faz a metade de menor peso dos picks de cada token usar uma cópia
de baixa precisão (`--low-bits 2.5`). Num miss, só essa cópia é lida (os
primeiros 56% do expert no arquivo de teste), e ela também pode ocupar o
cache. A metade de maior peso sempre precisa da cópia completa. Os detalhes
estão em `tools/routing-trace/tiered.py`, a versão simulada.

```bash
scripts/lowram-sweep.sh ~/experts96.bin tools/routing-trace/traces/qwen3-30b-a3b   # 8/16/24 GB
```

Resultados (mesmas condições acima, tok/s projetado para o 235B):

| RAM p/ experts | LRU | LFU | LFU + duas resoluções | **+ `--skip-last-miss`** |
|---|---|---|---|---|
| 8 GB | 2,09 (31,8%) | 2,36 (41,5%) | 2,93 (40,6%) | **3,33** |
| 16 GB | 3,17 (57,0%) | 3,28 (59,9%) | 4,10 (59,4%) | **4,46** |
| 24 GB | 4,31 (70,0%) | 4,43 (71,9%) | 5,58 (71,4%) | **5,92** |

Entre parênteses está a taxa de acerto. O modo em duas resoluções ganha
+29–40% sobre o LRU, com perplexidade ×1,034 em relação ao modelo de 4 bits
(medida pelo `tools/routing-trace/quality.py`). O ganho vem de misses mais
baratos, não de mais acertos. O LFU ajuda mais quanto menor o cache. As
listas de buffers livres ficam no máximo 2% acima do orçamento.
`--skip-last-miss` pula o pick de menor peso quando ele não está residente
(nem lê, nem calcula). Isso soma +6–14%, e a perplexidade vai a ×1,041.

### Prefetch 2–3 camadas à frente (preditor treinado)

`tools/routing-trace/predictor.py` treina, para cada distância d, um preditor
dos experts da camada l+d a partir do estado oculto da camada l. No teste
(decode, top-8), o acerto foi de 79,6% para 83,8% em d=2 e de 75,3% para 80,7%
em d=3, comparado ao pre-gating. Com os 12 melhores palpites, chega a 94,4% e
92,0%. O `replay` lê essas predições com `--prefetch-dist d` e
`--prefetch-width m`.

```bash
scripts/prefetch-sweep.sh ~/experts96.bin tools/routing-trace/traces/qwen3-30b-a3b-h 20,25
```

Resultados (cache LRU, mesmas condições acima, tok/s projetado para o 235B; a
rodada parou antes de d=3 com 25%):

| cache | prefetch | acerto | GB/token | tok/s |
|---|---|---|---|---|
| 20% | sem | 72,1% | 1,14 | **4,67** |
| 20% | pre-gate d=1 | 93,5% | 1,41 | 4,31 |
| 20% | preditor d=2, largura 8 | 92,5% | 1,41 | 4,41 |
| 20% | preditor d=2, largura 12 | 95,5% | 2,21 | 2,91 |
| 20% | preditor d=3, largura 8 | 90,8% | 1,45 | 4,26 |
| 20% | preditor d=3, largura 12 | 93,7% | 2,23 | 2,95 |
| 25% | sem | 79,8% | 0,82 | **6,03** |
| 25% | pre-gate d=1 | 95,0% | 1,05 | 5,76 |
| 25% | preditor d=2, largura 8 | 94,2% | 1,04 | 5,73 |
| 25% | preditor d=2, largura 12 | 96,4% | 1,71 | 3,94 |

Nenhum prefetch supera a rodada sem prefetch. O acerto sobe de 72% para 93%,
mas os palpites errados e os experts úteis que saem do cache aumentam a
leitura em 23–27% (largura 8) ou em 95–108% (largura 12). Com o SSD já a
10–13 GB/s, esse tráfego extra custa mais do que a latência que o prefetch
esconde. O preditor treinado é um pouco melhor que o pre-gating, mas não muda
a conclusão: com pouca RAM, só rende o que reduz os bytes lidos por token.

### Roteamento ciente de cache

`tools/routing-trace/cacheaware.py` faz o roteador preferir experts
residentes (`bias<d>` soma d ao log-prob deles; `swap<m>` troca os m picks de
menor peso) e grava as escolhas como trace. O replay usa esses traces nas
mesmas 11 sequências e 200 tokens de antes. Os traces incluem o prompt lido
token a token, então a base de comparação é o `ref` desta tabela, não as
tabelas acima. A qualidade está em `tools/routing-trace/README.md`.

| regra | ppl | 8 GB, LFU | 8 GB, LFU + duas resoluções + skip | 16 GB, LFU | 16 GB, LFU + duas resoluções + skip |
|---|---|---|---|---|---|
| ref | ×1,00 | 2,39 (43,0%) | 3,33 | 3,36 (61,4%) | 4,51 |
| swap1 | ×1,004–1,010 | 2,73 (49,8%) | 3,24 | 3,75 (66,5%) | 4,54 |
| swap2 | ×1,014–1,027 | 3,04 (56,1%) | 3,49 | 4,25 (71,3%) | 4,81 |
| bias0.5 | ×1,016–1,022 | 2,84 (54,0%) | 3,50 | 4,70 (73,5%) | 5,30 |
| bias1 | ×1,046–1,057 | 3,45 (62,9%) | 3,81 | **6,38** (82,0%) | 6,62 |

Os valores são tok/s projetados para o 235B, com o acerto entre parênteses. A
coluna ppl é sem duas resoluções (8 GB–16 GB). Com duas resoluções, os custos
quase se somam: o bias1 vai a ×1,076–1,088.

Com 16 GB, o roteamento ciente de cache substitui as duas resoluções. O
`bias0.5` só com LFU faz 4,70 tok/s com ×1,022, mais rápido e melhor que o
q4/q2 + skip (4,51 tok/s, ≈ ×1,04). O `bias1` faz 6,38 tok/s (+41%) com ×1,057.
Somar o q2 ao bias1 dá só +4%, por muita perda. O `swap1` não soma nada ao
skip, porque os dois atacam o mesmo pick de menor peso. Com 8 GB, as opções
ficam próximas (3,3–3,8 tok/s por ×1,04–1,08), porque o cache pequeno oferece
poucas alternativas boas.

Nas colunas com duas resoluções, as escolhas vieram do loop sem q2. O acerto
dos traces `-tiered` é quase igual (82,9% contra 82,7% no bias1 de 16 GB),
mas essas colunas não foram medidas com eles.

## Motor de verdade (`engine/` e `engine-rs/`)

Um comando só, pelo `engine/moe.py`:

```bash
cd engine
uv run moe.py build                                    # compila o motor (uma vez)
uv run moe.py get mlx-community/Qwen3-235B-A22B-4bit --fast   # baixa, empacota, cópias em codebook
uv run moe.py chat qwen3-235b-a22b-4bit --ram 16 --fast       # dimensiona o cache pela RAM e conversa
```

O motor roda Qwen3-MoE e a arquitetura DeepSeek-V3 / Kimi K2 (atenção MLA
absorvida, roteador sigmoid com bias de correção, experts compartilhados,
camadas densas iniciais, RoPE YaRN). No Moonlight-16B-A3B (a mesma
arquitetura do Kimi K2, 64 experts), a saída bate com o mlx-lm dentro do
ruído do próprio modelo (KL 2e-4 a 3e-3, top-1 98,5–100%).

`engine/pack.py` converte um modelo MLX em `model.safetensors` (tudo menos os
experts, que fica na RAM) e `experts.bin` (um slot por camada e expert,
alinhado a 16 KiB, lido com um único pread). O `moe-run` (`engine-rs/`, Rust
sobre mlx-rs) gera texto lendo os experts sob demanda, com o cache LFU por
camada e o roteamento ciente de cache (`--bias`). O `engine/engine.py` é a
referência em Python.

```bash
cd engine && uv run pack.py mlx-community/Qwen3-235B-A22B-4bit ~/packs/qwen3-235b
cd engine-rs && MLX_SYSTEM_DIR=<site-packages>/mlx cargo build --release
target/release/moe-run ~/packs/qwen3-235b --cache-gb 10 --bias 0.5 -p "..."   # ou --chat
```

O `MLX_SYSTEM_DIR` aponta para o MLX pré-compilado do pacote Python, então não
precisa de Xcode (a `mlx-sys` está em `engine-rs/vendor/` com essa opção). No
Qwen3-30B, a saída do `moe-run` é idêntica à do mlx-lm (64 tokens greedy).

Qwen3-235B no M5 Max, 10 GB de cache de experts, 100–200 tokens gerados,
perfil com `MOE_PROFILE=1` (ms por token no decode):

| versão | total | leitura | GPU | tok/s |
|---|---|---|---|---|
| primeira, bias 0,5 | 808 | 303 (+186 de cópia) | 251 | 1,24 |
| leitura direto na memória do MLX, buffers reusados | 527 | 307 | 203 | 1,90 |
| + prefetch da camada seguinte (pre-gating) | 445 | 202 | 225 | 2,25 |
| **+ arena de slots e um `gather_qmm` por projeção** | **304** | 216 | **83** | **3,29** |

A arena é um array por tensor do expert (`[slots, ...]`), compartilhado por
todas as camadas: um expert em RAM ocupa o mesmo slot nos nove arrays, a
leitura do SSD cai direto nele, e a camada roda seus experts com um único
`gather_qmm` por projeção, indexado pelo slot. Isso trocou ~30 lançamentos de
kernel por camada por ~6, e a espera da GPU caiu de 225 para 83 ms por token.
De quebra, os logits ficaram praticamente iguais aos do mlx-lm (KL 1e-4, top-1
100% em 64 tokens, `engine/test_rust.py`), porque o `gather_qmm` sobre um
array grande arredonda como o do modelo original.

O prefetch: o roteador da camada l+1 roda sobre a entrada da camada l, e os
experts que ele escolhe são lidos enquanto a GPU calcula a camada l. As
leituras de demanda passam sempre na frente, e os palpites que não são usados
têm os pedaços ainda na fila cancelados: 86% dos palpites são usados, e a
leitura sobe 11%. No replay com o SSD saturado, o prefetch tinha piorado o
tok/s; no motor, onde o SSD esperava a GPU, rende +14% (504 → 445 ms).

Orçamento de RAM e bias (`--bias`), 100 tokens, SSD a ~12 GB/s. A memória é
o pico medido pelo próprio MLX, que inclui a arena (cache + folga para um
prefill) e os 4,5 GB de pesos fixos:

| cache de experts | bias | acerto | GB/token | tok/s | pico MLX |
|---|---|---|---|---|---|
| 6 GB | 0 | 23,0% | 6,4 | 2,35 | 12 GB |
| 6 GB | 0,5 | 28,8% | 5,7 | 2,68 | 12 GB |
| **6 GB** | **1** | 33,5% | 5,1 | **3,03** | **12 GB** |
| 10 GB | 0 | 29,6% | 5,6 | 2,70 | 16 GB |
| 10 GB | 0,5 | 36,7% | 4,7 | 3,29 | 16 GB |
| **10 GB** | **1** | 43,4% | 3,9 | **4,05** | 16 GB |

Numa máquina de 16 GB, a configuração honesta é a de 6 GB de cache (pico de 12
GB). O prefill lê quase o modelo inteiro (~127 GB para 46 tokens, ~7 tok/s),
porque poucos tokens já usam quase todos os experts de cada camada.

### Experts em codebook no motor (`engine/packlo.py`, `--lo-frac`, `--lo-all`)

`packlo.py` acrescenta ao pack uma cópia de cada expert em codebook de 2,5
bits (`cbq.py`: linhas normalizadas, rotação de Hadamard, 1024 vetores de 4
pesos por camada e projeção; 0,56 dos bytes do q4; os resultados de qualidade
estão em `tools/routing-trace/README.md`). O `moe-run` roda essas cópias com
um kernel Metal próprio (`cbkernel.rs`), que reconstrói os pesos do codebook
dentro da multiplicação. Desde 2026-10-13 o `down_proj` da cópia fica em q3
afim por padrão (`--down 3`; `--down cb` mantém o codebook, `--down 4` copia
o q4 do pack): 0,64 dos bytes do q4 em vez de 0,56, KL 0,14 em vez de 0,19
nats no 30B com todos os picks em codebook, e o motor roda o `down` com
`gather_qmm` de 3 bits (sem o Hadamard na dimensão do `down`, o que também
libera o Moonlight-16B, cujo 1408 não é tamanho de Hadamard). Dois modos:

- **duas resoluções** (padrão com `experts_lo.bin` no pack): a metade de
  menor peso dos picks usa qualquer cópia residente ou lê a cópia em
  codebook; a outra metade exige q4; o pick de menor peso é pulado num miss.
  `--lo-frac` dá a fração do cache para cópias em codebook.
- **`--lo-all`**: todo pick pode usar a cópia em codebook (perplexidade
  ×1,139 no 30B, medida pelo `codebook.py`).

Qwen3-235B, 6 GB de cache (pico de ~12 GB), 100 tokens:

| modo | bias | acerto | GB/token | leitura | GPU | tok/s |
|---|---|---|---|---|---|---|
| só q4 | 0 | 23,0% | 6,1 | 284 | 107 | 2,52 |
| duas resoluções | 0 | 21,9% | 5,0 | 202 | 104 | 3,17 |
| só q4 | 1 | 33,6% | 5,0 | 201 | 98 | 3,27 |
| duas resoluções | 1 | 33,9% | 4,5 | 181 | 103 | 3,41 |
| **`--lo-all`, cache todo em codebook** | 0,5 | 39,8% | 2,7 | 81 | 100 | 5,20 |
| `--lo-all`, cache todo em codebook | 1 | 45,0% | 2,4 | 65 | 98 | 5,82 |
| **`--lo-all`, kernel otimizado** | **1** | 44,5% | 2,4 | 84 | **68** | **6,29** |
| `--lo-all`, 10 GB de cache (pico 15,5 GB) | 1 | 53,5% | 1,8 | 40 | 95 | 7,00 |
| **`--lo-all`, 10 GB, kernel otimizado** | **1** | 54,4% | 1,7 | 48 | **56** | **9,04** |

As duas resoluções rendem o que o replay previa quando o roteador é o
original (+26% com bias 0), mas quase nada com `bias 1` (+4%): o bias puxa
os picks de menor peso para os experts residentes, então os misses que
sobram são os de maior peso, que exigem q4. As duas técnicas disputam os
mesmos picks, e o bias ganha. Com todos os picks em codebook, os bytes por
token caem a 0,56 e o tok/s em 6 GB vai de 3,27 para 5,82 (×1,8), ao custo
de perplexidade ×1,14 (mais o do bias).

Um micro-benchmark de uma camada do 235B (`eval` por camada, como o motor)
mostrou que a sincronização custa um piso de ~0,16 ms por `eval`, o resto é
GPU de verdade, e a primeira versão do kernel do codebook era a maior parte
(~0,11 ms por chamada, contra ~0,02 do `gather_qmm`). O kernel otimizado
decodifica 4 grupos por lane e iteração (5 bytes de índices, `float4` da
ativação), guarda o codebook como `half4`, usa 8 linhas por threadgroup, e
um kernel fundido faz gate e up na mesma passada e já aplica `silu(gate)·up`:
456 → 94 µs por chamada (gate+up fundido: 133 µs). A GPU caiu de 98 para 68
ms por token, e o tok/s em 6 GB foi a 6,29 (9,04 em 10 GB). A leitura volta a
ser a maior parcela em 6 GB.

## Parâmetros

- `--preset qwen3-30b-a3b | qwen3-235b-a22b | deepseek-v3`: camadas MoE, top-k e
  tamanho do expert (gate + up + down) calculados a partir das dimensões do modelo.
  `--bits` (padrão 4.5 = 4 bits + escalas) muda a quantização.
- `--expert-mb`, `--layers`, `--topk`: sobrescrevem o preset.
- `--lookahead N`: quantas camadas à frente já têm leitura disparada. `0` = carga
  sob demanda; `N > 0` = prefetch com um preditor de roteador perfeito (cota superior
  do que prefetch pode render).
- `--compute-us`: espera ativa por camada depois que os experts chegam, fazendo as
  vezes da GPU. Com `--lookahead > 0`, mostra quanto do I/O some atrás do compute.
- `--threads`: workers de `pread`/`mmap` (padrão = top-k).
- `--verify`: confere cada expert carregado contra o padrão gravado pelo `gen`.

## Saída

```
method      pread
throughput  2.01 GB/s
decode      1.98 tok/s (506 ms/token, p50 505 ms)
layer stall p50 10.31 ms | p95 12.49 ms | p99 13.26 ms | max 13.64 ms
RESULT method=pread expert_bytes=2654208 topk=8 layers=48 lookahead=0 ... gbps=2.013 tok_s=1.976
```

- **throughput**: bytes de expert lidos por segundo.
- **decode**: tokens/s considerando só I/O (+ `--compute-us`), sem nenhum cache hit.
- **layer stall**: quanto tempo o "compute" ficou parado esperando os experts de uma camada.

(Os números acima são de um container Linux, só para ilustrar o formato.)

## O que este spike deve responder

1. Qual a banda real de leitura aleatória do SSD em blocos do tamanho de um expert,
   por método? Quanto falta para a banda sequencial anunciada?
2. `metal` (DMA direto para a GPU) ganha de `pread` + cópia? E com a GPU ocupada?
3. Quanto o prefetch (`--lookahead`) rende, e a partir de quando ele começa a
   atrapalhar por disputar o barramento de memória unificada?
4. Com esses números, qual o tok/s máximo por modelo sem cache, e qual taxa de
   acerto de cache seria necessária para chegar a uma meta (ex.: 5 tok/s)?

## Limitações conhecidas

- O roteador é uniforme aleatório. Roteadores reais têm localidade, que é
  justamente o que o cache vai explorar; isso fica para o próximo spike.
- `--compute-us` é espera ativa na CPU, não carga real de GPU, então não reproduz
  a disputa de banda de memória entre GPU e SSD. Para medir isso será preciso um
  kernel Metal de verdade rodando em paralelo.
- `--verify` lê só o cabeçalho de cada bloco de 4 KiB; ele confere offsets, não
  o conteúdo inteiro.
- Expert repetido dentro da mesma rodada pode vir do page cache no `metal`/`mmap`
  (com 1 token do 235B num arquivo de 96 GB, ~4% das leituras; com 5 tokens, ~17%).
  Para o número de SSD puro, use `pread` ou poucos tokens com `resident_pct=0`.

## Modo idêntico por faixa de RAM (2026-10-17)

Qwen3-235B no `moe-run` em modo idêntico ao modelo de 4 bits: `--bias 0
--no-tiered`, ou seja, roteador original, só experts q4, nada pulado (0
leituras em codebook e 0 picks pulados em todas as rodadas). Antes da rodada,
o `test_rust.py ~/packs/qwen3-30b --extra=--no-tiered` passou com KL de 1e-7 a
1,1e-4 e top-1 100%. Cada rodada é um processo novo (cache e KV frios), 200
tokens gulosos, três prompts (código em PT, chat em PT, matemática em EN), M5
Max, `experts.bin` com 0% no page cache antes e depois. Log em
`results/20261017-134215-235b-identical.log`.

| cache de experts | pico MLX | ≈ Mac | acerto | GB/token | tok/s (3 prompts) | média |
|---|---|---|---|---|---|---|
| 6 GB | 12,0 GB | 16 GB | 22,5–29,6% | 5,5–6,4 | 2,10–2,53 | 2,35 |
| 12 GB | 18,0 GB | 24 GB | 35,7–42,3% | 4,3–5,3 | 2,52–3,20 | 2,90 |
| 18 GB | 24,0 GB | 32 GB | 45,6–51,1% | 3,5–4,4 | 3,28–4,16 | 3,78 |
| 24 GB | 30,0 GB | 48 GB | 51,5–57,6% | 2,8–3,6 | 3,93–5,13 | 4,51 |

A primeira palavra sai em 6,3–7,3 s (prompts de 38–69 tokens). Em todas as
rodadas, GB/token × tok/s dá 13,2–14,5 GB/s: o SSD fica saturado, e o tok/s é
a banda do SSD dividida pelos GB por token. Num Mac com SSD mais lento, a
velocidade cai na mesma proporção. Os números são de resposta a frio; numa
conversa longa o cache esquenta e o acerto sobe.

O projeto mudou de pasta depois da compilação, e o `moe-run` procurava a
`libmlx.dylib` no caminho antigo (estas rodadas usaram `DYLD_LIBRARY_PATH`).
Recompilar com `MLX_SYSTEM_DIR` apontando para o `.venv` novo resolveu.

Numa conversa de 6 turnos com o mesmo processo (cache e KV persistem; o 6º
turno muda de assunto, de Rust para LGPD), 200 tokens por resposta, log em
`results/20261017-140257-235b-identical-session.log`:

| cache de experts | ≈ Mac | tok/s por turno | média | 1ª palavra (turnos 2–6) |
|---|---|---|---|---|
| 24 GB | 48 GB | 4,41 · 4,47 · 4,15 · 4,42 · 4,30 · 4,46 | 4,37 | 3,4–4,7 s |
| 6 GB | 16 GB | 2,50 · 2,43 · 2,34 · 2,47 · 2,30 · 2,47 | 2,42 | 4,4–6,5 s |

O cache não esquenta ao longo da conversa: o acerto se estabiliza já na
primeira resposta, e trocar de assunto não custa nada. A resposta a frio da
tabela anterior já é a velocidade de uso contínuo.

### Kimi-K2-Instruct (1T) em modo idêntico (2026-10-17)

Pack feito pelo `engine/pack.py` sem `--consume` (o download fica até o pack
ser validado). O passo do tokenizer falhou: o `AutoTokenizer` exige
`trust_remote_code` e a `transformers` instalada não lê o config do Kimi. O
`tokenizer.json` foi montado a partir do do Moonlight (mesmo `tiktoken.model`,
SHA-256 igual), com a regex trocada pela do `tokenization_kimi.py` (copiada,
não executada) e 5 tokens especiais renomeados (`<|start_header_id|>`,
`<|end_header_id|>`, `[EOT]`, `[UNK]`, `[PAD]`). Ele bate token a token com o
tiktoken + regex da Moonshot em 41 de 41 amostras (34.652 tokens: PT, EN,
chinês, código, CamelCase, emoji, template de chat).

Atenção: o `tokenizer.json` do pack do Moonlight tem a regex genérica da
conversão automática, sem `\p{Han}` e sem a separação por caixa da Moonshot,
e erra 2 dessas 41 amostras. O `test_rust.py` não pega isso porque compara
ids de tokens já prontos.

Antes da rodada, `test_rust.py ~/packs/moonlight-16b --model
mlx-community/Moonlight-16B-A3B-Instruct-4-bit --extra=--no-tiered` passou
(caminho DeepSeek-V3: KL de 2e-4 a 2,9e-3, no nível do ruído do próprio
modelo, top-1 98,5–100%). Mesmas condições do 235B (processo novo, 200
tokens, três prompts); log em `results/20261017-163340-kimi-identical.log`.

| cache de experts | pico MLX | ≈ Mac | acerto | GB/token | tok/s (3 prompts) | média |
|---|---|---|---|---|---|---|
| 3 GB | 19,6 GB | 24 GB | 7,8–13,6% | 11,1–12,1 | 1,23–1,36 | 1,29 |
| 8 GB | 24,0 GB | 32 GB | 13,9–22,1% | 9,9–11,2 | 1,32–1,54 | 1,42 |
| 16 GB | 31,5 GB | 48 GB | 20,7–28,4% | 9,0–10,3 | 1,44–1,70 | 1,56 |

A primeira palavra leva 15–21 s (prompts de 41–75 tokens). O SSD fica em
14,8–15,4 GB/s em todas as rodadas. O mínimo de memória é ~18 GB mesmo com
cache zero: além dos 6,8 GB fixos, a arena reserva uma camada inteira de
experts para o prefill (`experts + top_k` slots = 392 × 23,6 MiB = 9,7 GB).
Processar o prefill em lotes de experts reduziria essa reserva sem mudar o
resultado e é o que falta para o Kimi caber num Mac de 16 GB.

### Prefill em lotes e o kernel do mlx-lm (2026-10-17)

O `moe-run` reservava na arena uma camada inteira de experts para o prefill
(`experts + top_k` slots): 9,7 GB no Kimi. Agora o prefill lê e roda os
experts que faltam em lotes (`--prefill-batch 32`, o padrão; `0` volta ao
antigo), com a leitura do lote seguinte sobreposta à GPU do atual. A decisão
do cache depende só dos usos da chamada, então é tomada antes de ler, e só as
leituras que o cache admite seguram o slot além do seu lote: o cache termina
igual ao do caminho antigo. A arena passa a ter `2 × lote + 1 camada de cache
+ top_k` slots além do cache.

Ao validar com um prompt longo (`test_rust.py --long-prompt 800`, opção nova),
apareceu um defeito antigo. O MLX escolhe o kernel do `gather_qmm` pela razão
linhas/experts do array de pesos (segmentado se ≥ 16 linhas, índices
ordenados e linhas ≥ 4 × experts; senão, um produto matriz-vetor por linha),
e os dois arredondam diferente. O mlx-lm passa os experts da camada (128 no
Qwen3: segmentado a partir de 64 tokens); o motor passava a arena inteira,
então o kernel dependia do tamanho do cache. Agora toda chamada de prefill
segue a regra do mlx-lm: quando ele usaria o segmentado, a chamada recebe um
array só com os seus experts (preenchido com linhas nulas até 4 por expert);
senão, vai marcada como não ordenada. Os dois kernels dão a cada linha os
mesmos bits qualquer que seja a divisão em chamadas, o que torna o lote exato.

| teste (KL contra o mlx-lm) | motor antigo | motor novo (lote 32) |
|---|---|---|
| Qwen3-30B, prompt de 810 tokens, cache 0,5 GB | 1,5e-8 | 1,5e-8 |
| Qwen3-30B, prompt de 810 tokens, cache 100 GB | **1,27e-2 (falha)** | 1,5e-8 |
| Qwen3-30B, prompt de 124 tokens | cache-dependente | ~1e-7 |
| Moonlight, prompt de 787 tokens, cache 0,5 GB | 1,27e-3 (falha) | 1,27e-3 (falha) |
| Moonlight, prompt de 787 tokens, cache 100 GB | 1,63e-3 (falha) | 1,27e-3 (falha) |
| `test_rust.py` padrão, Qwen3-30B e Moonlight | passa | passa |

Com e sem lote, o motor novo dá logits bit a bit iguais, o mesmo texto e as
mesmas estatísticas de cache em 12 casos (Qwen3-30B e Moonlight; prompts de
28–33, 124–125 e 787–810 tokens; caches de 0,5 e 40 GB). O desvio do
Moonlight em prompt longo (2,2× o ruído do modelo, top-1 100%) já existia e
não vem dos experts: fica na arquitetura DeepSeek-V3 (provavelmente a
atenção MLA em sequências longas) e falta investigar.

Kimi-K2, mesmo prompt de 60 tokens, 40 tokens gerados
(`results/20261017-175352-kimi-prefill-batch.log`):

| cache | lote | arena | pico MLX | prefill | decode |
|---|---|---|---|---|---|
| 8 GB | 0 | 17,1 GB | 24,0 GB | 3,3 tok/s | 1,47 tok/s |
| 8 GB | 32 | 9,3 GB | **16,2 GB** | 3,4 tok/s | 1,47 tok/s |
| 8 GB | 64 | 10,9 GB | 17,8 GB | 3,4 tok/s | 1,47 tok/s |
| 1,5 GB | 32 | 3,3 GB | **10,2 GB** | 3,4 tok/s | 1,24 tok/s |

O prefill não perde velocidade, e o Kimi passa a rodar em modo idêntico com
pico de 10,2 GB, o que cabe num Mac de 16 GB. O `moe.py` calcula a reserva
com o lote (`PREFILL_BATCH`); num Mac de 16 GB, 70% da RAM deixa ~1 GB de
cache para o Kimi.
