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
