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
| `metal` | `MTLIOCommandQueue` carregando direto num `MTLBuffer` (storage shared). Requer macOS 14+ | Não documentado |

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

Antes de rodar `mmap`, rode `sudo purge` para esvaziar o page cache.

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
- O backend `metal` foi verificado só em compilação (`cargo check --target
  aarch64-apple-darwin`); ainda não rodou num Mac.
