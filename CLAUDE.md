# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é

Spike de pesquisa para rodar LLMs MoE grandes (alvo: Qwen3-235B-A22B) num Mac,
lendo os experts do SSD sob demanda. Há duas partes:

- **`expert-bench`** (Rust, `src/`): mede leituras reais do SSD. `gen` grava um
  arquivo de teste, `run` faz decode sintético com roteador aleatório e sem cache,
  e `replay` aplica um trace de roteamento real a um cache de experts em RAM.
- **`tools/routing-trace`** (Python + MLX, `uv`): captura o roteamento real do
  Qwen3-30B-A3B (um proxy do 235B), simula políticas de cache e mede a qualidade
  (perplexidade/KL) de experts em poucos bits.

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

cd tools/routing-trace
uv run capture.py --out traces/qwen3-30b-a3b        # baixa ~17 GB do modelo na 1ª vez
uv run sim.py traces/qwen3-30b-a3b --json traces/qwen3-30b-a3b.sim.json
uv run budget.py traces/qwen3-30b-a3b               # política de cache com pouca RAM
uv run tiered.py traces/qwen3-30b-a3b               # cache em duas resoluções (simulado)
uv run quality.py [--only <variantes>]              # qualidade de experts em baixa precisão
uv run basis.py                                     # base compartilhada entre experts
uv run dwq.py                                       # DWQ nas cópias q2 (treina escalas/biases)
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
- `<base>.json`: as dimensões do modelo e, para cada sequência, id, categoria,
  início e número de tokens. `replay.rs` lê esse JSON com um scan de chaves e
  não usa crate de JSON; mantenha o formato simples.

Os `.bin`/`.npy` estão no gitignore. Os JSONs de resultado em `traces/` são
versionados.

**Python**: `sim.py` define `Trace`, `split` (metade dos prompts para treino e
metade para teste, com categorias de um só prompt sempre no teste) e `TARGET`
(as dimensões do 235B e 14 GB/s de SSD, usados nas projeções de tok/s).
`budget.py` e `tiered.py` importam de `sim`; `tiered.py` importa de `budget`.
Rode os scripts de dentro de `tools/routing-trace`, porque os imports são
locais.

**`quality.py`**: `TieredMoE` envolve o bloco MoE de cada camada e manda parte
das chamadas para uma cópia em poucos bits (`low_bit_copy`), por regra (`all`,
`rank`, `p<frac>`, `skip<n>`, `rankskip[p]`). As cópias são requantizadas a
partir do q4, o que já custa ×1,003 (variante `q4 re-quant floor`). `dwq.py`
e `quality.py` usam o mesmo wrapper. Duas armadilhas já corrigidas: `copy.copy`
de um módulo MLX compartilha o `_no_grad` (descongelar a cópia descongelaria o
original), e os índices do roteador precisam de `stop_gradient` para treinar.

## O que já foi testado

Os números estão nos READMEs. Leia antes de repetir algo.

Funcionou e é o design atual para pouca RAM: LFU por camada com decaimento e
bypass, experts em duas resoluções escolhidas pelo peso do roteador (q4 nos
picks 1–4, q2 nos 5–8) e pular o pick 8 quando é miss. No replay real, o tok/s
no 235B foi de 2,09→3,33 (8 GB), 3,17→4,46 (16 GB) e 4,31→5,92 (24 GB), com
perplexidade ×1,041.

Não funcionou (não repita do mesmo jeito):
- compressão sem perda dos experts de 4 bits (zstd 1,07×, ideal 1,12×);
- q2 ou q3 em tudo, q2 em chamadas aleatórias, ou q3 + q2 por peso;
- rotação de Hadamard antes da quantização afim do MLX (piora);
- base compartilhada entre os experts de uma camada (D2-MoE/MoBE): os experts
  do Qwen3 não compartilham subespaço;
- DWQ com ~20 mil tokens de calibração: overfit e KL pior (precisa de ~2M);
- fixar experts populares, agrupar experts co-ativados no disco, aquecer o
  cache com o prefill, prefetch de 1 camada.

Próximas ideias, ainda não testadas: roteamento ciente de cache (Qualcomm,
arXiv 2412.00099; precisa recapturar o trace com os top-16 do roteador),
layout Matryoshka (o q2 como prefixo do q4), ajuste fino só do roteador e
kernel Metal real concorrendo com o I/O.

## Convenções

- Docs e resultados em pt-BR; código, comentários e mensagens de commit em inglês.
- Dependências Rust mínimas (`clap`, `libc`, `objc2*`). Evite adicionar crates.
- Projeções de tok/s são sempre para o 235B (94 camadas, top-8, ~10 MiB por
  expert em 4,5 bits), mesmo quando o trace vem do 30B.
