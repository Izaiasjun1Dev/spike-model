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

## Pouca RAM: compressão, política de cache e precisão (`budget.py`, `quality.py`)

25% dos experts do 235B são ~30 GB de RAM, mais do que a maioria dos Macs tem.

**Compressão sem perda não resolve.** Nos experts reais de 4 bits, os códigos
têm entropia de 3,71 bits (de 4). O zstd comprime os pesos 1,07×, e escalas e
biases 1,6×. O ideal teórico do expert inteiro é 1,12×. Além disso, o zstd
descomprime a ~1,5 GB/s numa thread, mais devagar que o SSD.

**Com cache pequeno, a política importa** (`uv run budget.py traces/qwen3-30b-a3b`).
Quando o cache não comporta o conjunto de experts de um token, o LRU global
fica trocando experts sem parar (0% de acerto com 4–6 GB). Um LFU com
decaimento, por camada e com bypass (só entra quem tem uso recente maior que
o de quem sai), mantém um subconjunto estável:

| RAM p/ experts (235B q4) | LRU global | LFU-decay | Belady (limite) |
|---|---|---|---|
| 4 GB | 0,0% | 26,6% | 38,7% |
| 8 GB | 31,1% | 41,6% | 58,8% |
| 16 GB | 56,6% | 60,4% | 76,1% |
| 32 GB | 79,5% | 81,4% | 90,0% |

**Menos bits por expert é a compressão de verdade, e o roteador diz onde ela
pode ser aplicada** (`uv run quality.py`). A referência é o modelo de 4 bits;
só os experts mudam. São 25 sequências e 10,3 mil tokens.

| experts | KL (nats) | top-1 igual | perplexidade |
|---|---|---|---|
| q4 requantizado (piso do método) | 0,008 | 96,3% | ×1,003 |
| q3 em tudo | 0,138 | 88,4% | ×1,076 |
| q2 em tudo | 0,405 | 76,8% | ×1,345 |
| q2 em 25% das chamadas, aleatórias | 0,083 | 89,3% | ×1,046 |
| q2 em 50% das chamadas, aleatórias | 0,172 | 85,2% | ×1,117 |
| **q2 só nos picks 5–8 do roteador (50%)** | **0,044** | **91,9%** | **×1,034** |
| q3 nos picks 1–4 e q2 nos 5–8 | 0,173 | 86,1% | ×1,112 |

Pôr em 2 bits os 4 experts de menor peso de cada token custa menos que q3 em
tudo e menos que q2 em só 25% das chamadas aleatórias. A precisão deve
seguir o peso do roteador, não ser fixa por expert.

### Cache em duas resoluções (`uv run tiered.py traces/qwen3-30b-a3b`)

No disco, cada expert fica em q4 e em q2. Os picks do topo (os 4 de maior peso
do roteador) precisam de q4; os de baixo usam qualquer cópia residente e, num
miss, leem a q2, que tem 56% dos bytes. A RAM é um orçamento em bytes por
camada, com LFU com decaimento e bypass, e o score é medido por byte. Uma
cópia q2 só vale pelos usos de baixo peso.

| RAM (235B) | q4: tok/s | q3: tok/s | **q4/q2 por peso: tok/s** |
|---|---|---|---|
| 8 GB | 3,0 | 4,3 | **3,9** |
| 16 GB | 4,4 | 6,9 | **5,8** |
| 24 GB | 6,5 | 11,0 | **8,5** |
| 32 GB | 9,4 | 18,6 | **12,5** |
| perplexidade | ×1,000 | ×1,076 | **×1,034** |

Os valores de tok/s são o limite do SSD. O replay real fica em ~70% disso. O
ganho do esquema vem de misses mais baratos, não de mais acertos: a taxa de
acerto é a mesma do q4. Ele fica no meio da fronteira qualidade × velocidade.
q3 em tudo é mais rápido, mas perde o dobro de qualidade. Misturar q3 e q2 é
pior que q3 sozinho.

## Pesquisa: ideias da literatura testadas no Qwen3 real

**E1: rotação de Hadamard antes de quantizar** (estilo QuIP#). Usei
`quality.py`, com a variante `hadamard` na cópia de baixa precisão.

| | sem rotação | com Hadamard |
|---|---|---|
| q3 em tudo | ×1,076 | ×1,109 |
| q2 em tudo | ×1,345 | ×1,610 |
| q2 nos picks 5–8 | ×1,034 | ×1,033 |

A rotação piora. A quantização afim do MLX com grupos de 64 já absorve os
outliers localmente. O ganho do QuIP# vem de codebooks de lattice e de
arredondamento guiado pela Hessiana, não só da rotação. Ressalva: os pesos de
partida já são q4. Partindo do bf16, o resultado pode ser outro.

**E2: base compartilhada entre experts** (estilo D2-MoE / MoBE). Usei
`basis.py`, com 5 camadas e as projeções gate, up e down. A tabela mostra a
energia dos pesos capturada com posto r:

| gate/up, camada típica | r = 256 (12,5%) | r = 512 (25%) |
|---|---|---|
| base única da camada | 21–24% | 35–39% |
| base própria de cada expert | 62–67% | 89–91% |
| matriz aleatória do mesmo formato | 58% | 87,5% |

Os experts do Qwen3 são quase de posto cheio, perto de uma matriz aleatória, e
não compartilham subespaço. Uma base comum captura menos que uma matriz
aleatória do mesmo posto. Subtrair o expert médio só remove ~1% da energia.
Para este modelo (128 experts pequenos), base comum na RAM + delta no SSD não
funciona. A medida é de energia pura, sem ponderar pelas ativações; uma SVD
ponderada pelas ativações poderia sair um pouco melhor.

**E3: pular os picks de menor peso** (estilo MC-MoE ODP). Os picks são
removidos em todos os tokens, sem renormalizar:

| | perplexidade | top-1 igual |
|---|---|---|
| pular o pick 8 | ×1,018 | 94,9% |
| pular os picks 7–8 | ×1,049 | 92,3% |

Pular o pick 8 só quando ele é miss custa ainda menos e corta ~1/8 das
leituras de miss. É um bom complemento ao cache em duas resoluções.

### Pular o pick 8 no miss + DWQ na cópia q2

**Pular o pick 8 quando ele é miss** (`tiered.py`, esquema `q4/q2 + skip 8`).
O pick 8 é o de menor peso do roteador. Se não estiver residente, não é lido
nem calculado.

| RAM (235B) | q4/q2 por peso | **+ pular o 8** |
|---|---|---|
| 8 GB | 3,9 tok/s | **4,3** |
| 16 GB | 5,8 | **6,5** |
| 24 GB | 8,5 | **9,7** |

Os valores são o limite do SSD. A qualidade (`quality.py`) passa de ×1,034
para ×1,041 quando o 8 é pulado em metade dos tokens, que é a taxa típica de
miss. Pular sempre custa ×1,050.

**DWQ nas cópias q2** (`dwq.py`). O professor é o modelo de 4 bits e só as
escalas e biases das cópias q2 treinam. A calibração usa 20 mil tokens dos
prompts de treino e a avaliação usa os prompts de teste.

| q2 nos picks 5–8 | KL | top-1 igual |
|---|---|---|
| antes | 0,045 | 92,1% |
| depois (lr 1e-5, 2 épocas) | 0,065 | 90,7% |

O DWQ piorou. Com 20 mil tokens ele se ajusta demais a esse texto; o DWQ do
mlx-lm usa ~2 milhões de tokens por padrão (2048 amostras × 1024). Se for
tentado de novo, deve ser feito direito no packer: é um custo único por modelo,
com mais dados e, de preferência, o modelo bf16 como professor.

### Roteamento ciente de cache (`cacheaware.py`)

O roteador passa a preferir experts que já estão em RAM, sem treino, na linha
de Skliar et al. 2024 (arXiv 2412.00099). Como o roteamento muda o cache e o
cache muda o roteamento, a medida é feita em loop fechado. O texto entra token
a token com KV cache, e cada camada tem seu próprio cache LFU com decaimento e
bypass, com o número de slots do orçamento no 235B. Os pesos de mistura são as
probabilidades do roteador, renormalizadas sobre os experts escolhidos.

- `bias<d>`: soma d ao log-prob dos experts residentes antes de escolher os 8.
- `swap<m>`: mantém os 8−m melhores picks originais e troca os m de menor peso
  pelos melhores residentes.

Cada variante grava seu trace em `traces/cacheaware/`, para o `replay` medir a
velocidade do mesmo roteamento. Os resultados abaixo usam 25 sequências e
10.271 tokens, comparados ao modelo de 4 bits.

| regra | acerto 8 GB | ppl 8 GB | acerto 16 GB | ppl 16 GB | top-1 igual (16 GB) |
|---|---|---|---|---|---|
| ref (roteador original) | 43,1% | ×0,998 | 61,5% | ×0,998 | 96,8% |
| swap1 | 50,0% | ×1,010 | 66,7% | ×1,004 | 95,3% |
| swap2 | 56,6% | ×1,027 | 71,5% | ×1,014 | 93,9% |
| swap4 | 68,8% | ×1,091 | 80,6% | ×1,050 | 90,0% |
| bias0.5 | 54,6% | ×1,016 | 74,0% | ×1,022 | 93,0% |
| bias1 | 64,0% | ×1,046 | 82,7% | ×1,057 | 89,5% |
| bias2 | 77,4% | ×1,227 | 94,1% | ×1,493 | 74,3% |

Com `--tiered`, a cópia q2 serve os picks 5–8:

| regra | ppl 8 GB | ppl 16 GB |
|---|---|---|
| ref | ×1,036 | ×1,036 |
| swap1 | ×1,046 | ×1,039 |
| swap2 | ×1,059 | ×1,048 |
| swap4 | ×1,117 | ×1,075 |
| bias0.5 | ×1,052 | ×1,053 |
| bias1 | ×1,076 | ×1,088 |

Os custos de qualidade quase se somam: q2 (×1,036) com bias1 (×1,05) dá
×1,08. O `bias` rende mais acerto por unidade de perda do que o `swap`, porque
pode trocar qualquer pick, não só os de peso baixo. O tok/s medido no replay
está no README da raiz. Com 16 GB, `bias0.5` sem q2 já é mais rápido e melhor
que o design q4/q2 + skip.

Uma armadilha: guardar os picks de cada token como arrays MLX pendentes prende
dois buffers Metal por camada, e a rodada morre com `Resource limit (499000)`
depois de ~5 mil tokens. Os picks passam para numpy ao fim de cada sequência.

### Experts em codebook (`codebook.py`)

Quantização vetorial no estilo AQLM/QuIP#, sem treino: cada linha do expert é
normalizada pela própria escala (RMS), dividida em vetores de d pesos, e cada
vetor vira o índice do centróide mais próximo entre K, num codebook por camada
e projeção ajustado por k-means. São log2(K)/d bits por peso. Com "Hadamard", o
codebook quantiza W·H (H é uma rotação ortonormal da dimensão de entrada), e a
ativação é girada do mesmo jeito. A reconstrução é guardada em 5 bits afins
para o forward (erro ~0,05, bem abaixo do erro do codebook) e, como no
`quality.py`, parte dos pesos de 4 bits.

| variante | bits/peso | erro relativo | ppl todos os experts | ppl picks 5–8 | KL picks 5–8 |
|---|---|---|---|---|---|
| q2 afim g64 (`quality.py`) | 2,5 | 0,41 | ×1,345 | ×1,034 | 0,044 |
| d4 k256 | 2,0 | 0,32 | ×183 | — | — |
| d4 k256 + Hadamard | 2,0 | 0,31 | ×1,398 | ×1,033 | 0,030 |
| **d4 k1024 + Hadamard** | **2,5** | **0,23** | **×1,139** | **×1,014** | **0,019** |

Sem a rotação, o codebook perde os pesos *outliers*, que o q2 afim preserva com
as escalas por grupo de 64. O resultado é catastrófico, mesmo com erro
relativo menor. Com a rotação, o codebook de 2,0 bits empata com o q2 afim de
2,5 bits nos picks 5–8, e o de 2,5 bits corta a perda pela metade nos dois usos
(×1,139 contra ×1,345 em todos; ×1,014 contra ×1,034 nos picks 5–8). Isso é só
k-means simples, sem ponderar pelas ativações e sem ajuste fino. Para virar
velocidade, falta um kernel Metal que decodifique o codebook dentro do matmul
(o MLX não tem). O d8 com 65.536 centróides ficou para depois (~4 h de
k-means).

Armadilha: a cópia em codebook do 30B tem ~29 bilhões de pesos. Guardada em 8
bits, ela ocupa ~31 GB, além dos 16 GB do modelo, e a máquina de 48 GB entra
em swap (uma avaliação levou 49 min). Em 5 bits cabe, e o `mx.set_cache_limit`
evita que os temporários do k-means se acumulem.
