# Compressão "ortogonal" dos experts: sem perda, layouts aninhados, esparsidade intra-expert e o lado Apple Silicon (out/2026)

Escopo: o que reduz os bytes em disco e por token num MoE em streaming (Kimi-K2-Instruct 4-bit MLX, 578 GB; Qwen3-235B), além de quantização e poda comuns. Todas as projeções de bytes por token abaixo são para o Kimi K2 com SSD de 14 GB/s, salvo indicação. "Sem perda" = bit-exato; "com perda" = muda os pesos.

Referência rápida do Kimi K2 (calculada a partir do `config.json` oficial — [config.json](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json)): hidden 7168, 61 camadas (camada 0 densa, 60 MoE), 384 experts roteados + 1 compartilhado, `moe_intermediate_size` 2048, top-8, MLA (q_lora 1536, kv_lora 512, 64 cabeças, nope 128 / rope 64 / v 128), vocab 163 840, embeddings não amarrados.

| Componente | Parâmetros | Bytes a 4,5 bits (q4 g64 afim) |
|---|---|---|
| Expert roteado (3 × 7168 × 2048) | 44,04 M | 24,8 MB |
| Todos os roteados (60 × 384) | 1 014,7 B | 570,8 GB |
| Experts compartilhados (60) | 2,64 B | 1,49 GB |
| MLP densa da camada 0 (3 × 7168 × 18432) | 0,40 B | 0,22 GB |
| Atenção MLA (61 × 101,1 M) | 6,17 B | 3,47 GB |
| Embedding + lm_head (2 × 163840 × 7168) | 2,35 B | 1,32 GB |
| Roteadores (60 × 384 × 7168) | 0,17 B | 0,09 GB |
| **Total** | **≈1 026 B** | **≈577 GB** (bate com os 578 GB do pack) |

Os não-experts são ≈11,7 B parâmetros (1,1 % do modelo): 6,6 GB a 4,5 bits, 12,5 GB a 8,5 bits (q8 g64), 23,5 GB em bf16. Por token, com cache frio, o decode lê 8 × 60 × 24,8 MB = 11,9 GB de experts → 1,2 tok/s a 14 GB/s; cada 1 % de hit no cache vale ~0,12 GB/token.

---

## 1. Compressão sem perda dos pesos (bf16 e, sobretudo, dos índices de 4 bits já quantizados)

### Takeaway
Em bf16 os ganhos sem perda são reais e bem medidos (DFloat11 ≈ 70 % do tamanho; ZipNN ≈ −33 %), porque o expoente tem baixa entropia. Para índices de 4 bits com escala por grupo (min/máx por 64, como o MLX afim) a redundância é pequena — o artigo de 2026 que mais se aproxima do limite de Shannon mede que formatos como AWQ/SmoothQuant por grupo ficam só 1,1–1,3× acima do limite, coerente com o 1,12× já medido no projeto. Os números de 2–10× que circulam são para INT4/FP4 por tensor (escala única), onde quase todo peso cai em 0 e a qualidade é ruim; não se aplicam a q4 g64.

### Cited Findings

**bf16 / fp8 (sem perda)**
- DFloat11: reduz LLMs e modelos de difusão em ~30 % (≈70 % do tamanho bf16), saída bit-a-bit idêntica; codificação de comprimento variável por frequência, decodificação na GPU com tabelas hierárquicas em SRAM, kernel de duas fases, descompressão por bloco do transformer; testado em Llama 3.3, Qwen 3, Mistral 3, FLUX.1, e Llama 3.1 405B (810 GB) em 8×80 GB. Ganhos citados são contra offload para CPU (2,3–46,2× em tok/s), não contra bf16 na GPU; não há número de latência absoluta — [arXiv 2504.11651](https://arxiv.org/abs/2504.11651)
- ZipNN: economia de ~33 % em modelos populares, >50 % em alguns; 17 % melhor que zstd em bf16, com 62 % mais velocidade; a fonte do ganho é a distribuição enviesada do expoente — [arXiv 2411.05239](https://arxiv.org/abs/2411.05239); [IBM Research](https://research.ibm.com/blog/Zip-NN-AI-compression). Há um seguimento para FP8/FP4 ([arXiv 2508.19263](https://arxiv.org/abs/2508.19263v1)); o trecho que vi fala em "até 83 % para FP8", ambíguo (tamanho final ou economia).
- NeuZip: codificação por entropia só dos bits de expoente, sem perda; o trecho encontrado cita razão de ≈1,5× num exemplo — [arXiv 2410.20650](https://arxiv.org/html/2410.20650v1)
- ZipServ (ASPLOS '26): formato de comprimento fixo "TCA-TBE" (tensor-core-aware triple bitmap) com decodificação em tempo constante + GEMM fundido (ZipGEMM); reduz o modelo em até 30 %, kernel até 2,21× vs cuBLAS, 1,22× fim a fim vs vLLM. Sem perda. Alvo: pesos bf16 em tensor cores NVIDIA — [arXiv 2603.17435](https://arxiv.org/abs/2603.17435)
- Não encontrei nenhum port de DFloat11 (ou equivalente) para Metal/MLX — busca em out/2026 sem resultado.

**Índices já quantizados (sem perda sobre a quantização)**
- "Approaching Shannon Bound with Lossless LLM Weight Compression" (2026): avalia bf16, FP8-E5M2, INT8, FP4-E2M1, INT4, SmoothQuant q8 e AWQ q4 em Qwen2-1.5B, Mistral-7B, Qwen-14B, DeepSeek-67B, Mixtral-8x22B e Llama-3.1-405B. **INT4/FP4 "puros" têm entropia de 0,6–1,0 bit** (gap 6–10×), **mas AWQ/SmoothQuant por grupo ficam só 1,1–1,3× acima do limite de entropia**, atribuído às frequências menos enviesadas e aos metadados de escala por grupo. Modelo de entropia: histograma de ordem zero por camada (sem contexto), rANS (b = 12 bits) com codebook por camada, tiles decodificáveis independentemente, decodificação em shared memory + GEMM fundido (base DietGPU + CUTLASS). Sem perda (bit-exato). Custo no TPOT mediano: Qwen-14B 71→81 ms (+14 %) a 1024 tokens; Mixtral-8x22B 110→170 ms (+55 %) — [arXiv 2606.15789](https://arxiv.org/html/2606.15789)
- "On the Compressibility of Quantized LLMs" (2024): só INT8. Sobre INT8 SmoothQuant, zstd/FSE/Huffman dão mais 1,66–2,22× (OPT-1.3b→30b); sobre LLM.int8(), 1,11–1,30×. Quantização por tensor comprime mais que por canal. Nada sobre 4 bits — [arXiv 2403.01384](https://arxiv.org/html/2403.01384)
- EntroLLM (2025): Huffman sobre pesos quantizados; afirma média de até 1,39 bit para pesos de 4 bits (texto do snippet corrompido; verificar no artigo) — [arXiv 2505.02380](https://arxiv.org/html/2505.02380)
- Deep Compression (2016, referência histórica): Huffman rendeu só 1,2× sobre a quantização — citado em [arXiv 2606.15789](https://arxiv.org/html/2606.15789)
- Projeto comunitário ECQ (drxddy): entropia de 1,12–1,54 bit em modelos de 0,1–1,5 B (Qwen2.5-1.5B 1,12 bit → "3,58×"); **não diz qual quantização de 4 bits foi usada nem o tamanho do grupo, não conta escalas/biases, os tok/s (42,9→140 num M3 Pro) são projeções de largura de banda e o kernel Metal "é o próximo passo"** — [github.com/drxddy/ecq](https://github.com/drxddy/ecq)
- Pedido de recurso no MLX (issue #3043, 22/jan/2026, mesmo autor): entropia medida de ~2,17 bits em pesos de 4 bits (1,84×), ~1,8 bit em 3 bits (1,7×), ~1,5 bit em 2 bits (1,3×), "em Llama/Qwen", sem detalhar; propõe rANS por linha com decode fundido ao GEMV (modo fundido e modo cache). **Nenhuma resposta dos mantenedores; fechado em 26/jan com o autor migrando para uma extensão externa (`mlx-entropy-ext`); sem números de Metal** — [ml-explore/mlx#3043](https://github.com/ml-explore/mlx/issues/3043)
- Decodificadores rANS em GPU NVIDIA (DietGPU) chegam a dezenas–centenas de GB/s, conforme citado em [arXiv 2606.15789](https://arxiv.org/html/2606.15789). Nenhum número equivalente para Metal foi encontrado.
- Na mesma linha, o artigo de 2026 registra que descomprimir antes/durante o GEMM exige memória extra para os pesos descomprimidos, o que pode anular a economia de banda — [arXiv 2606.15789](https://arxiv.org/html/2606.15789)

### Inferences
- A entropia do índice de 4 bits é função da granularidade da escala. Com min/máx por grupo de 64 (MLX afim), cada grupo usa por construção os níveis 0 e 15 e a distribuição interna é uma gaussiana truncada esticada pelo range do grupo → ≈3,5–3,6 bits de entropia → 1,1–1,15×. É exatamente o que o projeto mediu (ideal 1,12×, zstd 1,07×) e o que o artigo de Shannon reporta para AWQ (1,1–1,3×). Os 1,8× do issue do MLX e os 3,3× do ECQ só são plausíveis com escala por tensor/por canal, que é a quantização que ninguém usa em 4 bits por causa da qualidade.
- Logo, "compressão sem perda do q4 g64" é um beco: no máximo ~1,1–1,2×, com custo de um decodificador que precisa entregar ≥14 GB/s para não virar gargalo. Para render mais, a quantização precisa ser **projetada para a entropia** (grupos grandes com clipping, grade não uniforme, ou codebook/trellis) — ou seja, é lossy no desenho e sem perda só no armazenamento: o caminho do codebook de 2,5 bits do projeto e do QTIP já é esse.
- Onde a codificação por entropia pode valer no projeto: (a) nos **metadados** — a 4,5 bpw, escala+bias f16 por 64 são 0,5 bit/peso = 11 % do arquivo; comprimir escalas (8 bits + super-escala por linha, à la Q4_K) é lossy-determinístico e dá ~5 %; (b) nos **não-experts em bf16** se um dia ficarem em bf16 (23,5 GB → ~16,5 GB com DFloat11), irrelevante se ficarem a 4–8 bits.
- No streaming, o decodificador roda **uma vez por miss** (do SSD para a arena), não a cada uso, e pode rodar na CPU enquanto a GPU calcula; a sobrecarga de +14–55 % no TPOT dos artigos (GPU descomprimindo a cada camada) não se transfere diretamente. Mas o ganho é limitado pelo item anterior.

### Gaps
- Nenhuma medição de rANS/Huffman em Metal (GB/s) foi encontrada; só NVIDIA.
- Nenhum artigo mediu entropia/compressão de índices 4-bit com **escala por grupo de 64 min/máx** explicitamente (o mais próximo é AWQ g128 no artigo de Shannon).
- Não achei trabalho que use modelo de entropia **condicional/contextual** (vizinhos, posição no grupo) para índices quantizados; todos são de ordem zero. O ganho extra é desconhecido (provavelmente pequeno, pois os índices dentro de um grupo são quase independentes).

---

## 2. Quantização aninhada / Matryoshka (prefixo de 2 bits dentro de um código de 4 bits)

### Takeaway
MatQuant (DeepMind, 2025) mostra que o aninhamento int8⊃int4⊃int2 funciona, mas **com treino** (OmniQuant 10 M tokens ou QAT 100 M tokens, em 2–9 B); a fatia int2 fica 1–7 pontos acima de int2 independente e a fatia int4 fica dentro de ~0,5 ponto de int4 treinado à parte. **Fatiar sem treino é ruim** (int4 fatiado de int8 OmniQuant: 38,5 vs 73,1 no Mistral 7B). Layouts por plano de bits existem (Any-Precision LLM, AnyBCQ, GRINQH, Bit-by-Bit) e são viáveis em kernel, mas tudo foi mostrado em ≤13 B.

### Cited Findings
- MatQuant: int4/int2 aninhados nos MSBs do int8; um só modelo serve as três precisões. int2 MatQuant vs int2 padrão (média de tarefas): OmniQuant +1,0 (Gemma-2 2B), +3,1 (Gemma-2 9B), +3,0 (Mistral 7B); QAT +4,5 / +6,3 / +7,0. int4 MatQuant vs int4 independente: 67,03 vs 66,58 (2B), 74,33 vs 73,83 (9B), 73,62 vs 73,06 (Mistral) com OmniQuant. **int4 simplesmente fatiado do int8 OmniQuant: 62,87 / 72,26 / 38,51.** Bit-widths interpolados sem treino (int6, int3) ficam a ≤1,7 pontos das versões dedicadas. Variante "Extra Precision" (um bit a mais só para outliers, ≈2,05 bits): +3,2 a +4,9 pontos sobre int2 MatQuant. Treino: OmniQuant com 128 amostras C4 (≈10 M tokens; 20 M para o baseline int2); QAT com 100 M tokens (300 M para atn+FFN), em TPUv5e — [arXiv 2502.06786](https://arxiv.org/html/2502.06786)
- Any-Precision LLM (ICML 2024): PTQ leve que "sobrepõe" modelos de 3..n bits num único footprint comparável ao de n bits, com motor de serviço próprio; código aberto (SNU-ARC) — [arXiv 2402.10517](https://arxiv.org/abs/2402.10517)
- AnyBCQ (ICLR 2026): pesos como planos binários com fatores de escala; "expansão progressiva" refina escalas reaproveitando os códigos binários; até 3,0× sobre fp16 e 1,2× sobre o estado da arte multi-precisão — [arXiv 2510.10467](https://arxiv.org/html/2510.10467v2)
- GRINQH (2026): PTQ weight-only que atribui canais a níveis de precisão por magnitude de ativação; layout de memória **aninhado em planos de 2 bits** (cada plano é um refinamento sucessivo) num kernel GPU próprio; exige a "propriedade aninhada", que a quantização uniforme e alguns não uniformes (k-means) têm; melhor que baselines fixos/mistos a 3–4 bits e "ainda eficaz" a 2 bits, em Llama3 e Qwen3; arquivo de figura indica Qwen3-8B/GSM8K/RTX 4090 — [arXiv 2606.23419](https://arxiv.org/abs/2606.23419)
- Bit-by-Bit (ACL 2026): grade de l bits como versão grosseira da de h bits sob a mesma escala — o código de l bits é obtido **removendo os (h−l) LSBs**; "treina uma vez, serve qualquer precisão", com QAT progressiva por bloco e divisão de canais outliers — [arXiv 2604.07888](https://arxiv.org/pdf/2604.07888)

### Inferences
- Aritmética no MLX afim: se q4 = round((w−β)/s) e o prefixo de 2 bits é q4 ≫ 2, a dequantização do prefixo com a mesma (s, β) dá níveis {0, 4, 8, 12}·s + β — truncamento, com viés médio de −1,5·s. Um offset fixo de +1,5·s na dequantização do prefixo (custo zero no kernel) corrige o viés, mas o erro continua o de uma grade de 4 níveis fixada pela grade de 16: pior do que um q2 g64 otimizado (que escolhe seus próprios min/máx) e muito pior do que o codebook de 2,5 bits do projeto. O MatQuant diz com números que, sem co-treinar, a fatia é ruim.
- Bytes por token: um prefixo de 2 bits lido do disco custa 2 bits + os mesmos metadados (0,5 bit) = 2,5 bpw, igual ao q2 g64 independente. O ganho do Matryoshka é **só de disco** (uma cópia em vez de duas: 578 GB em vez de 578 + 320 GB) e de arena (um refinamento pode "completar" um expert já residente em 2 bits lendo só os LSBs: +2 bits em vez de +4,5). Isso exige layout por **planos de bits** (plano alto e plano baixo contíguos, cada um alinhado a 16 KiB) e um kernel próprio — o `gather_qmm` do MLX espera nibbles empacotados em uint32.
- Alternativa que não quebra a qualidade da base: **quantização residual** — base = a cópia de 2,5 bits em codebook (já existente), delta = (w − base) em afim de 2 bits g64 (2,5 bpw). Total 5 bpw (contra 4,5 do q4), "completar" um expert residente custa 2,5 bpw, e a base é a melhor possível. A qualidade de base+delta vs q4 precisa ser medida (sem fonte).
- Para "co-treinar" como o MatQuant em Kimi K2 seriam ~10 M tokens de OmniQuant por expert-set; nada disso foi mostrado acima de 9 B.

### Gaps
- Nenhum resultado de Matryoshka/aninhado em modelos >13 B nem em MoE.
- GRINQH e FlexMoE só em PDF; não extraí tabelas de qualidade por bit-width nem velocidade do kernel aninhado.
- Não há medição de qualidade de "prefixo truncado + offset" para afim g64; precisaria ser feita localmente (é barato: `quality.py` já tem o wrapper).

---

## 3. Esparsidade intra-expert (ler só parte do expert por token)

### Takeaway
Em SwiGLU sem ReLU a esparsidade é "suave": um artigo de 2026 mede que até 85–90 % dos neurônios de um expert roteado podem ser zerados **mantendo 95 % da acurácia** (Qwen3.5-35B-A3B: corte em 84,5 %; 122B-A10B: 87,4 %; Llama-4-Maverick: 90,8 %), mas a 85 % no Qwen3.5-35B a acurácia média cai de 70,9 para 67,5 (−4,8 %), e o método precisa calcular o `gate` denso primeiro (1/3 dos bytes). Em modelos ReLU (OPT) o Apple "LLM in a flash" lê ~2–3 % da FFN por token; isso não se transfere para SwiGLU sem re-treino (TurboSparse/dReLU).

### Cited Findings
- "Uncovering Intra-expert Activation Sparsity..." (2026): 8 MoEs de 1 B a 400 B (Granite-1B-A400M, OLMoE, DeepSeek-V2-Lite, GPT-OSS-20B, Qwen3.5-35B/122B/397B, Llama-4-Maverick). Ordena as saídas pós-ativação por expert e zera as menores; "corte de esparsidade" = máximo que mantém 95 % da média (ARC-C/E, HellaSwag, Winogrande, TruthfulQA). Cortes: 26,5 % (Granite-1B) a 90,8 % (Maverick); Qwen3.5-35B 84,5 %, 122B 87,4 %. Num expert do Qwen3.5-35B, 54,85 % das saídas são zero ou quase zero. Com shared experts densos, <2 % dos neurônios roteados bastam na análise (122B/397B). Sem preditor: gate denso + limiar por modelo (tabela no boot), O(n). Speedup da camada MoE 1,5–2,5× a 85–87 % (2,5× na 122B em RTX 4090); fim a fim só 1,0–1,2×. **Pesos residentes na GPU; não trata de carregar partes do expert do SSD; não reporta bytes.** vLLM a 85 % no Qwen3.5-35B: 67,5 vs 70,9 — [arXiv 2605.08575](https://arxiv.org/html/2605.08575v1)
- FloE (2025): esparsificação contextual de `gate` e `down` dos experts para inferência on-the-fly em GPU com pouca memória; observa alta esparsidade interna dos experts em todas as camadas — [arXiv 2505.05950](https://arxiv.org/pdf/2505.05950)
- Polar Sparsity (NeurIPS 2025): esparsidade contextual age em grão mais fino que o roteamento MoE (neurônios, não experts); nota que ativações suaves como SwiGLU tornam as ativações mais densas — [arXiv 2505.14884](https://arxiv.org/pdf/2505.14884)
- LLM in a flash (Apple): OPT 6.7B ≈97 % esparso, Falcon 7B "ReLU-ficado" ≈95 %; com janela de k = 5 tokens cada token precisa de ~2,4 % dos neurônios (OPT) e 3,1 % (Falcon); tráfego flash→DRAM por passo: 13,4 GB (baseline) → 0,9 GB (preditor) → 0,2 GB (preditor + janela), com metade do modelo em DRAM, em M1 Max e 16 bits; preditor low-rank (rank 128/1024) com 5 % FN / 7 % FP, 1,25 % do tamanho do modelo; **row-column bundling** (concatenar a linha de `up` e a coluna de `down` do mesmo neurônio) dobra o chunk (32 KiB no OPT) e sobe a leitura aleatória de 1,25 para 2,25 GB/s (latência de I/O 164→87 ms); leitura sequencial de 1 GiB >6 GiB/s no MacBook Pro M2; recomenda leituras ≥32 KiB em 32 threads. Total ~190 ms/token (OPT) e ~250 ms (Falcon) na CPU. **Só modelos ReLU**; o texto não trata SwiGLU — [arXiv 2312.11514](https://arxiv.org/html/2312.11514v2)
- TurboSparse/PowerInfer-2: SwiGLU e GeGLU têm esparsidade limitada e trocar por ReLU não basta; propõem dReLU com re-treino; Mistral/Mixtral passam a ativar 2,5 B / 4,3 B parâmetros por iteração, 2–5× no decode; TurboSparse-Mixtral-47B a 11,68 tok/s num OnePlus 12 (19 GB) e 2,13 tok/s com 7 GB — [arXiv 2406.06282](https://arxiv.org/html/2406.06282v2); [HackerNoon](https://hackernoon.com/turbosparse-mobile-22x-faster-mixtral-inference-on-powerinfer-2)
- FlexMoE (2026, poda aninhada intra-expert, **estática**, não por token): no Qwen2-57B-A14B poda 50 % dos parâmetros dos experts roteados sem fine-tune mantendo ~99,8 % do desempenho; um único fine-tune de recuperação a 40 % para os sub-orçamentos — [arXiv 2606.27866](https://arxiv.org/abs/2606.27866)
- Samoyeds: esparsidade estruturada 2:4 em experts para tensor cores esparsos (NVIDIA; sem equivalente em Metal) — [arXiv 2503.10725](https://arxiv.org/pdf/2503.10725)

### Inferences
- Bytes por expert do Kimi com o esquema "gate denso + up/down só dos neurônios ativos" a 85 % de esparsidade: 1/3 + 2/3 × 0,15 = 43 % → 10,7 MB em vez de 24,8 MB (2,3× menos). Com preditor low-rank no lugar do gate denso (como LLM in a flash/FloE), 15 % + preditor → ~3,7 MB + erro de predição. Mas o custo em qualidade a 85 % (−4,8 % de acurácia no Qwen3.5-35B) é bem maior que a tolerância atual do projeto (×1,02–1,04 em ppl); a esparsidade "grátis" provavelmente está em 50–60 % (o artigo mostra 55 % de saídas quase-zero), que daria 1/3 + 2/3 × 0,45 = 63 % dos bytes (1,6×).
- Granularidade de I/O: um neurônio do Kimi é 7168 pesos × 4,5 bits = 4 KB em `up` + 4 KB em `down` (precisa de `down` armazenado transposto, [2048, 7168]) = 8 KB com bundling; a página de 16 KiB força pacotes de 2 neurônios. Leitura de 300 neurônios × 16 KiB = 4,9 MB por expert, mas em leituras aleatórias de 16 KiB, que no M1 Max renderam 1,25–2,25 GB/s (vs >6 GB/s sequencial) — o ganho em bytes pode virar perda em tempo. O `expert-bench` já mede exatamente isso (tamanho de leitura × método); vale um preset "neurônio" antes de qualquer outra coisa.
- O cache em RAM passa a ser por neurônio (janela temporal, como no LLM in a flash), e a arena `[slots, ...]` com `gather_qmm` por slot deixa de funcionar: precisa de um kernel que faça gather de linhas dentro do expert. FlexMoE (estático, 50 % sem perda declarada) é compatível com a arena atual: basta empacotar experts com metade dos neurônios — é a opção de menor risco de engenharia, mas tem **um** resultado, num modelo (Qwen2-57B), sem reprodução.
- Os experts compartilhados do Kimi (1,5 GB a 4,5 bits) são densos por token e devem ficar residentes; a observação do artigo de que "com shared experts densos <2 % dos roteados bastam" é de análise, não de execução, e vale ser testada no Qwen3-30B como proxy.

### Gaps
- Nenhum trabalho mede **bytes lidos do SSD** com esparsidade intra-expert em SwiGLU; os ganhos reportados são de FLOPs/latência com pesos na GPU.
- Nenhum resultado em Kimi K2 ou DeepSeek-V3 (o maior MoE do artigo de 2026 é Llama-4-Maverick, 400 B).
- Curva completa esparsidade × qualidade (em ppl) para 30–60 % não foi extraída (figuras no HTML além do que li).

---

## 4. Base compartilhada + delta entre experts em tempo de execução

### Takeaway
D²-MoE (ICML 2025) e derivados (Camera, DeltaLLM) decompõem experts em base compartilhada + deltas de baixo posto com 40–60 % de compressão em Mixtral/Phi-3.5/DeepSeek-V2/Qwen2; nada em escala Kimi/DeepSeek-V3, nenhum resultado de streaming só do delta, e o projeto já viu que os experts do Qwen3 não compartilham subespaço.

### Cited Findings
- D²-MoE: base compartilhada (podada semi-dinamicamente) + deltas por expert comprimidos por SVD; >13 % melhor que outros compressores a 40–60 % de compressão em Mixtral, Phi-3.5, DeepSeek e Qwen2 MoE — [ICML 2025](https://proceedings.mlr.press/v267/gu25c.html); [GitHub](https://github.com/lliai/D2MoE)
- Camera (2025): compressão conjunta multi-matriz via análise de redundância de "micro-experts"; relaxa a hipótese do D²-MoE combinando fusão de experts com compressão de delta — [arXiv 2508.02322](https://arxiv.org/html/2508.02322v2)
- DeltaLLM (2025): deltas de baixo posto entre pesos compartilhados, para modelos densos — [arXiv 2501.18596](https://arxiv.org/pdf/2501.18596)
- Nenhuma fonte encontrada sobre delta streaming em DeepSeek-V3/Kimi K2 (busca em out/2026).

### Inferences
- Para o Kimi (384 experts por camada, muito mais que os 8 do Mixtral), a hipótese de base comum tende a ser ainda mais fraca do que no Qwen3 (128 experts), onde o projeto já falhou (`basis.py`). Só valeria revisitar com o delta **por grupo de experts co-ativados** (clustering), e o custo de qualidade a 40–60 % nos artigos é do nível de poda, não de "quase sem perda".

### Gaps
- Sem números de qualidade por modelo no D²-MoE além do agregado; sem Kimi/DeepSeek-V3.

---

## 5. Apple Silicon: modos de quantização do MLX, kernels e o que a comunidade decodifica rápido

### Takeaway
O MLX (out/2026) quantiza em `affine` (2/3/4/5/6/8 bits, grupos 32/64/128, escala+bias no dtype de entrada), `mxfp4` (g32, escala e8m0 → 4,25 bpw), `mxfp8` (g32) e `nvfp4` (g16, escala e4m3 → 4,5 bpw); não há codebook/LUT nativo. O `gather_qmm` Metal recebeu ganhos de 1,2–1,3× em 2026 para afim e nvfp4. Não encontrei medição publicada de `gather_qmm` a 2/3 bits vs 4 bits em M-series. Kernels LUT (T-MAC, BitNet.cpp) são CPU; IQ-quants em Metal eram lentos antes do M4 e um pacote comunitário (`mlx-iqk`, set/2026) traz IQ1/IQ2/IQ3 para experts roteados no MLX, sem números públicos.

### Cited Findings
- `mx.quantize(w, group_size, bits, mode)`: afim — grupos 32/64*/128, bits 2/3/4*/5/6/8, escala no dtype de entrada, com bias; mxfp4 — g32, 4 bits, escala e8m0, sem bias; mxfp8 — g32, 8 bits, e8m0; nvfp4 — g16, 4 bits, escala e4m3, `global_scale` opcional. Fórmula afim: s = (max−min)/(2^b−1), ŵ = round((w−min)/s); empacotamento em uint32 a partir dos bits baixos (8 nibbles por uint32) — [docs MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- PR "[Metal] gather_qmm improvement": ~1,2–1,3× nas tabelas de benchmark para afim e nvfp4 — [ml-explore/mlx#4572](https://github.com/ml-explore/mlx/pull/4572); PR que adiciona `gather_qmv_fast_medium` para K%6 com bits ≥4 (4/5/6/8), com testes em nvfp4/mxfp4/mxfp8 — [ml-explore/mlx#4598](https://github.com/ml-explore/mlx/pull/4598)
- Em ROCm (fork mlxcel), os caminhos rápidos de `gather_qmm` eram só afim e mxfp4 ficava mais lento (gpt-oss-20b prefill 512 tokens em 67,8 s) — sinal de que os modos não-afim recebem menos otimização — [lablup/mlxcel#2106](https://github.com/lablup/mlxcel/issues/2106)
- Backend Metal do HF Transformers usa kernels derivados do MLX (`affine_qmm_t`) com 2, 4 e 8 bits — [HF docs](https://huggingface.co/docs/transformers/v5.14.0/quantization/metal)
- `mlx-iqk` 0.1.3 (5/set/2026, MIT, autor "steadfastgaze"): IQ1_S_R4, IQ2_KS, IQ2_K, IQ3_K para projeções roteadas (MoE) e IQ4_KS/IQ4_K/IQ5_KS/IQ5_K/IQ6_K para tensores densos, com encoder CPU (código do ik_llama.cpp) e kernels Metal; testes bit-exatos vs CPU; `bits_per_weight()` inclui metadados; **sem velocidades nem qualidade na página** (aponta para `bench/RESULTS.md`) — [PyPI mlx-iqk](https://pypi.org/project/mlx-iqk/)
- llama.cpp Metal: IQ2_XS num 120B a ~0,1 tok/s num M1 Max e prompt de 9 min ("Apple Silicon really slow on below 4 bit"); IQ4_NL tão rápido quanto Q4_K; IQ3_S "funciona em Metal, mas ainda lento"; no M4 Max os IQ "ficaram comparáveis aos K-quants"; o commit do IQ4_XS registra TG-128 no M2 Max 28,8 → 21,7 t/s e "Metal / Apple Silicon don't like my quants" — [discussão 5617](https://github.com/ggml-org/llama.cpp/discussions/5617); [commit IQ4_XS](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/0becb22ac05b6542bd9d5f2235691aa1d3d4d307)
- T-MAC (CPU, LUT bit-serial): BitNet-b1.58-3B a ~30 tok/s em 1 núcleo e 71 tok/s em 8 núcleos de um M2 Ultra; kernel até 6,6× (3,6× médio) sobre llama.cpp CPU; afirma que a CPU pode igualar a GPU do mesmo aparelho — [arXiv 2407.00088](https://arxiv.org/html/2407.00088v2)
- bitnet.cpp TL2_0 (CPU): até 1,19× sobre T-MAC no M2 Ultra; 7,45 tok/s para um ternário de 100 B no M2 Ultra; TQ1_0 (1,69 bpw) é mais lento que TQ2_0 (2 bpw) por desalinhamento de memória — [arXiv 2502.11880](https://arxiv.org/html/2502.11880v1)
- QTIP (trellis "bitshift", códigos gaussianos gerados por computação): nenhuma implementação Metal/MLX encontrada — [arXiv 2406.11235](https://arxiv.org/pdf/2406.11235)
- DFloat11/ZipNN em Metal: nenhuma — busca sem resultados.

### Inferences
- Para o motor: mxfp4 corta 5,6 % dos bytes vs afim g64 (4,25 vs 4,5 bpw) e nvfp4 não corta nada; ambos têm grade E2M1 (8 magnitudes) que tende a perder para a afim de 16 níveis em qualidade — é uma troca lossy, não um ganho grátis (sem fonte comparando no Kimi).
- Os bits ímpares do afim (3, 5, 6) existem no `quantize` e no `gather_qmv_fast_medium`; q3 g64 = 3,5 bpw = −22 % de bytes por expert (19,3 MB). O projeto já viu que q3 em tudo não compensa em qualidade; mas q3 "em picks 3–4" + q2/codebook em 5–8 não aparece nos READMEs.
- Kernels LUT à la T-MAC são de CPU; no Mac o caminho é o que o projeto já faz (kernel Metal de codebook). A lição do TQ1_0 vs TQ2_0 (1,69 bpw mais lento que 2 bpw) confirma: só vale um formato de bits fracionários se a decodificação for alinhada (índices de codebook de 8/16 bits, não bitstreams).
- Vale baixar o `mlx-iqk` e medir no Qwen3-30B: IQ2_K/IQ3_K para experts roteados é exatamente o caso de uso, e os formatos IQ-K do ik_llama.cpp (não-lineares por bloco) costumam ter melhor ppl que afim no mesmo bpw.

### Gaps
- Nenhum número público de `gather_qmm` a 2/3 bits vs 4 bits em M-series (o projeto tem a melhor posição para medir).
- Velocidades e qualidade do `mlx-iqk` não estão na página do PyPI.
- Não achei a lista de modos que `gather_qmm` aceita na doc (a página do `quantize` não diz); os PRs indicam afim, nvfp4, mxfp4 e mxfp8.

---

## 6. Não-experts (atenção, embeddings, shared experts) do Kimi K2 e sua precisão

### Takeaway
Pelo `config.json`, os não-experts são ≈11,7 B parâmetros (1,1 % do modelo): 6,6 GB a 4,5 bits, 12,5 GB a q8 g64, 23,5 GB em bf16. Como são usados em todos os tokens, precisam ficar residentes; a 14 GB/s, só a atenção (3,5 GB a 4,5 bits) custaria 0,25 s/token se fosse lida do SSD.

### Cited Findings
- Hiperparâmetros usados no cálculo (tabela no topo) — [config.json](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json)
- Cartões de modelo: ~1 T total / 32 B ativos (≈1,04 T / 32,6 B exatos), 384 experts, 8 por token + 1 compartilhado, 61 camadas (1 densa), MLA, 64 cabeças (metade do DeepSeek-V3, para reduzir memória); nenhuma fonte dá a divisão por componente — [Composio](https://composio.dev/blog/notes-on-kimi-k2); [Replicate](https://replicate.com/moonshotai/kimi-k2-instruct)
- Nenhum estudo encontrado sobre a sensibilidade à quantização da atenção MLA/shared expert especificamente no Kimi K2.

### Inferences
- A divisão calculada (1 014,7 B roteados; 6,17 B atenção; 2,64 B shared; 2,35 B emb+head; 0,40 B MLP densa; 0,17 B roteadores) fecha em ≈1 026 B, e 578 GB ÷ 1 026 B = 4,5 bits/parâmetro — o pack é uniformemente q4 g64 (4 + 32/64), sem precisão mista.
- Orçamento residente para "1 T em 16 GB": não-experts a 4,5 bits = 6,6 GB, sobrando ~9 GB para arena + KV; a q8 (12,5 GB) não cabe. Mistura razoável: shared experts + `o_proj`/`kv_b` a 6–8 bits (são pequenos: shared 1,5 GB a 4,5 bits → 2,8 GB a 8,5) e o resto a 4,5. O embedding de entrada não precisa ficar residente (gather de uma linha por token: pode ficar em mmap); o `lm_head` (0,66 GB a 4,5 bits) precisa. Isso é um orçamento, não uma medição.
- Como os experts são 98,9 % dos bytes, qualquer técnica desta nota só importa aplicada aos experts; otimizar os não-experts move <2 % do disco.

### Gaps
- Sem fonte sobre quanta qualidade se perde quantizando a atenção MLA do Kimi a 4 bits vs 8 bits (os packs comunitários de 4 bits, incluindo o usado no projeto, não publicam ablação).

---

## Síntese quantitativa: bytes por expert do Kimi K2 (24,8 MB em q4 g64) sob cada técnica

| Técnica | Natureza | bpw ou fração | MB/expert | GB/token (8×60, cache frio) | Evidência | Escala mostrada |
|---|---|---|---|---|---|---|
| q4 g64 afim (atual) | lossy | 4,5 | 24,8 | 11,9 | — | — |
| zstd/rANS sobre q4 g64 | sem perda | ÷1,07–1,2 | 20,7–23,2 | 9,9–11,1 | projeto (1,07×/1,12×); [2606.15789](https://arxiv.org/html/2606.15789) (AWQ 1,1–1,3×) | 1,5–405 B |
| rANS sobre INT4 por tensor | sem perda sobre quant ruim | 0,6–1,0 bit entropia | ~4–6 | ~2–3 | [2606.15789](https://arxiv.org/html/2606.15789) | 1,5–405 B; qualidade de INT4 por tensor não serve |
| mxfp4 (g32, e8m0) | lossy | 4,25 | 23,4 | 11,2 | [docs MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) | formato, sem ablação |
| q3 g64 afim | lossy | 3,5 | 19,3 | 9,3 | docs MLX | — |
| q2 g64 / codebook 2,5 bits | lossy | 2,5 | 13,8 | 6,6 | projeto | 30 B / 235 B |
| Prefixo Matryoshka 2 bits (sem treino) | lossy | 2,5 (mesmos bytes que q2) | 13,8 | 6,6 | [MatQuant](https://arxiv.org/html/2502.06786): fatia sem treino é ruim | 2–9 B |
| Base codebook 2,5 + delta q2 (residual) | lossy | 5,0 total; +2,5 para completar | 27,5 | — | proposta, sem fonte | — |
| FlexMoE 50 % neurônios (estático) | lossy | ×0,5 | 12,4 | 5,9 | [2606.27866](https://arxiv.org/abs/2606.27866): 99,8 % no Qwen2-57B | 57 B, 1 artigo |
| Intra-expert 85 % com gate denso | lossy, por token | ×0,43 | 10,7 | 5,1 | [2605.08575](https://arxiv.org/html/2605.08575v1): −4,8 % acurácia | 1–400 B, pesos em GPU |
| Intra-expert 85 % com preditor | lossy, por token | ×0,15 + preditor | ~3,7 | ~1,8 | [LLM in a flash](https://arxiv.org/html/2312.11514v2) (só ReLU) | 7 B |
| Intra-expert ~55 % (zona "quase zero") | lossy, por token | ×0,63 | 15,6 | 7,5 | [2605.08575](https://arxiv.org/html/2605.08575v1) (55 % das saídas ≈0) | inferência |

Leituras em I/O: as fileiras de esparsidade trocam leituras sequenciais de 24,8 MB por leituras aleatórias de 8–16 KiB; o LLM in a flash mediu 1,25–2,25 GB/s nesse regime num M1 Max contra >6 GB/s sequencial — o `expert-bench` deve medir isso antes de qualquer decisão.

## Implicações para o motor (ordem sugerida por relação ganho/risco)

1. **Não investir em entropia sobre q4 g64** (teto ~1,1–1,2×, decodificador ≥14 GB/s necessário, nada em Metal). Se codificação por entropia entrar, é como parte de um quantizador projetado para isso (grupos grandes + clipping, ou codebook) — o codebook de 2,5 bits já ocupa esse lugar.
2. **FlexMoE-style (metade dos neurônios, estático)**: compatível com a arena e o `gather_qmm` atuais; medir no Qwen3-30B com `quality.py` (ordenar neurônios por norma × ativação média de calibração e cortar 30/40/50 %). Risco: um artigo, um modelo.
3. **Esparsidade por token com gate denso**: layout `[gate | up | downᵀ]` com `up`/`downᵀ` em pacotes de 2 neurônios (16 KiB) ordenados por co-ativação; medir primeiro no `expert-bench` a vazão de leituras de 16–64 KiB aleatórias; depois o custo em ppl a 40–60 % no Qwen3-30B. Exige kernel de gather de linhas intra-expert e cache por neurônio.
4. **Matryoshka**: só vale como layout de disco (uma cópia) e para "completar" experts residentes; a qualidade do prefixo truncado precisa ser medida localmente; a variante residual (codebook + delta q2) preserva a base.
5. **Experimentar `mlx-iqk`** (IQ2_K/IQ3_K nos experts) como alternativa ao codebook próprio, medindo ppl e tok/s no Qwen3-30B.
6. Não-experts: manter a 4,5 bits residentes (6,6 GB); subir shared experts/`o_proj` para 8 bits só se sobrar orçamento; o embedding de entrada pode sair da RAM.
