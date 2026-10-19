# Compressão estrutural de MoE: poda, fusão e baixo posto de experts (estado em out/2026)

Contexto: o alvo é reduzir os bytes totais (e, idealmente, os bytes lidos por token do SSD) do Kimi-K2-Instruct (384 experts/camada, top-8, 578 GB em 4 bits) e do Qwen3-235B-A22B (128 experts, top-8), sem perder qualidade. A base compartilhada entre experts (estilo D²-MoE/MoBE) já foi testada no Qwen3-30B neste projeto e não funcionou. As notas abaixo cobrem o que a literatura mede em modelos ≥235B e o que só existe em modelos menores (rotulado).

Convenção: "% de experts podados" é a fração de experts roteados removida por camada. "% dos parâmetros totais" é calculado quando a fonte dá os dois números. Vírgula decimal nas tabelas em pt-BR, mas os valores das fontes são copiados como estão (ponto) para não introduzir erro.

## 1. Poda de experts (REAP e seguidores): quanto cabe podar em modelos de 235B+ e a que custo

### Takeaway
REAP (Cerebras, ICLR 2026) é o único método com números em Kimi-K2 (1T) e Qwen3-Coder-480B: podar 25% dos experts é quase sem perda em código e tool-use (Kimi-K2: Code Avg 0,631→0,640; MC Avg 0,780→0,773); podar 50% mantém código (0,624) mas derruba conhecimento geral (MC Avg 0,780→0,643, −17,5% relativo) e multi-turn tool-use (BFCL Multi-Turn 0,355→0,164). A calibração define o que sobrevive: calibrar só em C4 destrói a geração de código.

### Cited Findings
- REAP = Router-weighted Expert Activation Pruning: score por expert = média sobre tokens roteados de (gate × norma da ativação do expert); poda one-shot, sem retreino; aceito no ICLR 2026; código em github.com/CerebrasResearch/reap — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006871); [GitHub](https://github.com/CerebrasResearch/reap)
- Modelos avaliados (Tabela 1 do paper): ERNIE-4.5-21B-A3B (64 experts), Qwen3-30B-A3B (128), Mixtral-8x7B (8), GLM-4.5-Air (128), Llama-4-Scout (16), Qwen3-Coder-480B-A35B-FP8 (160 experts, 480,2B totais, 35B ativos), Kimi-K2-Instruct-W4A16 (384 experts + 1 compartilhado, 1026,4B totais, 32B ativos, esparsidade 97,66%) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- **Kimi-K2-Instruct-W4A16** (Tabela 3; Eval+ / LiveCode / Code Avg / SWE-Bench-V / BFCL Non-Live / Live / Multi-Turn / Overall / MC Avg) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2):

| Poda | Método | Eval+ | LiveCode | Code Avg | SWE-V | BFCL NL | BFCL Live | BFCL MT | BFCL Ov. | MC Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| 0% | base | 0.828 | 0.434 | 0.631 | 0.554 | 0.840 | 0.802 | 0.355 | 0.666 | 0.780 |
| 25% | Frequency | 0.486 | 0.082 | 0.284 | 0.000 | 0.644 | 0.603 | 0.045 | 0.431 | 0.604 |
| 25% | EAN | 0.779 | 0.379 | 0.579 | 0.562 | 0.819 | 0.802 | 0.335 | 0.652 | 0.703 |
| 25% | REAP | 0.840 | 0.440 | 0.640 | 0.580 | 0.842 | 0.801 | 0.263 | 0.635 | 0.773 |
| 50% | Frequency | 0.112 | 0.000 | 0.056 | 0.000 | 0.255 | 0.397 | 0.003 | 0.218 | 0.439 |
| 50% | EAN | 0.722 | 0.253 | 0.487 | 0.576 | 0.778 | 0.767 | 0.173 | 0.573 | 0.587 |
| 50% | REAP | 0.819 | 0.429 | 0.624 | 0.576 | 0.785 | 0.743 | 0.164 | 0.564 | 0.643 |

- **Qwen3-Coder-480B-A35B-FP8** (Tabela 3) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2):

| Poda | Método | Eval+ | LiveCode | Code Avg | SWE-V | BFCL NL | BFCL Live | BFCL MT | BFCL Ov. | MC Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| 0% | base | 0.841 | 0.431 | 0.636 | 0.540 | 0.866 | 0.825 | 0.380 | 0.690 | 0.750 |
| 25% | REAP | 0.831 | 0.416 | 0.624 | 0.540 | 0.878 | 0.823 | 0.392 | 0.698 | 0.748 |
| 50% | REAP | 0.822 | 0.415 | 0.619 | 0.522 | 0.849 | 0.801 | 0.371 | 0.674 | 0.692 |
| 50% | Frequency | 0.007 | 0.012 | 0.010 | 0.000 | 0.200 | 0.392 | 0.000 | 0.197 | 0.506 |

- **GLM-4.5-Air** (106,9B, 128 experts; Tabela 2): REAP 25% Code Avg 0.580→0.585, Math Avg 0.882→0.879, MC Avg 0.747→0.674; REAP 50%: Code 0.515, Math 0.857, MC 0.554. Os métodos de fusão a 50% colapsam (M-SMoE Code 0.284, HC-SMoE 0.419) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- **Qwen3-30B-A3B** (Tabela 2): REAP 25% Code 0.558→0.551, MC 0.721→0.665; 50%: Code 0.541, MC 0.503. HC-SMoE 25% é melhor em MC (0.674) mas pior em código (0.505) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- Médias no texto: em QA de múltipla escolha, HC-SMoE e REAP perdem ~4% (25%) e ~13% (50%), excluindo os modelos grandes; em código REAP perde 1,9% (25%) e 6,9% (50%); nos modelos grandes, código não-agêntico perde 0,16% (25%) e 1,2% (50%); tool-use a 50% perde 5,9% — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- Blog da Cerebras: Qwen3-Coder-480B a 50% retém 97,6% de código não-agêntico e 96,7% no SWE-Bench; Qwen3-30B-A3B a 50% retém 95,9% de geração de código; GLM-4.5-Air 94,1% no LiveCodeBench a 50% — [Cerebras blog](https://www.cerebras.ai/blog/reap)
- Calibração: modelos até 110B usam 1 024 amostras empacotadas em 2 048 tokens; ≥110B usam 12 228 amostras de até 16 384 tokens sem truncar. Dados: C4 + evol-codealpaca (MC e código); ≥110B somam xlam-function-calling e SWE-smith-trajectories; WritingPrompts (escrita); tulu-3-sft-personas-math (matemática) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- **Dependência do domínio de calibração** (Tabela A8, calibração só em C4): REAP no Qwen3-30B-A3B cai para Code Avg 0.508 (25%) e 0.217 (50%), contra 0.551/0.541 com calibração de código; ERNIE-21B vai a 0.133/0.008; MC sob C4 fica "ligeiramente maior". O texto: código "abissal, com a maioria dos modelos falhando em gerar saída coerente". Modelos de grão fino (Qwen3, ERNIE) degradam mais (Fig. A8). Uma mistura geral (código + escrita + matemática, 3× amostras) é a que melhor preserva tudo (Fig. A9) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- Formulação unificada (jun/2026): os scores one-shot são a família S(b, α, β) = (frequência de roteamento, expoente do gate, expoente da ativação): Frequency=(0,0,0), SEER=(0,1,0), EAN=(0,0,1), REAP=(1,1,1). Para poda agnóstica de tarefa, as médias sem gate (MAN=(1,0,1), MSAN=(1,0,2)) vencem; para poda específica de tarefa, (0,1,1)/(0,2,2) vencem. Qwen3-30B-A3B a 25% com calibração C4: média geral Full 0.724, Frequency/SEER/EAN 0.411, REAP 0.619, MAN 0.707 (Eval+ 0.868 vs 0.871 do modelo cheio). A frequência de roteamento é a principal fonte de dependência da calibração. Só avaliou até 30B; memória cai ~pela metade a 50% (Qwen3: 56,92→29,93 GB bf16) — [arXiv 2606.15716](https://arxiv.org/html/2606.15716)
- NAEE ("Not All Experts are Equal", 2024): poda por busca exaustiva da combinação por camada que menos desvia a saída (64 amostras de calibração) + skip dinâmico do 2.º expert quando o peso é pequeno; só Mixtral-8x7B/8x22B — [arXiv 2402.14800](https://arxiv.org/pdf/2402.14800); descrição em [arXiv 2407.00945](https://arxiv.org/pdf/2407.00945)
- MoE-Pruner (2024): score Wanda × gate do roteador, poda não estruturada (esparsidade de pesos, não de experts) em Mixtral-8x7B/8x22B; a 50% de esparsidade mantém 99% após destilação por expert — [arXiv 2410.12013](https://arxiv.org/pdf/2410.12013)
- Fisher-MoE / "Less is MoE" (jun/2026): poda de dimensões intermediárias do FFN (não de experts inteiros), 50% de compressão do MoE; em Qwen1.5-MoE (calibração GSM8K) retém GSM8K 35,0 vs 35,9 base, enquanto poda por expert cai a 18,0; zerar 12 de ~1,35M dimensões escolhidas por GSM8K derruba GSM8K de 35,9% para 0,8% (importância muito concentrada por domínio); reduz parâmetros ativos em 15,4% e VRAM em 43,4%; não testou >100B — [arXiv 2606.05538](https://arxiv.org/html/2606.05538)
- "Finding Fantastic Experts" (2025): argumenta que estimativas one-shot (como NAEE) são piores que reestimação iterativa — [arXiv 2504.05586](https://arxiv.org/abs/2504.05586v1)
- Benchmark de composição MoE-XBench (ago/2026): "a razão de compressão não prevê a perda": podar 19,35% dos experts sobe a PPL 28,5%, quantizar 69,5% (Q4_K_M) sobe 0,7%; no Qwen3-235B a PPL sobe 54% com poda(20%)+quant, contra 31% no Qwen3.6-35B ("arquitetura importa mais que escala") — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)

### Inferences
- Para um assistente de uso geral (não só código), 25% é o teto seguro no Kimi-K2: a 50%, MC Avg cai 17,5% relativo e BFCL multi-turn cai pela metade. Os 97% "near-lossless" do blog valem para código, que é o domínio da calibração.
- Como REAP é um score (1,1,1) com forte peso de frequência, e a formulação unificada mostra que a frequência é o que mais torna a poda dependente da calibração, uma poda "geral" do Kimi deveria usar MAN/MSAN ou pelo menos uma mistura de calibração ampla (como a Fig. A9 do REAP). Isso só foi medido até 30B.
- Podar não reduz os parâmetros ativos por token (top-8 continua top-8; os checkpoints continuam "A35B"/"A32B"). O que muda para o motor de streaming é o universo de experts por camada (384→288 ou 192), o que sobe a taxa de acerto do cache a orçamento fixo.

### Gaps
- Não há número de REAP para Qwen3-235B-A22B (só Qwen3-30B e Qwen3-Coder-480B) nem para DeepSeek-V3 ou GLM-4.5 cheio; o MoE-XBench tem Qwen3-235B, mas só com poda a 20% medida por PPL.
- O paper REAP não publica tamanho em GB nem tok/s; não discute poda não uniforme por camada.
- Não encontrei o paper SEER-MoE original; só aparece como caso (0,1,0) da formulação unificada.

## 2. Checkpoints prontos (tamanhos e licenças)

### Takeaway
A Cerebras publicou checkpoints REAP para Qwen3-Coder-480B (363B a 25%, 246B a ~49%), GLM-4.5-Air, GLM-4.6 (218B/252B/268B), MiniMax-M2 e Kimi-Linear, mas **não** para Kimi-K2; o único Kimi podado é de terceiros (Kimi-K2.5 REAP-50%, 519B params, 288 GB em INT4). Para o Qwen3-Coder-REAP-363B existem GGUFs da Unsloth de 104 GB (IQ1_S) a 219 GB (Q4_K_M).

### Cited Findings
- cerebras/Qwen3-Coder-REAP-363B-A35B: 363B totais, 35B ativos, 120 de 160 experts por camada (25%), BF16, calibração evol-codealpaca + xlam-function-calling-60k + SWE-smith-trajectories, Apache 2.0; variante FP8 separada; tabela de benchmarks "TBD for BF16 model" — [HF](https://huggingface.co/cerebras/Qwen3-Coder-REAP-363B-A35B)
- cerebras/GLM-4.6-REAP-218B-A32B: 40% podado, 96 de 160 experts, MIT, derivado de GLM-4.6-FP8, destinado a quantização posterior; 7 quantizações derivadas no model tree — [HF](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B)
- cerebras/GLM-4.5-Air-REAP-82B-A12B: 25% (96 de 128), HumanEval 92,7→89,6, LiveCodeBench 39,6→42,9, MIT — [espelho GitCode do card](https://gitcode.com/hf_mirrors/cerebras/GLM-4.5-Air-REAP-82B-A12B/blob/main/README.md)
- Coleção Cerebras lista ainda Qwen3-Coder-REAP-246B-A35B-FP8 e 25B-A3B, GLM-4.6 em 218B/252B/268B, MiniMax-M2 e Kimi-Linear — [HF collections](https://huggingface.co/collections/cerebras); [Cerebras blog](https://www.cerebras.ai/blog/reap)
- Os endpoints públicos da Cerebras servem só modelos não podados — [Cerebras inference docs](https://inference-docs.cerebras.ai/models)
- kwanhee/Kimi-K2.5-REAP50-INT4-W4A16: base Kimi-K2.5, 192 de 384 experts, 519B params, 288,4 GB em 58 shards, INT4 g32 simétrico herdado do release (atenção, expert compartilhado, MLP denso, lm_head e torre de visão em bf16). Vs. Kimi-K2.5 denso: OpenLLM v1 80,77 vs 82,43 (98,0%), AIME25 0,910 vs 0,957 (95,1%), GPQA-diamond 0,788 vs 0,895 (88,0%), MATH500 0,978 vs 0,964 — [HF](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)
- unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF: UD-IQ1_S 104 GB, UD-IQ1_M 116 GB, Q2_K 132 GB, UD-Q2_K_XL 139 GB, UD-IQ3_XXS 155 GB, Q3_K_M 173 GB, IQ4_XS 197 GB, Q4_K_M 219 GB, Q5_K_M 257 GB, Q6_K 298 GB, Q8_0 386 GB, BF16 726 GB; sem benchmarks — [HF](https://huggingface.co/unsloth/Qwen3-Coder-REAP-363B-A35B-GGUF)
- Outros REAP de terceiros: MiniMax-M2-REAP-162B-A10B-AWQ-4bit, Qwen3.6-28B-REAP20-A3B-GGUF, KAT-Coder-V2.5-REAP-50-NVFP4A16 — [HF cyankiwi](https://huggingface.co/cyankiwi/MiniMax-M2-REAP-162B-A10B-AWQ-4bit/blob/main/README.md); [HF barozp](https://huggingface.co/barozp/Qwen3.6-28B-REAP20-A3B-GGUF)

### Inferences
- Para o Kimi-K2-Instruct é preciso rodar o REAP localmente (o repo é público): o paper fez isso sobre o checkpoint W4A16, com 12 228 amostras de até 16k tokens, o que exige passar o modelo inteiro pela calibração (na prática, uma máquina com o 1T em memória ou um pipeline camada a camada).
- O Kimi-K2.5-REAP50 de terceiros mostra que a poda a 50% custa 12% em GPQA mesmo com calibração desconhecida, coerente com a queda de MC no paper.

### Gaps
- Nenhum card da Cerebras informa o tamanho em GB; o paper diz que Kimi-K2 4-bit + 50% de poda dá "redução total de 87,5%" em relação ao bf16 cheio.
- O Kimi-K2.5 não é o Kimi-K2-Instruct; os números não são transferíveis diretamente.

## 3. Fusão de experts (M-SMoE, HC-SMoE, MergeMoE, REAM)

### Takeaway
Em modelos de grão fino (Qwen3, GLM-4.5-Air, ERNIE), a fusão perde para a poda em tarefas generativas e colapsa a 50% (M-SMoE no GLM-4.5-Air: Code Avg 0,580→0,284; HC-SMoE no Qwen3-30B: WildBench 0,811→0,008). HC-SMoE só ganha em múltipla escolha a 25%. Nenhum método de fusão foi medido em ≥235B.

### Cited Findings
- M-SMoE / MC-SMoE (ICLR 2024): alinha permutações de neurônios, agrupa experts em torno de dominantes, funde ponderando por frequência; depois decompõe em baixo posto + esparso. Reporta até 80% de memória e 20% de FLOPs a menos "quase sem perda", principalmente em Switch Transformer; o código suporta Mixtral-8x7B — [arXiv 2310.01334](https://arxiv.org/abs/2310.01334v1); [GitHub](https://github.com/UNITES-Lab/MC-SMoE/blob/main/README.md)
- HC-SMoE (ICML 2025): clustering hierárquico pelas saídas dos experts, sem retreino; remove 25% e 50% dos parâmetros de experts em Qwen1.5-MoE e Mixtral — [arXiv 2410.08589](https://arxiv.org/abs/2410.08589)
- No REAP (Tabelas 2/A6/A7), os resultados de fusão em modelos de grão fino:

| Modelo | Poda | M-SMoE Code/MC | HC-SMoE Code/MC | REAP Code/MC | Base Code/MC |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 25% | 0.537 / 0.558 | 0.505 / 0.674 | 0.551 / 0.665 | 0.558 / 0.721 |
| Qwen3-30B-A3B | 50% | 0.397 / 0.451 | 0.364 / 0.542 | 0.541 / 0.503 | |
| GLM-4.5-Air | 25% | 0.528 / 0.596 | 0.550 / 0.704 | 0.585 / 0.674 | 0.580 / 0.747 |
| GLM-4.5-Air | 50% | 0.284 / 0.444 | 0.419 / 0.564 | 0.515 / 0.554 | |
| ERNIE-4.5-21B | 50% | 0.089 / 0.465 | 0.270 / 0.569 | 0.441 / 0.576 | 0.523 / 0.721 |
| Mixtral-8x7B | 50% | 0.063 / 0.568 | 0.114 / 0.667 | 0.165 / 0.673 | 0.296 / 0.736 |

— [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- Argumento do REAP: a fusão tem um erro irredutível porque perde o controle fino de roteamento (o roteador passa a enviar tokens diferentes para o mesmo expert fundido) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- MergeMoE (out/2025): funde as saídas (não os parâmetros) com matrizes de compressão ajustadas por mínimos quadrados; Qwen3-30B-A3B com 128→64 experts nas camadas 28–47 (30B→25B): ARC-C 63,48 (M-SMoE 61,77; cheio 67,49), HellaSwag 74,93 (cheio 76,38); não compara com REAP — [arXiv 2510.14436](https://arxiv.org/html/2510.14436v1)
- MoE-XBench: REAM (fusão) a 25% perde 6,05 pontos vs 5,31 do REAP — a perda "no eixo dos experts" não é específica do REAP — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- Router KD (mar/2026): após qualquer compressão sem retreino (REAP, CFES, MoBE, TD-MoE, HC-SMoE, M-SMoE), recalibrar só o roteador (0,04% dos parâmetros do Qwen3-30B, ~2 h, KL contra o modelo original em C4) melhora "a maioria dos benchmarks" no Qwen3-30B (128→80 experts); ganhos marginais no Mixtral; não recupera colapsos — [arXiv 2603.02217](https://arxiv.org/html/2603.02217)

### Inferences
- Fusão não é candidata para Kimi/Qwen3: os únicos números em grão fino mostram perdas maiores que a poda ao mesmo tamanho, e a fusão exige requantizar (escalas de bloco compartilhadas), o que a poda não exige.
- A recalibração do roteador é um complemento barato a qualquer poda e alinha com o plano do projeto de ajustar só o roteador; falta a tabela numérica (apêndice cortado).

### Gaps
- Nenhum número de fusão em modelos ≥235B.
- Tabelas numéricas do Router KD não foram lidas (apêndice G além do limite do fetch).

## 4. Baixo posto / fatoração (MoBE, D²-MoE, MoE-SVD, MoLAE)

### Takeaway
MoBE (ICLR 2026) é o único método de fatoração medido em Qwen3-235B, DeepSeek-V3 e Kimi-K2: reduz 24–30% dos parâmetros totais com −0,6 a −1,3 pontos na média de 15 benchmarks (Kimi-K2 82,4→81,1; Qwen3-235B 81,5→80,9; DeepSeek-V3 79,3→78,0), contra −2,8 no Qwen3-30B. Mas o modelo fatorado não reduz (e pode aumentar) os parâmetros ativos, precisa de bases compartilhadas de centenas de GB no Kimi, e a reconstrução por expert passa por uma não linearidade, o que inviabiliza o streaming. D²-MoE e MoE-SVD só têm números em ≤57B e perdem muito em Qwen3.

### Cited Findings
- MoBE: cada matriz up/gate vira W^i = A^i · f(Σ_j α^{i,j} B^j), com A^i específico do expert (posto r = p, a dimensão intermediária), B^j bases compartilhadas por camada, f = SiLU ou tanh; down_proj não é fatorado ("guarda conhecimento crítico"); ajuste por Adam (lr 0,07) em H100 minimizando o erro de reconstrução, sem dados de calibração — [arXiv 2508.05257](https://arxiv.org/html/2508.05257v1)
- Número de bases m: 32 para Qwen3-30B e Qwen3-235B; 64 para DeepSeek-V3; 128 para Kimi-K2 (384 experts divididos em dois grupos de 64 bases) — [arXiv 2508.05257](https://arxiv.org/html/2508.05257v1)
- Resultados (média de 15 benchmarks; "ratio" = % dos parâmetros totais comprimidos) — [arXiv 2508.05257](https://arxiv.org/html/2508.05257v1):

| Modelo | Método | Ratio | Média |
|---|---|---|---|
| Qwen3-30B-A3B-2507 | original | 0% | 78.6 |
| | D²-MoE | 24% | 66.4 |
| | MoLAE | 24% | 67.5 |
| | MoBE | 24% | 75.8 |
| DeepSeek-V3-0324 | original | 0% | 79.3 |
| | MoLAE | 30% | 73.1 |
| | MoBE | 30% | 78.0 |
| Qwen3-235B-A22B-2507 | original | 0% | 81.5 |
| | MoLAE | 24% | 73.2 |
| | MoBE | 24% | 80.9 |
| | MoBE† | 24% | 79.7 |
| Kimi-K2-Instruct | original | 0% | 82.4 |
| | MoLAE | 24% | 76.6 |
| | MoBE | 24% | 81.1 |
| | MoBE† | 24% | 80.8 |

- MoBE aumenta os parâmetros ativados (o termo A^i soma ao custo); MoBE† compensa reduzindo os experts ativos de 8→6 (ou 6→4 nos pequenos). O paper não afirma reduzir compute por token, diz que exige múltiplas chamadas ao kernel fused-MoE ("ineficiente") e deixa um mega-kernel como trabalho futuro; não menciona quantização — [arXiv 2508.05257](https://arxiv.org/html/2508.05257v1)
- D²-MoE foi excluído dos modelos grandes por custo computacional em 8×H100 — [alphaxiv 2508.05257](https://www.alphaxiv.org/abs/2508.05257.md)
- D²-MoE (ICML 2025): W_i = W_b + ΔW_i, base por fusão ponderada por Fisher, SVD nos deltas, poda semi-dinâmica da base; afirma reduzir os parâmetros ativos de k·m para ≈(k·p% + s%)·m. Similaridade CKA entre experts 0,3–0,5 (Mixtral, Phi-3.5, DeepSeekMoE); as matrizes V dos deltas no Mixtral têm CKA ≈0,9 (motiva compartilhar V). Resultados (PPL WikiText-2 / acc média 0-shot) — [arXiv 2502.17298](https://arxiv.org/html/2502.17298v1):

| Modelo | Ratio | NAEE | D²-MoE | Original |
|---|---|---|---|---|
| Mixtral-8x7B | 20% | 4.77 / 0.58 | 4.65 / 0.60 | 3.98 / 0.63 |
| Mixtral-8x7B | 40% | 6.44 / 0.48 | 5.28 / 0.57 | |
| Mixtral-8x7B | 60% | 11.43 / 0.36 | 6.46 / 0.52 | |
| DeepSeekMoE-16B | 40% | 8.55 / 0.46 | 7.93 / 0.49 | 6.38 / 0.56 |
| Qwen2-57B-A14B | 40% | 6.81 / 0.55 | 8.19 / 0.58 | 5.12 / 0.59 |

- MoE-SVD (ICML 2025): SVD por expert, seleção de camadas por sensibilidade, uma matriz V compartilhada entre experts e poda top-k das U; "60% de compressão e 1,5× de velocidade com perda mínima" em Mixtral, Phi-3.5, DeepSeek e Qwen2 (só resumo) — [ICML 2025](https://icml.cc/virtual/2025/poster/44786); [PMLR](https://proceedings.mlr.press/v267/li25az.html)
- No benchmark do REAP, as variantes de decomposição não entram; no Router KD, MoBE e TD-MoE ("expert editing") são avaliados no Qwen3-30B e também se beneficiam da recalibração do roteador — [arXiv 2603.02217](https://arxiv.org/html/2603.02217)

### Inferences
- Os modelos grandes perdem menos com MoBE (−0,6 a −1,3 vs −2,8 no 30B), e o paper atribui isso à queda da esparsidade e à escala, não a um "subespaço compartilhado" mais forte; ele não faz essa afirmação. O que MoBE compartilha não é um subespaço linear: cada expert mistura as bases com coeficientes α e aplica uma não linearidade f antes de multiplicar por A^i, por isso funciona onde a base linear (D²-MoE, o teste local do projeto) falha.
- Contas minhas com as dimensões públicas (Kimi-K2: d=7168, p=2048, 384 experts, 61 camadas MoE; Qwen3-235B: d=4096, p=1536, 128 experts, 94 camadas): no Kimi, as bases B^j somam 128 × 2 × (2048×7168) ≈ 3,8 B parâmetros por camada ≈ 229 B no total (≈115 GB em 4 bits); no Qwen3-235B, 32 × 2 × (1536×4096) ≈ 403 M por camada ≈ 38 B no total (≈19 GB em 4 bits). Os bytes por expert cairiam ~47% (Kimi) e ~42% (Qwen3-235B), mas a reconstrução f(Σ α B) tem de ler todas as bases da camada a cada token (os 8 experts escolhidos podem compartilhar a passagem): ~1,9 GB/camada no Kimi e ~200 MB/camada no Qwen3-235B em 4 bits, de RAM, por token. No Qwen3-235B isso é ~19 GB/token de banda de RAM; no Kimi é inviável. Para o motor de streaming, MoBE não serve sem repensar a arquitetura (ex.: manter f(ΣαB) materializado, o que anula a economia).
- D²-MoE reduz de verdade os bytes ativos (só os deltas de baixo posto por expert), mas suas perdas em Qwen3-30B (78,6→66,4 a 24%) confirmam o resultado local de que os experts do Qwen3 não compartilham base linear; não há motivo para esperar diferente no 235B sem evidência.

### Gaps
- Nenhum paper de fatoração publica KL ou perplexidade em Qwen3-235B/Kimi-K2 (só médias de benchmarks).
- MoBE não publica tamanho em GB, horas de GPU nem throughput; não há checkpoint MoBE público encontrado.
- Não encontrei um método "SVD ponderado por Fisher/espectral" com números em ≥235B.

## 5. Nível de camada: remover ou condensar camadas MoE

### Takeaway
Condensar uma camada MoE em densa (expert compartilhado + poucos experts com gate fixo, CD-MoE) vence remover a camada: no DeepSeekMoE-16B, 9 de 27 camadas condensadas dão −27,5% de memória com 90% da acurácia sem treino e 98% com SFT leve. Só medido em 16B e Qwen1.5-MoE-A2.7B. Em escala de pré-treino, SlimQwen (Qwen3-Next-80B→23B) mostra que remover os últimos 25% das camadas é a melhor poda de profundidade, mas exige 120–400 B tokens de destilação.

### Cited Findings
- CD-MoE (TMLR 2026): condensa camadas selecionadas por divergência de Jensen-Shannon (greedy, 100 amostras de C4); a camada perde o roteamento e vira densa: sempre o expert compartilhado (CD-MoE-S) ou mais 6 experts roteados com gate fixo médio (CD-MoE-SR). DeepSeekMoE-16B (média de 8 tarefas, original 61,6) — [arXiv 2412.00069](https://arxiv.org/html/2412.00069v2):

| Método | Camadas | Memória | Speedup | Acc | % |
|---|---|---|---|---|---|
| LayerTrim (sem SFT) | 8 | 72,2% | 1,34× | 49,4 | 80% |
| BlockTrim (sem SFT) | 8 | 71,3% | 1,42× | 52,2 | 85% |
| CD-MoE-S (sem SFT) | 8 | 73,1% | 1,34× | 55,1 | 89% |
| CD-MoE-SR (sem SFT) | 9 | 72,5% | 1,26× | 55,5 | 90% |
| CD-MoE-S + SFT | 8 | 73,1% | 1,34× | 59,1 | 96% |
| CD-MoE-SR + SFT | 9 | 72,5% | 1,26× | 60,5 | 98% |

- Qwen1.5-MoE-A2.7B (original 65,5): CD-MoE-S em 6 camadas → 62,6 (96%) com 78,3% da memória, vs LayerTrim 53,6; o SFT atualiza só as camadas condensadas (~3,8% dos parâmetros) — [arXiv 2412.00069](https://arxiv.org/html/2412.00069v2)
- SlimQwen (mai/2026, equipe Qwen): Qwen3-Next-80B-A3B (48 blocos, 512 experts, top-10+1) → 23B-A2B: remove os últimos 25% das camadas, hidden 2048→1536, funde 512→256 experts, top-10→top-8. Com 120 B tokens de KD: média 73,45 vs 82,68 do professor (86,5%) e 61,66 de inicialização aleatória; com 400 B tokens, MMLU 77,39. Poda de profundidade: remover as últimas camadas supera a seleção por similaridade de ativação (MMLU one-shot 73,86 vs 41,95, de um professor de 75,62, num 15B-A3B de 24 camadas com 4 camadas removidas) — [arXiv 2605.08738](https://arxiv.org/html/2605.08738)
- Memória do SlimQwen: pico 156,56→43,30 GB; throughput vLLM 142,58→210,87 tok/s — [arXiv 2605.08738](https://arxiv.org/html/2605.08738)

### Inferences
- Uma camada condensada lê zero experts do SSD; se o Kimi (que tem 1 expert compartilhado, como DeepSeekMoE) aceitasse condensar ~1/3 das camadas como o 16B, os bytes lidos por token cairiam ~1/3, o que é mais do que qualquer poda de experts dá ao motor de streaming. Mas o único dado é em 16B; o Qwen3-235B não tem expert compartilhado, então só a variante SR (6 experts fixos) se aplica.
- A regra "remover as últimas camadas" do SlimQwen só vale com retreino de centenas de bilhões de tokens; sem treino, o próprio SlimQwen mostra GSM8K caindo a 2,05.

### Gaps
- Nenhum resultado de remoção/condensação de camadas em modelos ≥100B sem retreino.
- Não encontrei "expert layer skipping" dinâmico medido em modelos grandes.

## 6. Composição com quantização

### Takeaway
Quantizar primeiro e podar depois é a ordem que a literatura apoia: no Qwen3-30B, 4-bit com 128 experts (Eval+ 80,5, 25% do tamanho) supera bf16 com 64 experts (78,0, 50% do tamanho); 4-bit + 50% de poda dá 77,6 a 12,5% do tamanho; 2-bit + 50% de poda colapsa (28,6). O MoE-XBench confirma que a poda domina a perda (Score_robust ~88 com ou sem quantização) e que a quantização quase não soma perda (PPL 15,81→16,21 a 50% de poda).

### Cited Findings
- REAP, Tabela A5 (Qwen3-30B-A3B, evol-codealpaca, LLM Compressor) — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2):

| Quantização | Experts | Tamanho relativo | Eval+ |
|---|---|---|---|
| W16A16 | 128 | 100% | 81,4 |
| W16A16 | 64 | 50% | 78,0 |
| W4A16-G128 | 128 | 25% | 80,5 |
| W4A16-G128 | 64 | 12,5% | 77,6 |
| W2A16-G128 | 64 | 12,5% | 28,6 |

- Kimi-K2 e Qwen3-Coder-480B foram calibrados, podados e avaliados já quantizados (W4A16 e FP8); a poda não exige reconciliar escalas de bloco nem requantizar, a fusão exige; Kimi-K2 4-bit + 50% = "redução total de 87,5%", que só por quantização pediria 2 bits — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- MoE-XBench (Qwen3-30B-A3B, 7 workloads) — [arXiv 2608.21693](https://arxiv.org/html/2608.21693):

| Config | Tamanho | Média | Score_perf | Score_robust |
|---|---|---|---|---|
| BF16 | 61,10 GB | 0,8143 | 100,00 | 100,00 |
| Só poda (28,0%) | 44,00 GB | 0,7710 | 96,96 | 88,87 |
| Só Q4_K_M (69,5%) | 18,63 GB | 0,8127 | 99,91 | 97,00 |
| Poda + Q4_K_M (78,0%) | 13,42 GB | 0,7615 | 96,35 | 88,91 |

- PPL WikiText-2 por poda × bits (Qwen3-30B): 0% 6,82/6,818; 20% 8,67/8,80; 30% 10,04/10,23; 50% 15,81/16,21 (bf16 / Q4_K_M) — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- Qwen3.6-35B-A3B: só poda 25% → 0,8196; só quant → 0,8263; poda+quant (76,9%, 16,04 GB) → 0,8191 (base 0,8262) — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- "Composição não aditiva": ao mesmo alvo de memória, IQ1_M sobe a PPL muito mais que Q4_K_M + poda; Qwen3-235B: +54% de PPL com poda+quant — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- Um estudo com Qwen3.6-35B a 30% de poda achou que a perda em GGUF Q4_K_M vem da remoção de experts, não da conversão BF16→GGUF; um score "consciente de quantização" não superou o REAP padrão — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- Fisher-MoE 50% + AWQ 4-bit: VRAM ~6,5 GiB no Qwen1.5-MoE — [arXiv 2606.05538](https://arxiv.org/html/2606.05538)
- Kimi-K2.5-REAP50 em INT4 g32: 288 GB, OpenLLM 98,0% do denso — [HF](https://huggingface.co/kwanhee/Kimi-K2.5-REAP50-INT4-W4A16)

### Inferences
- Para o pack do projeto (q4 + codebook de 2,5 bits), a ordem compatível com os dados é: quantizar (ou partir do W4A16 oficial) → REAP 25% calibrado numa mistura geral → opcionalmente codebook nos picks de menor peso. O 2-bit uniforme em tudo + poda não tem evidência favorável (28,6 no REAP; IQ1_M no XBench).
- Ganho total esperado no Kimi em 4 bits com 25% de poda: 578 GB → ~440 GB (os experts são quase tudo); com 50%: ~300 GB, com a perda de MC/tool-use da seção 1.

### Gaps
- Nenhum estudo mede KL contra o modelo cheio após poda (a métrica usada no projeto); só PPL e benchmarks.
- Não há composição medida de poda + codebook/VQ (QTIP, codebook 2,5 bits).

## 7. O que reduz os bytes ativos por token (o que importa no streaming do SSD)

### Takeaway
Nenhum dos métodos com números em ≥235B reduz os bytes por token: REAP mantém top-8 e os parâmetros ativos (os checkpoints seguem A35B/A32B; sem speedup no H100, 0,94–1,03×); MoBE aumenta os ativos e precisa de bases enormes residentes. O que reduz bytes por token são métodos sem números em modelos grandes: D²-MoE (deltas de baixo posto), Fisher-MoE (−15,4% ativos), CD-MoE (camadas sem experts roteados) e os truques já usados no projeto (skip do pick 8, bias de cache).

### Cited Findings
- REAP: a poda só muda o número total de experts; parâmetros ativos aparecem só nos nomes dos checkpoints (35B) e o blog não discute velocidade — [Cerebras blog](https://www.cerebras.ai/blog/reap); [arXiv 2510.13999](https://arxiv.org/html/2510.13999v2)
- MoE-XBench: só-poda tem speedup 0,94–1,03× no H100; poda+Q4_K_M dá 1,37× prefill e 1,30× decode vs BF16 (o ganho vem da quantização) — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- MoBE: "pode aumentar os parâmetros ativados"; MoBE† corta 8→6 experts ativos para compensar — [arXiv 2508.05257](https://arxiv.org/html/2508.05257v1)
- D²-MoE: parâmetros ativos por token caem de k·m para ≈(k·p% + s%)·m — [arXiv 2502.17298](https://arxiv.org/html/2502.17298v1)
- Fisher-MoE: −15,4% de parâmetros ativos, 1,21× throughput (Qwen1.5-MoE) — [arXiv 2606.05538](https://arxiv.org/html/2606.05538)
- CD-MoE: camadas condensadas não roteiam; speedup 1,19–1,34× — [arXiv 2412.00069](https://arxiv.org/html/2412.00069v2)
- NAEE: skip dinâmico do 2.º expert deixa ~1,5 experts ativos em média no Mixtral — [arXiv 2407.00945](https://arxiv.org/pdf/2407.00945)

### Inferences
- A poda ajuda o motor de forma indireta: com 288 experts em vez de 384 por camada e o mesmo cache em GB, a fração do tráfego de roteamento coberta pelo cache sobe (menos experts para cobrir, e os removidos são os de menor gate×ativação, que já eram pouco acertados). Isso só pode ser medido com o trace real do modelo podado; a literatura não traz esse número.
- Como a poda é ortogonal ao bias de cache e às duas resoluções, o ganho esperado no motor é o de reduzir o universo (hit rate) e o tamanho do pack, não o de bytes por miss.
- Se o objetivo é bytes por token, a linha mais promissora e ainda sem medição em escala é condensar camadas (CD-MoE), porque elimina o I/O de camadas inteiras, seguida de poda de dimensões intermediárias (Fisher-MoE), que encolhe cada expert mantendo a estrutura do pack.

### Gaps
- Nenhuma medição de hit rate de cache ou de bytes lidos por token num modelo podado.
- Nenhum número de D²-MoE, Fisher-MoE ou CD-MoE acima de 57B.
