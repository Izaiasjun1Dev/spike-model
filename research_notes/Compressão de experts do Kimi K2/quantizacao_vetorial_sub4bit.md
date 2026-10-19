# Quantização sub-4-bit de pesos (vetorial, codebook, trellis) para LLMs grandes — estado da arte em outubro de 2026

Notas de pesquisa para o projeto de streaming de experts do SSD (Kimi-K2-Instruct 1T, Qwen3-235B-A22B). O foco é o que a literatura alcança em 2–3 bits/peso em modelos de 70B+ e o que isso custa. Convenções: "bpw" = bits por peso incluindo escalas/índices quando a fonte os inclui; "W2" = perplexidade no WikiText-2; "×1,xx" = razão da perplexidade quantizada sobre a de fp16/bf16 (calculada por mim a partir dos números citados). Quando o número vem de um modelo de 7–13B, está marcado.

Aviso de leitura: os papers usam contextos diferentes para a perplexidade (QuIP#/QTIP/PV-Tuning: 4096 para Llama 2 e 8192 para Llama 3; GPTVQ/VPTQ: 2048 para Llama 2), então só compare linhas dentro da mesma tabela ou com o mesmo fp16 de referência.

---

## Pergunta 1 — O que QuIP#, AQLM, PV-Tuning, QTIP, GPTVQ, VPTQ, HIGGS e os sucessores de 2025–2026 reportam em 2, 2,5 e 3 bits em Llama-2/3 70B ou maiores?

### Takeaway
Em modelos densos de 70B, a fronteira em 2 bits é o trellis (QTIP) com ajuste fino, ou PV-Tuning sobre AQLM: Llama-2-70B fica em W2 3,70–3,78 contra 3,12 (×1,19–1,21) e Llama-3-70B em 4,57–4,97 contra 2,59 (×1,76–1,92). Em 3 bits o custo cai para ×1,05 (Llama-2-70B) a ×1,2 (Llama-3-70B); em 4 bits é ≈×1,01–1,06. Nenhum método reporta 2,5 bits em 70B diretamente; os pontos mais próximos são GSQ não-uniforme (2,62 bpp, escalar) e SignRoundV2 (2,5 bits misto W2/W4), ambos com perda de ~1,5–3 pontos de acurácia zero-shot.

### Cited Findings

**QTIP (trellis, Cornell, NeurIPS 2024)** — [arXiv 2406.11235](https://arxiv.org/html/2406.11235), [repo](https://github.com/Cornell-RelaxML/qtip)

Llama 2 (ctx 4096), HYB com ajuste fino (Tabela 5 do paper):

| Método | bits | W2 2-7B | W2 2-13B | W2 2-70B | C4 2-70B |
|---|---|---|---|---|---|
| FP16 | 16 | 5,12 | 4,57 | 3,12 | 4,97 |
| AQLM/GPTVQ ("X") | 4 | 5,21 | 4,65 | 3,19 | 5,03 |
| QuIP# | 4 | 5,19 | 4,63 | 3,18 | 5,02 |
| QTIP | 4 | 5,17 | 4,61 | 3,16 | 5,00 |
| X | 3 | 5,38 | 4,78 | 3,36 | 5,17 |
| QuIP# | 3 | 5,41 | 4,78 | 3,35 | 5,15 |
| QTIP | 3 | 5,28 | 4,69 | **3,26** (×1,045) | 5,08 |
| X | 2 | 6,14 | 5,33 | 3,83 | 5,62 |
| QuIP# | 2 | 6,19 | 5,35 | 3,91 | 5,71 |
| QTIP | 2 | 5,86 | 5,11 | **3,70** (×1,19) | 5,48 |

Zero-shot Llama-2-70B QTIP (ArcC/ArcE/PiQA/Wino): FP16 51,1/77,7/81,1/77,0; 4 bits 50,0/77,8/81,3/76,9; 3 bits 50,3/78,2/80,6/77,0; 2 bits 48,0/76,3/80,2/75,1. — [QTIP Tabela 6](https://arxiv.org/html/2406.11235)

Llama 3 70B (ctx 8192, Tabela 7): BF16 W2 2,59 / C4 5,78; QTIP 4 bits 2,75 / 5,83 (×1,06); 3 bits 3,18 / 5,98 (×1,23); 2 bits **4,97 / 6,80 (×1,92)**. QuIP# nos mesmos bits: 2,99; 3,59; 5,77. — [QTIP Tabela 7](https://arxiv.org/html/2406.11235)

Llama 3.1 Instruct (Tabela 8, só W2; a atribuição de linhas a tamanhos é inferência minha a partir da ordem e é consistente com o BF16 3,52 do 70B-Instruct reportado por outro paper, ver abaixo):

| Modelo | Referência | QTIP 4 bits | QTIP 3 bits | QTIP 2 bits | PV-Tuning ~2 bits |
|---|---|---|---|---|---|
| 3.1 405B Inst. | 1,70 (FP8 da Meta) | 1,79 | 2,05 (×1,21) | 3,29 (×1,94) | — |
| 3.1 70B Inst. | 3,52 (BF16) | 3,73 | 4,12 (×1,17) | 5,08 (×1,44) | 5,70 (2,01 bits) |
| 3.1 8B Inst. | 6,50 (BF16) | 6,61 | 6,80 | 7,82 | 8,45 (2,07 bits) |

— [QTIP Tabela 8](https://arxiv.org/html/2406.11235); o BF16 3,52 do Llama-3.1-70B-Instruct aparece também em [XOR-Trellis](https://arxiv.org/html/2610.00432).

Parâmetros: 1MAD/3INST com L=16, V=1, k=2/3/4; HYB com L=16, V=2, Q=9 (codebook de 2^9×2 = 2 KiB). O paper reporta bits nominais (2/3/4) e não detalha o overhead de escalas. — [QTIP](https://arxiv.org/html/2406.11235)

**QuIP# (E8P, Cornell, ICML 2024)** — [arXiv 2402.04396](https://arxiv.org/html/2402.04396), [repo](https://github.com/Cornell-RelaxML/quip-sharp)

Llama 2 70B (ctx 4096): FP16 3,12/4,97; 4 bits 3,18/5,02; 3 bits 3,35/5,15; 2 bits 3,91/5,71 (×1,25). Sem ajuste fino: 4 bits 3,18; 3 bits 3,41; 2 bits 4,16 (×1,33). Ablação sem E8 (codebook escalar/lattice simples) em 2 bits: 4,58. Zero-shot 2-70B em 2 bits: 48,7/77,3/80,3/75,9 contra 51,1/77,7/81,1/77,0. — [QuIP#](https://arxiv.org/html/2402.04396)

E8P: codebook de dimensão 8 com 2^16 entradas, armazenado como tabela-fonte de 2^8 entradas (1 KiB); 16 bits por 8 pesos = 2 bpw; os vetores de sinal somam < 0,01 bpw. 3 bits = E8P 2 bits + codebook E8 de 1 bit (RVQ); 4 bits = duas rodadas de E8P. — [QuIP#](https://arxiv.org/html/2402.04396)

**AQLM e PV-Tuning (ISTA-DASLab)** — [repo AQLM](https://github.com/Vahe1994/AQLM), [PV-Tuning arXiv 2405.14852](https://arxiv.org/html/2405.14852)

| Modelo | Método | bpw | W2 | C4 | zero-shot médio |
|---|---|---|---|---|---|
| Llama 2 70B | FP16 | 16 | 3,12 | 4,97 | 72,40 |
| | AQLM | 2,07 | 3,94 | 5,72 | 68,75 |
| | QuIP# | 2,01 | 3,91 | 5,71 | 68,60 |
| | PV-Tuning | 2,07 | **3,78** (×1,21) | 5,56 | 70,72 |
| | PV-Tuning | 1,14 | 5,52 | 7,50 | 64,58 |
| | PV-Tuning | 1,01 | 6,09 | 8,20 | 61,55 |
| Llama 3 70B | FP16 | 16 | 2,59 | 5,78 | 75,37 |
| | PV-Tuning | 2,07 | **4,57** (×1,76) | 6,56 | 70,38 |
| | PV-Tuning | 1,14 | 7,76 | 8,93 | 58,01 |

— [PV-Tuning Tabela 2](https://arxiv.org/html/2405.14852)

AQLM Llama-3-70B 1x16 (sem PV): MMLU 5-shot 0,79→0,75, 21,9 GB; Llama-2-70B 1x16 W2 3,83 (18,8 GB), 2x8 W2 4,21 (18,2 GB); PV-tuned Llama-3-70B 1x16g16 (~1,5 bpw?) W2 8,67, 13 GB. — [README AQLM](https://github.com/Vahe1994/AQLM)

Quanto o PV-Tuning recupera (Llama-2-7B, W2): AQLM 2,01 bpw só com ajuste contínuo 6,69 → PV+STE 5,90; GPTQ 2,14 bpw 16,77 → 8,43; VQ 1,58 bpw 8,17 → 7,32. — [PV-Tuning Tabela 1](https://arxiv.org/html/2405.14852)

**GPTVQ (Qualcomm)** — [arXiv 2402.15319](https://arxiv.org/html/2402.15319). Ctx 2048, Llama-2-70B FP16 W2 3,31:

| bpw | 1D | 2D | 4D |
|---|---|---|---|
| 2,125 | 5,03 | 4,72 | **4,44** (×1,34) |
| 2,25 | 4,85 | 4,58 | **4,36** (×1,32) |
| 3,125 | 3,64 | **3,58** (×1,08) | — |
| 4,125 | 3,39 | 3,38 | — |

Overhead de codebook incluído: 0,125 bpv (0,25 em b=2). Codebook de 8 bits por entrada; grupos de 256–65 536 pesos por codebook. — [GPTVQ](https://arxiv.org/html/2402.15319)

**VPTQ (Microsoft)** — [arXiv 2409.17066](https://arxiv.org/html/2409.17066), [repo](https://github.com/microsoft/VPTQ)

Llama-2-70B (ctx 2048, W2/C4/AvgQA): FP16 3,12/4,97/70,2; VPTQ 2,07 bits 3,93/5,72/68,6; 2,11 bits 3,92/5,71/68,7; AQLM 2,07 3,94/5,72/68,8; QuIP# 2,0 3,91/5,71/69,0; GPTVQ 2,25 4,25/6,9/68,5. Llama-3-70B: FP16 2,9/75,3; VPTQ 4,05 bits 3,15/74,7; 3,01 bits **3,81/73,7** (×1,31); 2,02 bits **5,6/70,9** (×1,93); 2,07 bits 5,66/70,7. Mistral-7B 2,04 bits 5,64 contra 4,77. — [VPTQ](https://arxiv.org/html/2409.17066)

Configuração 2 bits do Llama-2-70B (2,07 bpw): 1% de outliers com v=4, k=8192; o resto v=12, k1=4096, k2=4096 (residual), 4 grupos. v=8/k=65 536 aparece só em ablação (2,19 bpw no 13B) e é a configuração dos checkpoints públicos (nomes `v8-k65536-256`). Bits/peso = log2(k1)/v + log2(k2)/v + overhead de codebook (v×(k1+k2)×16 bits por camada). — [VPTQ Apêndice C.2](https://arxiv.org/html/2409.17066)

**HIGGS (Hadamard + grades MSE-ótimas, data-free; Yandex/ISTA)** — [arXiv 2411.17525](https://arxiv.org/html/2411.17525). Llama-3.1-70B base: FP16 W2 2,541; 3,25 bits HIGGS dinâmico **3,466** (×1,36), HQQ 4,057, NF 41,76, AF 102 350; 4,02 bits HIGGS dinâmico 2,827 (×1,11). Llama-3.1-8B (3,25 bits): FP16 5,607; HIGGS p=2 7,110, p=4 6,643, dinâmico 6,388, GPTQ 7,133, HQQ 7,317, NF 7,683. Grades de dimensão p ∈ {1,2,3,4}, total de pontos 2^(k·p) ≤ ~2^10 para caber na shared memory. — [HIGGS](https://arxiv.org/html/2411.17525)

**Sucessores 2025–2026**

- **XOR-Trellis (2026)**, trellis sem Hadamard, ~2,25 bpw efetivos (uma escala E5M3 por grupo de 32): Llama-3.1-70B-Instruct BF16 3,52/6,46/80,63 (W2/C4/acc); QTIP-3INST 6,31/8,02/77,96; XOR-Trellis 5,62/7,58/78,43. Llama-3.1-8B: BF16 5,84; QTIP-3INST 8,48; XOR-Trellis 8,00. Calibração 384 sequências RedPajama; sem kernel de decode medido (dá estimativas de hardware). — [arXiv 2610.00432](https://arxiv.org/html/2610.00432)
- **CurveTQ (2026)**, trellis sem rotação com busca ponderada pela curvatura: 1–3 pontos de acurácia acima de QTIP e Proteus em 2 bits em três modelos Instruct de 4–8B e "primeiro resultado de trellis num MoE de 35B" (nome do modelo e números não estão na página de resumo; sem HTML disponível). — [arXiv 2610.09212](https://arxiv.org/abs/2610.09212)
- **GSQ (Gumbel-Softmax, escalar, ISTA 2026)** — Llama-3.1-70B-Instruct zero-shot médio: FP16 78,99; QTIP 2,00 bits 77,25; PV-Tuning 2,07 76,27; GSQ 2,13 bpp 75,57; GSQ não-uniforme 2,37 bpp 77,40 e **2,62 bpp 77,50**; em 3 bits GSQ 3,13 77,99, QTIP 3,00 78,17; GPTQ 2,25 bits 57,38, EfficientQAT 2,25 71,43. — [arXiv 2604.18556](https://arxiv.org/html/2604.18556v2), [repo](https://github.com/IST-DASLab/GSQ)
- **SignRoundV2 (Intel, 2025)** — acurácia média (5 tarefas), Llama3-70B: FP16 75,28; 2 bits G128 SignRoundV2* 70,16 (G64 70,93); AQLM 70,10; EfficientQAT 67,57; **2,5 bits (misto W2/W4 G128) 72,37–72,68**. Llama2-70B 2 bits: FP16 72,41; QuIP# 70,91; AQLM 70,84; SignRoundV2* 68,82. Qwen3-32B 2,5 bits: 74,96 contra 78,05 (96% de recuperação). — [arXiv 2512.04746](https://arxiv.org/html/2512.04746)
- **KronQ (USC/Yale, COLM 2026)** — Llama-3-70B 2 bits W2 7,93 (GPTQ/GPTAQ divergem, > 2000); comentário de usuário no HF cita 4,48 em W3 e 8,43 em W2. Usa covariância de gradientes e rotação bidirecional. — [arXiv 2607.07964](https://arxiv.org/abs/2607.07964)
- **YAQA (Cornell 2025)** — arredondamento adaptativo com Hessianas Kronecker sobre a KL do modelo inteiro; ~30% menos KL que GPTQ/LDLQ; revisão secundária cita KL 0,335 em Llama-3.1-70B-Instruct a 2 bits contra 0,497 (LDLQ). — [arXiv 2505.22988](https://arxiv.org/abs/2505.22988v1)
- **Q-Palette (2025)** — família de quantizadores fracionários: TCQ 1,5–5,0 em passos de 0,5, VQ 1,5–6,0, Half-TCQ 1,75/2,25/2,75…; em Llama-3.1-8B a 2,5 bits o gap de distorção é VQ (d=2,3,4) 0,128 contra TCQ 0,031 (ideal gaussiano 0); a 3,25 bits W2 6,48 (TCQ) contra HIGGS 6,64 e FP16 5,61; QTIP data-free 3 bits 6,81. — [arXiv 2509.20214](https://arxiv.org/html/2509.20214)
- **Bielik-Q2-Sharp (2026, 11B)** — comparação prática: QuIP# E8P12 ~2,4 bpw 3,26 GB, QTIP ~2,4 bpw 3,27 GB, AQLM ~2,4 bpw 3,62 GB, VPTQ ~3,58 bpw 5,0 GB; acurácia MC 78,1–79,4 em todos ("teto independente do método"); QuIP# ≈ GGUF IQ2_XXS (71,92 contra 72,07 em 22 tarefas); os métodos com rotação aprendida (SpinQuant, Butterfly) passaram no MC e falharam na geração. — [arXiv 2603.04162](https://arxiv.org/html/2603.04162)

### Inferences
- Em 70B denso, "quase sem perda" (≤ ×1,05 de W2) só acontece em 3 bits no Llama-2-70B (QTIP ×1,045, GPTVQ 2D 3,125 bpw ×1,08) e em 4 bits para qualquer família. Llama 3/3.1 e Qwen3 são bem mais sensíveis: a 3 bits a melhor marca é ×1,17–1,23 e a 2 bits ×1,44–1,9.
- Os métodos convergem: a 2 bits em Llama-2-70B, QuIP#, AQLM, VPTQ e GPTVQ-4D ficam entre 3,91 e 4,44; só ajuste fino de verdade (QTIP+FT 3,70, PV-Tuning 3,78) separa do pelotão. O ganho de trellis sobre E8P sem FT é pequeno em 70B (Tabela 3 do QTIP: 3,87–3,90 contra 3,91).
- Interpolando QTIP em Llama-2-70B (2 bits ×1,19; 3 bits ×1,045), um trellis de ~2,5 bits ficaria perto de ×1,08–1,10 num 70B denso. É uma extrapolação minha, não um número reportado.

### Gaps
- Nenhum paper reporta 2,5 bits em modelos ≥ 70B com codebook/trellis; só escalar (GSQ 2,62, SignRoundV2 2,5 misto) e o Q-Palette em 8B.
- A Tabela 8 do QTIP (405B) foi lida de HTML com rótulos deslocados; conferir no PDF antes de citar.
- Os apêndices de HIGGS para 405B e o conteúdo completo do CurveTQ não estavam acessíveis.

---

## Pergunta 2 — Quais desses métodos foram aplicados a MoE (DeepSeek-V3/R1, Kimi K2, Qwen3-235B, Mixtral) e com que números?

### Takeaway
Há poucos resultados de codebook/trellis em MoE grandes: CCQ (Baidu) em DeepSeek-V3-0324 (2,06–2,75 bpw: −1,3 a −2,3 pontos de média em GSM8K/C-Eval/MMLU, 184–230 GB) e ERNIE-4.5-300B-A47B; GSQ (escalar) nos experts do Kimi-K2.5 (4,5→2,13 bpp, ~260 GB, resultados mistos); VPTQ publicou checkpoints de DeepSeek-R1 (v8, k=65 536, sem ajuste fino) sem números; AQLM e GPTVQ têm números de Mixtral-8x7B; QTIP/QuIP#/PV-Tuning não reportam MoE.

### Cited Findings

**DeepSeek-V3-0324 com CCQ (Convolutional Code Quantization, Baidu, 2025)** — [arXiv 2507.07145](https://arxiv.org/html/2507.07145)

| Config | bpw | GSM8K | C-Eval | MMLU | Média | Memória |
|---|---|---|---|---|---|---|
| Baseline | 8 | 96,21 | 87,31 | 86,50 | 90,01 | 642 GB |
| GPTQ | 4 | 94,77 | 87,66 | 86,60 | 90,04 | 346 GB |
| AWQ | 4 | 96,13 | 87,22 | 86,38 | 89,43 | 328 GB |
| CCQ | 2,75 | 95,68 | 85,08 | 85,24 | 88,67 | 230 GB |
| CCQ | **2,5** | 95,60 | 84,11 | 85,07 | **88,26** | 212 GB |
| CCQ | 2,06 | 94,92 | 84,07 | 84,29 | 87,76 | 184 GB |

ERNIE-4.5-300B-A47B: WINT8 89,39 (281 GB); WINT4 89,14 (149 GB); WINT2/CCQ 87,36 (89 GB, −68% contra WINT8). CCQ é código convolucional decodificado por deslocamento de bits (sem codebook armazenado) com "Code Cluster" para valores de 8 bits; sem perplexidade, sem tokens de calibração nem tempo de quantização no paper. — [CCQ](https://arxiv.org/html/2507.07145)

**Kimi-K2.5 com GSQ (escalar, só experts não compartilhados)** — [arXiv 2604.18556](https://arxiv.org/html/2604.18556v2), [repo](https://github.com/IST-DASLab/GSQ): base 4,5 bpp → 2,13 bpp (~260 GB); calibração OpenThoughts, 4096 sequências × 4096 tokens, 10 épocas por bloco; hardware "8× H100/H200 (nó completo), 1–2 nós". Benchmarks (base → GSQ): AIME25 95,33 → 93,00; **GPQA Diamond 89,29 → 76,57**; LiveCodeBench 61,37 → 69,37; MATH500 96,68 → 97,32; MRCR 32–64k 57,10 → 53,73. O paper afirma que métodos vetoriais ficam "proibitivamente caros" em MoE porque treinam codebooks e fazem atualizações por bloco. — [GSQ](https://arxiv.org/html/2604.18556v2)

**Kimi-K2.6 em GGUF (Unsloth Dynamic, comunidade)**: Dynamic 2-bit 340–350 GB, perplexidade 2,4131; UD-Q4_K_XL 584 GB ppl 1,8420; UD-Q8_K_XL 595 GB ppl 1,8419 (o modelo é INT4 nativo por QAT, então Q4 ≈ lossless e 2 bits custa ×1,31). — [Unsloth docs](https://unsloth.ai/docs/models/kimi-k2.6); a família K2-Thinking é lançada já em INT4 com QAT — [HF moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)

**DeepSeek-V3 em Q2_K (llama.cpp, estudo de 2025)**: média ponderada 68,73 contra 75,79 em Q4_K_M. — [arXiv 2505.02390](https://arxiv.org/html/2505.02390v2). DeepSeek-V3 4 bits W2: HQQ 5,38, SINQ 5,31 — [SINQ arXiv 2509.22944](https://arxiv.org/pdf/2509.22944)

**DeepSeek-R1 em VPTQ**: coleção "vptq-deepseek-r1-without-finetune" com `deepseek-r1_v_8_k_65536`, `_256` (residual), `_mp4` e `_mixed_mp4`; o card do modelo exigiu autenticação e a coleção não mostra tamanho nem avaliação. O README do VPTQ afirma "só ~17 horas para quantizar o Llama-3.1 405B" sem dizer o hardware. — [coleção HF](https://huggingface.co/collections/VPTQ-community/vptq-deepseek-r1-without-finetune-67d0832c203afd208bb8449e), [README VPTQ](https://github.com/microsoft/VPTQ)

**Qwen3-235B-A22B (GGUF, comunidade ubergarm)**: BF16 438 GiB W2 5,3141; ubergarm mix-IQ3_K 107 GiB 5,4403 (×1,024, KLD 0,0146); unsloth UD-Q3_K_XL 97 GiB 5,5695 (×1,048, KLD 0,029); lmstudio Q3_K_L 104 GiB 5,6582 (×1,065). Essas misturas equivalem a ~3,5–3,9 bpw médios (107 GiB sobre 235B parâmetros ≈ 3,9 bpw; 97 GiB ≈ 3,5; cálculo meu). — [gist ubergarm](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)

**Qwen3-30B-A3B (proxy do 235B; ik_llama.cpp, receitas por tensor)** — bf16 W2 9,0656; atenção em IQ4_KS, experts em baixa precisão:

| Receita | experts | GiB | bpw médio (meu cálculo, 30,5B par.) | W2 | × bf16 |
|---|---|---|---|---|---|
| IQK-1 | IQ2_KS (down das 6 primeiras camadas em IQ4_KS) | 8,62 | 2,43 | 9,9517 | ×1,098 |
| IQK-2 | IQ2_K | 9,18 | 2,58 | 9,7154 | ×1,072 |
| IQK-3 | IQ2_K, down em IQ3_K | 10,23 | 2,88 | 9,3908 | ×1,036 |
| IQK-4 | + up/gate IQ3_K em 26 camadas | 11,45 | 3,23 | 9,1057 | ×1,004 |
| IQK-5 | IQ3_K | 12,62 | 3,55 | 9,0147 | ×0,994 |
| unsloth UD-IQ2_M | — | 10,1 | 2,84 | 10,3726 | ×1,14 (KLD 0,0666) |
| bartowski IQ2_M | — | 9,7 | 2,73 | 9,9788 | ×1,10 (KLD 0,0691) |

— [discussão ik_llama.cpp #359](https://github.com/ikawrakow/ik_llama.cpp/discussions/359), [gist ubergarm](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)

**Mixtral-8x7B**: AQLM 1x16 W2 3,35 (ctx 8k), 12,6 GB — [README AQLM](https://github.com/Vahe1994/AQLM); GPTVQ (ctx 2048, FP16 3,84): 2,125 bpw 1D 8,37 / 2D 5,92 / 4D **5,28** (×1,375), zero-shot 54,9/64,4/68,7 contra 75,9; 2,25 bpw 4D 5,16; 3,125 bpw 2D 4,27 (×1,11), zero-shot 75,1 — [GPTVQ](https://arxiv.org/html/2402.15319)

**Trabalhos específicos de MoE (escalares)**: EAC-MoE (ACL 2025) mostra que 2 bits desloca a seleção de experts e que um objetivo de MSE no roteador funciona mal com muitos experts (DeepSeek-MoE-16B, 6 de 64) — [ACL 2025](https://aclanthology.org/2025.acl-long.633.pdf); QuantMoE-Bench conclui que a frequência de uso do expert é boa heurística para alocar bits — [arXiv 2406.08155](https://arxiv.org/pdf/2406.08155); MoEQuant (amostragem balanceada por expert) tem 3/4 bits em DeepSeek-MoE-16B e Mixtral — [arXiv 2505.03804](https://arxiv.org/pdf/2505.03804); MiLo (MLSys 2025) soma compensadores de baixo posto aos experts quantizados — [MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/9032e5c9ec394ce768a2fa9bdc56af6c-Paper-Conference.pdf). Relato de comunidade: DeepSeek V4 Flash a ~2 bits quantizando só os experts roteados (IQ2_XXS/Q2_K) e mantendo o resto em Q8 — [antirez no X](https://x.com/antirez/status/2048885632869523869)

### Inferences
- O padrão que funciona em MoE grandes é o mesmo do projeto: experts roteados em 2–3 bits, atenção/shared/roteador em ≥ 4–8 bits. Com isso, DeepSeek-V3 a 2,5 bpw (CCQ) perde ~1,8 pontos de média e o Qwen3-30B com experts em IQ2_K (2,58 bpw médio) perde ×1,07 de W2.
- O ×1,14 do codebook próprio de 2,5 bits ("lo-all") fica na faixa do UD-IQ2_M (×1,14 a 2,84 bpw) e abaixo da receita IQ2_K com imatrix (×1,07 a 2,58 bpw). Cuidado: o ×1,14 do projeto é relativo ao q4 do MLX e o da tabela é relativo ao bf16, e o corpus de perplexidade é outro.
- Ninguém publicou um checkpoint de Kimi K2 ou Qwen3-235B em formato de codebook; em DeepSeek-R1 só o VPTQ, sem avaliação pública.

### Gaps
- Números de qualidade dos checkpoints VPTQ de DeepSeek-R1 (card bloqueado por 401).
- Nome e números do MoE de 35B no CurveTQ (sem HTML).
- Perplexidade de Qwen3-235B-A22B abaixo de ~3,5 bpw médios: não achei medição (o gist só tem misturas Q3).
- O GSQ não reporta tamanho total nem GPU-horas do Kimi-K2.5 (apêndice G.4 truncado).

---

## Pergunta 3 — Escolhas de projeto do codebook e o gap entre codebook de 2,5 bits e afim de 4 bits

### Takeaway
A literatura convergiu para (a) rotação de Hadamard/incoerência antes de qualquer codebook, (b) dimensão 8 com 2^16 entradas estruturadas (E8P) ou trellis com L=16 estados e codebook de 2^9×2 (QTIP HYB), e (c) ajuste fino dos parâmetros contínuos (codebooks, escalas, normas) e, no PV-Tuning, dos próprios índices. Trellis dá o menor gap de distorção em 2–2,5 bits (Q-Palette: 0,03 contra 0,13 de VQ d≤4); RVQ é como QuIP#/VPTQ chegam a 3 e 4 bits. O gap "2,5 bits codebook contra 4 bits afim" em 70B+ não é medido diretamente; os pontos mais próximos dão −1,8 pontos de média (CCQ 2,5 em DeepSeek-V3 contra GPTQ 4 bits) e ×1,07 de W2 (IQ2_K 2,58 bpw no Qwen3-30B contra ×1,005 de Q4_K_M).

### Cited Findings
- Dimensão e tamanho: QuIP# usa d=8, 2^16 entradas (tabela-fonte de 2^8, 1 KiB); 16 bits por 8 pesos. — [QuIP#](https://arxiv.org/html/2402.04396). AQLM 1x16 (um codebook de 2^16, grupo 8) é a configuração "melhor"; 2x8 (dois codebooks de 2^8) é "OK" e mais rápida. — [README AQLM](https://github.com/Vahe1994/AQLM). VPTQ: v=8/k=65 536 (+256 residual) nos checkpoints públicos; v=12 com k1=k2=4096 e 1% de outliers em v=4/k=8192 no paper. — [VPTQ](https://arxiv.org/html/2409.17066). GPTVQ: 1D/2D/4D com codebooks pequenos (8 bits por entrada, 0,125–0,25 bpw de overhead); 4D ganha de 2D que ganha de 1D em 2,125 bpw (4,44 < 4,72 < 5,03 no Llama-2-70B). — [GPTVQ](https://arxiv.org/html/2402.15319). HIGGS: grades de p ∈ {1,2,3,4}, ≤ ~2^10 pontos; p=2 é o padrão prático. — [HIGGS](https://arxiv.org/html/2411.17525)
- Trellis contra lookup: QTIP HYB (L=16, V=2, Q=9, 2 KiB) iguala a velocidade do QuIP# e melhora a qualidade; o lookup puro de 2^14 (32 KB) "é grande demais para o L1 das GPUs atuais". Em Llama-2-70B sem ajuste fino, 1MAD/3INST 2 bits dão 3,87/3,90 contra QuIP# 3,91 e AQLM 3,83 — a vantagem do trellis sem FT em 70B é pequena; com FT, 3,70 contra 3,91. — [QTIP](https://arxiv.org/html/2406.11235)
- Fracionário: Q-Palette mede a 2,5 bits (Llama-3.1-8B) gap de distorção VQ-2,3,4 0,128 / TCQ-2,3,4 0,031 / TCQ-ALL 0,026 contra 0 do quantizador gaussiano ideal; TCQ é o melhor em qualidade e VQ o mais rápido em decode. — [Q-Palette](https://arxiv.org/html/2509.20214)
- RVQ: QuIP# 3 bits = E8P 2 bits + E8 1 bit; 4 bits = 2× E8P. — [QuIP#](https://arxiv.org/html/2402.04396). VPTQ usa k2 residual em todas as configurações. — [VPTQ](https://arxiv.org/html/2409.17066)
- Ajuste fino depois de quantizar: QuIP# 2 bits Llama-2-70B 4,16 → 3,91 (W2) com FT de codebooks/escalas/normas (~50 GPU-h). — [QuIP#](https://arxiv.org/html/2402.04396). PV-Tuning (otimiza também os índices discretos) leva AQLM 7B de 6,69 (só contínuo) para 5,90 e 70B de 3,94 para 3,78. — [PV-Tuning](https://arxiv.org/html/2405.14852). VPTQ ajusta centróides e layer-norm por camada em 128 amostras C4 por 100 iterações. — [VPTQ](https://arxiv.org/html/2409.17066)
- Rotação: QuIP# ablação "sem E8" e sem incoerência a 2 bits sobe para 12,3 (7B) e 4,87 (70B, ctx 2048). — [QuIP#](https://arxiv.org/html/2402.04396). HIGGS/QTIP/QuIP#/Q-Palette todos usam Hadamard; XOR-Trellis e CurveTQ (2026) tiram a rotação e compensam com a busca ponderada pela curvatura (LDL/Hessiana). — [XOR-Trellis](https://arxiv.org/html/2610.00432), [CurveTQ](https://arxiv.org/abs/2610.09212)
- Gap 2,5 contra 4 bits em MoE grande: DeepSeek-V3-0324 CCQ 2,5 bpw 88,26 contra GPTQ 4 bits 90,04 e 8 bits 90,01 (212 GB contra 346 GB). — [CCQ](https://arxiv.org/html/2507.07145). SignRoundV2 2,5 bits (W2/W4 misto) Llama3-70B 72,37 contra 75,28 (96%); 2 bits 70,16. — [SignRoundV2](https://arxiv.org/html/2512.04746). GSQ 2,62 bpp Llama-3.1-70B-Instruct 77,50 contra 78,99 (−1,5 pontos). — [GSQ](https://arxiv.org/html/2604.18556v2)
- Qwen3-30B-A3B: experts IQ2_K (2,58 bpw) W2 ×1,072; Q4_K_M ×1,015; IQ4_KS ×0,99. — [#359](https://github.com/ikawrakow/ik_llama.cpp/discussions/359), [gist](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)
- Afim MLX como referência: `affine` armazena escala e bias no dtype de entrada por grupo de 32/64/128; 4 bits g64 = 4,5 bpw e 2 bits g64 = 2,5 bpw (cálculo meu, 2×16/64). — [docs mlx.core.quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)

### Inferences
- O codebook próprio (Hadamard + codebook pequeno de 2,5 bits, ×1,14) está no regime do GPTVQ 2D/4D e dos IQ2 do llama.cpp. O que os melhores métodos fazem a mais: (1) Hessiana por camada (LDLQ/GPTQ-style) no arredondamento, (2) d=8 estruturado ou trellis com 2^16 estados efetivos, (3) ajuste fino dos codebooks/escalas com destilação. O item 3 sozinho vale ~6% de W2 em 70B (QuIP#: 4,16 → 3,91).
- Gap plausível entre 2,5 bits bem feito e 4 bits afim num MoE grande: ~1,5–2 pontos de benchmark ou ×1,05–1,08 de perplexidade, com base nos pontos CCQ/GSQ/IQ2_K acima. Para "quase sem perda" (≤ ×1,02) os dados apontam para ≥ 3,2 bpw médios nos experts (IQK-4 ×1,004 a 3,23 bpw; QTIP 3 bits ×1,045 em 70B denso).

### Gaps
- Nenhuma fonte compara um codebook de 2,5 bits com q4 afim no mesmo modelo MoE ≥ 200B com perplexidade.
- Não achei estudo de "fine-tuning só de codebooks" isolado (sem tocar nas normas/escalas) em 70B+.

---

## Pergunta 4 — Custo de calibração e computação; PTQ puro contra ajuste fino

### Takeaway
Os métodos de codebook com ajuste fino custam de dezenas a centenas de GPU-horas em 70B (QuIP# ~50 GPU-h de FT mais "algumas horas num nó de 8 GPUs"; VPTQ 19–20 h em 4×A100 ≈ 80 GPU-h; AQLM 3 d 18 h em 8×A100 ≈ 720 GPU-h; PV-Tuning 2–8 dias em 8 GPUs). Os escalares recentes custam 2,5–6 GPU-h em 70B (SignRoundV2). Em 1T MoE só o GSQ publicou um recipe: 1–2 nós de 8× H100/H200, 16,8M tokens de calibração. HIGGS é data-free.

### Cited Findings

| Método | Tipo | Calibração | Custo reportado |
|---|---|---|---|
| QuIP# | PTQ + FT de codebooks/escalas/normas | Hessianas de RedPajama (tamanho não extraído); FT em "pequeno dev set" | 70B: "algumas horas num nó de 8 GPUs" para quantizar; FT ~50 GPU-h — [QuIP#](https://arxiv.org/html/2402.04396) |
| QTIP | PTQ + FT (mesmos hiperparâmetros do QuIP#) | Hessianas: Llama 2 6144×2048 (≈12,6M tokens); Llama 3/3.1 4096×8192 (≈33,6M); 405B 2048×8192 (≈16,8M, "por falta de tempo") | GPU-horas não reportadas — [QTIP](https://arxiv.org/html/2406.11235) |
| AQLM | PTQ + FT por bloco | RedPajama 4096 ctx, 1024 amostras no exemplo | 7B: ~1 dia em 1 A100; 70B: 10–14 dias em 1 GPU ou 3 d 18 h em 8×A100 — [README](https://github.com/Vahe1994/AQLM) |
| PV-Tuning | FT global (índices + contínuos), destilação KL | amostra oficial de 1B tokens da RedPajama; batch 2^20 tokens | "2 dias em 8 GPUs para superar as baselines, 8 dias para convergir"; 1,5× o FT do QuIP#; 70B precisa de 8×A100 — [PV-Tuning](https://arxiv.org/html/2405.14852) |
| VPTQ | PTQ + FT leve por camada (centróides, norms, 100 it.) | 128 segmentos de C4; Hessianas de RedPajama-1T-Sample | 70B: 19–20 h em 4×A100-80GB; 405B "~17 h" (hardware não dito) — [VPTQ](https://arxiv.org/html/2409.17066), [README](https://github.com/microsoft/VPTQ) |
| GPTVQ | PTQ (GPTQ-style, sem FT) | 128×2048 tokens WikiText2 (≈262k) | 70B: 3–11 h em 1 H100 — [GPTVQ](https://arxiv.org/html/2402.15319) |
| HIGGS | data-free (variante dinâmica: 287k tokens para KL) | nenhuma | não reportado — [HIGGS](https://arxiv.org/html/2411.17525) |
| SignRoundV2 | PTQ escalar com sinal aprendido | 128×2048 tokens (Pile) | Llama2-70B em A100-80GB: 2,5 GPU-h (V2*, 512 amostras: 6 h); a mesma tabela dá QuIP# 270 h, AQLM 336 h, EfficientQAT 41 h — [SignRoundV2](https://arxiv.org/html/2512.04746) |
| GSQ | PTQ escalar por bloco (Gumbel-Softmax) | 4096×4096 tokens (≈16,8M; OpenThoughts para Kimi) | Kimi-K2.5: 8× H100/H200, 1–2 nós; 10 épocas por bloco — [repo GSQ](https://github.com/IST-DASLab/GSQ) |
| Bielik 11B (comparativo) | — | Hessianas 512×4096 CulturaX | QuIP# 8–10 h A100; QTIP 12 h A100 (~14 min/camada) + 2 h FT e2e em H200; VPTQ 7 h 26 min H200; AQLM 38 h A100 (~46 min/camada); Hessianas 40 min H200 — [Bielik-Q2-Sharp](https://arxiv.org/html/2603.04162) |
| mlx-lm DWQ | FT de escalas/biases por destilação | 1024 amostras × 2048 tokens (padrão) | sem tempo reportado — [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) |

### Inferences
- Escalando o custo por camada do Bielik (QTIP ~14 min/camada de 11B numa A100) pela razão de parâmetros, as 61 camadas do Kimi K2 (384 experts por camada) passariam de 1000 GPU-horas de A100 só na quantização por Hessiana — coerente com a afirmação do GSQ de que VQ é "proibitivo" em MoE de 1T. PTQ escalar com Hessiana (GPTQ-like) e HIGGS data-free são os únicos com custo compatível com um Mac ou com uma instância única.
- O fator mais barato que mais recupera qualidade é o ajuste fino de parâmetros contínuos com destilação (QuIP# ~50 GPU-h em 70B dá −6% de W2 em 2 bits); isso é exatamente o que o `mlx_lm.dwq` faz (escalas/biases) e o que o projeto já tentou com DWQ em 20k tokens (overfit). O PV-Tuning usa 1B tokens; o QuIP# não diz, mas o batch do PV-Tuning é 1M tokens.

### Gaps
- GPU-horas de QTIP/HIGGS em 70B e 405B não estão nos papers.
- Tokens exatos de calibração do QuIP# e da variante do VPTQ para 405B.

---

## Pergunta 5 — Custo de decode: kernels de lookup/trellis são rápidos em decode limitado por memória? O que existe em Metal/MLX?

### Takeaway
Em GPU NVIDIA, lookup (E8P) e trellis (QTIP HYB) atingem a mesma velocidade e são limitados por banda: 2 bits dá ~3,4× o fp16 em 7B e 22–23,5 tok/s em Llama-2-70B numa RTX 6000 Ada; AQLM com codebook 1x16 é ~2,7× mais lento que isso; VPTQ e CCQ reportam GEMV mais rápido que AQLM/VPTQ respectivamente. Em CPU/NPU (GPTVQ) o decode por LUT também bate INT4. Não encontrei port de QTIP/QuIP#/AQLM para Metal; o MLX só tem `affine` (2/3/4/5/6/8 bits), `mxfp4`, `mxfp8` e `nvfp4`, mais os métodos aprendidos do mlx-lm (DWQ/AWQ/GPTQ/dynamic_quant); projetos comunitários portam K-quants/IQ do ggml (mlx-kquant) e codebooks de KV-cache (VeloxQuant, TurboQuant).

### Cited Findings
- Decode bs=1, RTX 6000 Ada (960 GB/s): Llama-2-7B FP16 55,9 tok/s; AQLM 2 bits 81,5; QuIP# 2 bits 186; QTIP 2 bits 188; QTIP 3 bits 161; QTIP 4 bits 140. Llama-2-70B: AQLM 8,78; QuIP# 22,2; QTIP 2 bits 23,5; 3 bits 19,1; 4 bits 16,3. — [QTIP](https://arxiv.org/html/2406.11235), [repo](https://github.com/Cornell-RelaxML/qtip)
- QuIP# em RTX 4090: 2-70B 2 bits 32,74 tok/s (56,8% da banda de pico); 2-7B 170,5 tok/s. Modelos de 3 bits são "significativamente mais lentos" por falta de kernel matvec otimizado. — [QuIP#](https://arxiv.org/html/2402.04396), [repo](https://github.com/Cornell-RelaxML/quip-sharp)
- VPTQ (medições próprias, conflitam com as dos autores do QuIP#): Llama-2-70B 2 bits VPTQ 9,7 tok/s, AQLM 6,9, QuIP# 1,9, GPTQ 2,38; 7B VPTQ 39,9 contra FP16 38,32. — [VPTQ Tabela 2](https://arxiv.org/html/2409.17066)
- CCQ, GEMV 4096×4096 M=1 em H20: VPTQ 0,058 ms, WINT2 padrão 0,027, CCQ 0,034 ("até 2× sobre VPTQ"); grouped-GEMM 256×1024×7168 M=64: W8A16 2,655 ms, W4A16 1,628, W2A16-CCQ 1,386. — [CCQ](https://arxiv.org/html/2507.07145)
- HIGGS/FLUTE em RTX 4090, Llama-3.1-8B: FP16 57 tok/s; FLUTE 2 bits 173, 3 bits 150, 4 bits 139 (batch 1). — [HIGGS](https://arxiv.org/html/2411.17525)
- Q-Palette, 4090, Llama-3.1-8B, speedup sobre FP16: a 2,5 bits VQ-CC 3,36×, VQ-TC 3,24×, TCQ-TC 3,13× (bs=1); a 2 bits TCQ-TC 3,57× contra QTIP 2,91×. — [Q-Palette](https://arxiv.org/html/2509.20214)
- GPTVQ em Snapdragon X Elite, Llama-3-8B: INT4 llama.cpp 4,5 bpv 17,95 tok/s; INT4 g128 23,81; VQ 2D 3,125 bpv 26,15 tok/s (−19% de memória, +10% de throughput). LUT de 6 bits → 8 bits. — [GPTVQ](https://arxiv.org/html/2402.15319)
- AQLM kernels: CUDA 1x16 "até ~1,3×", 2x8 "até ~3,0×", Numba Kx8 em CPU "até ~4,0×" (baseline não declarada). — [README AQLM](https://github.com/Vahe1994/AQLM)
- MLX `mx.quantize`: modos `affine` (grupos 32/64/128; 2, 3, 4, 5, 6, 8 bits; escala e bias no dtype de entrada), `mxfp4` (g32, escala e8m0), `mxfp8` (g32, e8m0), `nvfp4` (g16, escala e4m3, `global_scale`); nenhum modo de codebook/VQ. — [docs MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html). Issue aberta: a escala nvfp4 do MLX usa E4M3 com sinal em vez de UE4M3. — [espelho da issue 2962](https://upd.dev/ml-explore/mlx/issues/2962)
- mlx-lm: `mlx_lm.dwq` (ajusta escalas/biases e parâmetros não quantizados por destilação do modelo cheio; 1024 amostras × 2048; "funciona melhor de 2 a 4 bits", g32 ajuda), `mlx_lm.awq` (32 amostras), `mlx_lm.gptq`, `mlx_lm.dynamic_quant` (mixed precision por sensibilidade, 4/5 bits padrão). — [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)
- Metal comunitário: mlx-kquant (K-quants e iq4_nl nativos em Metal, tabelas de codebook IQ derivadas do ggml) — [repo](https://github.com/asher/mlx-kquant); VeloxQuant-MLX (kernels Metal de codebook para KV-cache, argmin em registradores) — [repo](https://github.com/meirm/VeloxQuant-MLX); turboquant-mlx (dequant por lookup de centróide, "correto, lento") — [repo](https://github.com/sharpner/turboquant-mlx). Nenhum port de QTIP/QuIP#/AQLM/VPTQ para Metal apareceu na busca.
- Bielik: QTIP exige o loader próprio do repositório Cornell e versões pinadas; QuIP# não suporta transformadas butterfly no kernel; SpinQuant precisa de R3/R4 em runtime que nenhum engine público implementa. — [Bielik-Q2-Sharp](https://arxiv.org/html/2603.04162)

### Inferences
- Para decode de 1 token num MoE, o que importa é bytes lidos por expert; um kernel de lookup com codebook de 1–2 KiB (E8P fonte, QTIP HYB) cabe em threadgroup memory do Metal, então a mesma lógica dos kernels CUDA deve valer no Mac. O custo extra é a Hadamard no produto (já existente no `cbkernel.rs` do projeto).
- Decode de 3 bits é sistematicamente mais lento que 2 e 4 nos kernels publicados (QuIP# sem kernel; QTIP 161 contra 188 e 140), por desalinhamento de bits — relevante para a escolha de 2,5 ou 3 bits no formato do arquivo.

### Gaps
- Nenhuma medição de kernel de codebook em Apple Silicon para pesos de LLM.
- O `mlx_lm.dwq` não publica números de KL/ppl nem tempo.

---

## Pergunta 6 — Checkpoints abertos de Kimi K2 / DeepSeek-V3 / Qwen3-235B nesses formatos e seus tamanhos

### Takeaway
Em formatos de codebook só existem: DeepSeek-R1 (VPTQ-community, v8/k65536[/256], sem ajuste fino, sem tamanho nem avaliação públicos), Llama-3.1-405B (QTIP relaxml 2/3/4 bits, variantes TP8; VPTQ 1,375–4 bits), Mixtral-8x7B e Qwen2-72B (AQLM/PV 1–2 bits), Mistral-Large-123B (VPTQ). Kimi K2 e Qwen3-235B só existem em escalar (GGUF Unsloth/ubergarm; GSQ do Kimi-K2.5 numa coleção ISTA sem tamanhos listados).

### Cited Findings
- relaxml: Llama-2-70B E8P 2 bits, E8PRVQ 3 e 4 bits; QTIP 2/3/4 bits para Llama 2/3/3.1 incluindo `Llama-3.1-405B-Instruct-QTIP-2Bit` (o card mostra "54B params" nos tensores int16 empacotados, sem tamanho em GB; variantes `TP8` com "qualidade um pouco pior"). — [repo QTIP](https://github.com/Cornell-RelaxML/qtip), [card 405B](https://huggingface.co/relaxml/Llama-3.1-405B-Instruct-QTIP-2Bit), [repo QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)
- ISTA-DASLab: Llama-2-70B AQLM 1x16 18,8 GB e 2x8 18,2 GB; Llama-3-70B 1x16 21,9 GB; PV 1x16g16 13 GB; Mixtral-8x7B 1x16 12,6 GB; Command-R+ 31,9 GB; Qwen2-72B(-Instruct) PV 1 e 2 bits 1x16; Llama-3.1-70B-Instruct PV 2 bits 18,8 GB. — [README AQLM](https://github.com/Vahe1994/AQLM)
- VPTQ-community: coleções para DeepSeek-R1 (2.x/2/3 bits), Llama-3.3-70B (4/3/2/1,875/1,625), Llama-3.1-405B (4/3/2/1,875/1,625/1,5/1,43/1,375), Mistral-Large-2407 123B, Qwen-2.5-72B (4/3/2,38/2,25/2/1,94); os bits dos nomes "excluem o overhead do codebook"; 70B × 3 bits / 8 = 26,25 GB sem overhead. — [README VPTQ](https://github.com/microsoft/VPTQ), [coleção DeepSeek-R1](https://huggingface.co/collections/VPTQ-community/vptq-deepseek-r1-without-finetune-67d0832c203afd208bb8449e)
- Kimi K2.6 GGUF (Unsloth): Dynamic 2-bit 340–350 GB; UD-Q4_K_XL 584 GB; UD-Q8_K_XL 595 GB; K2.5 tem 1 bit em 240 GB. — [Unsloth K2.6](https://unsloth.ai/docs/models/kimi-k2.6), [Unsloth K2.5](https://unsloth.ai/docs/models/kimi-k2.5)
- GSQ: coleção `ISTA-DASLab/gsq` com Kimi-K2.5 a 2,13 bpp (~260 GB), em compressed-tensors (vLLM) e GGUF. — [repo GSQ](https://github.com/IST-DASLab/GSQ)
- Qwen3-235B-A22B GGUF: ubergarm mix-IQ3_K 107 GiB; unsloth UD-Q3_K_XL 97 GiB; lmstudio Q3_K_L 104 GiB. — [gist ubergarm](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)
- MLX: os checkpoints `mlx-community` usam `affine` (o Kimi-K2-Instruct-4bit do projeto, 578 GB, é 4,5 bpw afim g64, cálculo meu a partir do formato). — [docs MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)

### Inferences
- Não há atalho: para ter Kimi K2 ou Qwen3-235B num codebook de 2–3 bits, o projeto precisa quantizar por conta própria. O VPTQ do DeepSeek-R1 é a única evidência de que alguém rodou VQ num MoE de 671B, e sem ajuste fino.

### Gaps
- Tamanhos em GB e qualidade dos checkpoints VPTQ de DeepSeek-R1 e QTIP 405B.
- Se existe algum checkpoint Qwen3-235B ou Kimi em VPTQ/QTIP depois de março de 2025: nenhum apareceu nas buscas.
