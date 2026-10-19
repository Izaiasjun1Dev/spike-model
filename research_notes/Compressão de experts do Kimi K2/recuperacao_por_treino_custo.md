# Recuperação de qualidade por treino em pesos de 2–3 bits e custo para 235B / 1T (out. 2026)

Escopo: métodos baseados em treino (QAT, destilação para pesos quantizados, ajuste
fino só dos parâmetros de quantização) que levam pesos de 2–3 bits perto da
qualidade de 4 bits; dados e computação que exigem; custo estimado para
Qwen3-235B-A22B e Kimi-K2 (1T); e, em segundo plano, ajuste fino só do roteador
para roteamento ciente de cache. Tudo que é estimativa minha está marcado como
**[estimativa]**. Números de modelos de 7–13B estão marcados como tal.

Convenções: tok = tokens; GPU-h = horas de GPU; ppl = perplexidade; "×" é a
razão entre a ppl quantizada e a ppl de referência. Preços em dólar, out. 2026.

## Pergunta 1 — Quais métodos levam 2–3 bits perto de 4 bits, e quanto recuperam?

### Takeaway

Em modelos de 70B, o ajuste fino bloco a bloco (EfficientQAT) e a quantização
vetorial com ajuste fino dos códigos discretos (PV-Tuning) levam 2 bits a
×1,2–1,4 da ppl fp16 (Llama-2-70B: 3,32 → 4,52–4,61 no EfficientQAT; 3,12 →
3,78 no PV-Tuning), e 3 bits a ×1,09 (3,32 → 3,61). Treinar só escalas/códigos
contínuos recupera parte do ganho (no 7B, ~45% do intervalo entre PTQ e o
resultado final); o resto exige mexer nos pesos ou nos códigos discretos. Não
existe, até out. 2026, nenhum lançamento público de QAT em 2 bits para
DeepSeek-V3 ou Kimi K2; o Kimi K2 Thinking é QAT em INT4 só nos experts.

### Cited Findings

**Resultados em 70B (ppl WikiText-2 / C4, contexto 2048)**

| Modelo | Método | Bits | WikiText-2 | C4 | × ppl (Wiki) | Fonte |
|---|---|---|---|---|---|---|
| Llama-2-70B | fp16 | 16 | 3,32 | 5,52 | 1,00 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | GPTQ (PTQ) | 3 g128 | 3,85 | 5,85 | 1,16 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | OmniQuant (escalas/clipping aprendidos) | 3 g128 | 3,78 | 5,85 | 1,14 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | EfficientQAT | 3 g128 | 3,61 | 5,71 | 1,09 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | OmniQuant | 2 g128 | 6,55 | 8,52 | 1,97 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | EfficientQAT | 2 g128 | 4,61 | 6,48 | 1,39 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | EfficientQAT | 2 g64 | 4,52 | 6,38 | 1,36 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-2-70B | AQLM (codebook 2×8, PTQ + FT) | ~2 | 4,49 | 6,36 | 1,35 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-3-70B | fp16 | 16 | 2,85 | 6,73 | 1,00 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-3-70B | EfficientQAT | 3 g128 | 4,21 | 7,46 | 1,48 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-3-70B | EfficientQAT | 2 g128 | 6,38 | 9,53 | 2,24 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| Llama-3-70B | OmniQuant | 2 g128 | 16,79 | 15,06 | 5,9 | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |

- A tabela do PV-Tuning usa outra baseline fp16 (3,12 / 4,97 para Llama-2-70B;
  contexto diferente). Nela: AQLM 2,07 bits = 3,94 / 5,72 (×1,26); PV-Tuning
  2,07 bits = 3,78 / 5,56 (×1,21). Para 13B: fp16 4,57, AQLM 5,65, PV-Tuning
  5,12 (×1,12); para 7B: fp16 5,12, AQLM 6,64, PV-Tuning 5,84 (×1,14) —
  [PV-Tuning, Tabela 2](https://arxiv.org/html/2405.14852v2).
- Ablação do PV-Tuning (**Llama-2-7B**, AQLM a 2,01 bits, Wiki2 / C4): só
  calibração 7,38 / 9,34; só parâmetros contínuos (escalas/codebooks) 6,69 /
  8,77; STE 6,41 / 8,63; PV em subespaço 6,13 / 8,35; PV + STE 5,90 / 7,43 —
  [PV-Tuning, Tabela 1](https://arxiv.org/html/2405.14852v2). Ou seja, ajustar só
  o contínuo tira ~0,7 de ppl; mexer nos códigos discretos tira outros ~0,8.
- Ablação do EfficientQAT (**Llama-2-7B**, w2g64): nada = ppl 453; só Block-AP
  (treina todos os parâmetros do bloco) = 8,53; só E2E-QP (treina só os passos
  de quantização, fim a fim) = 9,33; os dois = 7,68. O ganho vem sobretudo do
  bloco a bloco; o E2E só com escalas acrescenta ~0,85 de ppl —
  [EfficientQAT](https://arxiv.org/html/2407.11062v3).
- OmniQuant (só clipping e transformações equivalentes aprendidos, 128
  amostras): Llama-2 7–70B em 1–16 h numa A100-40G; a 3 bits no 70B fica em
  3,78 contra 3,31 fp16 (×1,14), mas a 2 bits degrada muito (6,55 no 70B) —
  [OmniQuant](https://arxiv.org/abs/2308.13137v3), [SliM-LLM](https://arxiv.org/pdf/2405.14917).
- BitDistiller (autodestilação com KL "confidence-aware", **só até 13B** para
  linguagem geral; 34B em código): Llama-2-13B bf16 ppl 4,88 / MMLU 55,5; 3 bits
  5,20 / 53,2; 2 bits 6,78 / 37,5 (OmniQuant 2 bits 25,69 / 26,1; GPTQ 15,08 /
  23,7). Llama-2-7B 2 bits 8,08 contra 5,47 bf16 (×1,48). Com só 2 mil amostras
  o WizardCoder-7B leva ~3 h numa A100-80G; o LLM-QAT no mesmo modelo usou 100
  mil amostras e ~281 GPU-h (o LLM-QAT original usou 64 GPUs) —
  [BitDistiller](https://arxiv.org/html/2402.10631v1).
- ParetoQ (Meta, 2025): **só até Llama-3-8B** (e MobileLLM 125M–1,5B). Llama-3-8B:
  fp16 ppl 6,2 / acc 70,2; 2 bits 8,0 / 67,0; 3 bits 7,0 / 68,4; 4 bits 6,8 /
  69,0. Conclusão deles: ternário, 2 e 3 bits ficam no mesmo nível na curva
  tamanho × acurácia e superam 4 bits e binário; há uma transição entre 2 e 3
  bits (a 3+ bits os pesos ficam perto da distribuição pré-treinada; a 2 bits e
  abaixo mudam drasticamente) — [ParetoQ](https://arxiv.org/html/2502.02631v2),
  [blog PyTorch](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/).
- Kimi K2 Thinking: "QAT na fase de pós-treino, com quantização INT4 só de pesos
  nos componentes MoE"; 1T total, 32B ativos, 384 experts + 1 compartilhado,
  top-8, 61 camadas (1 densa); "~2× de velocidade de geração, sem perda"; todos
  os benchmarks reportados em INT4; **nenhuma comparação com bf16/fp8 e nenhuma
  variante abaixo de 4 bits** — [model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking).
- Reprodução do pipeline INT4 QAT pela equipe de RL do SGLang (slime +
  Megatron-LM): pesos mestres bf16, quantização falsa + STE, W4A16; em
  Qwen3-30B-A3B, Qwen3-235B-A22B e "K2-like ~1 TB"; o INT4 QAT acompanha bf16
  no reward de treino e no AIME-2024, com gap de logprob treino–inferência
  menor que o do FP8; INT4 PTQ sem QAT acumula erro ao longo dos passos. Não
  publicam GPU-h, tokens nem memória; não tentam 2–3 bits —
  [LMSYS blog](https://www.lmsys.org/blog/2026-01-26-int4-qat).
- mlx-lm DWQ: "ajusta os parâmetros não quantizados (inclusive escalas e biases
  de quantização)"; o professor é o modelo não quantizado, e sugerem destilar de
  um professor em 8 bits para economizar memória; padrões: 1024 amostras,
  comprimento 2048 (sugerem 512 para economizar), batch 8, 4 bits; "funciona
  melhor destilando para 2–4 bits"; group size 32 dobra os parâmetros
  treináveis e pode melhorar; nada sobre overfitting, MoE ou modelos grandes —
  [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md).
- Cards comunitários de DWQ num MoE de ~230B (MiniMax-M2, catalystsec): batch 1,
  allenai/tulu-3-sft-mixture, ~1,1 M tokens, mlx-lm 0.28.4–0.30.7; ppl de 4 bits
  6,581 → 6,431 com DWQ, 3 bits 7,802 → 7,434 (avaliação em 210 amostras × 512
  tok) — [card MiniMax-M2-4bit-DWQ](https://huggingface.co/catalystsec/MiniMax-M2-4bit-DWQ/blob/main/README.md),
  [card 3bit](https://huggingface.co/catalystsec/MiniMax-M2-3bit-DWQ/blob/main/README.md)
  (só vi o trecho de busca; a página devolve 401 ao fetch).
- Não achei nenhum lançamento ou paper de QAT em 2 bits para DeepSeek-V3 ou Kimi
  K2 (busca em out. 2026 devolve só comparativos de produto) — busca sem
  resultado relevante ([exemplo](https://www.clarifai.com/blog/kimi-k2-vs-deepseek-v3/r1)).

### Inferences

- Para o alvo do projeto (codebook de 2,5 bits a ×1,14 como PTQ puro), os
  resultados de 70B sugerem que ajuste fino bloco a bloco + escalas/codebooks
  deve chegar a algo entre ×1,05 e ×1,10 (3 bits no EfficientQAT dá ×1,09; 2
  bits ×1,36; 2,5 bits em codebook está entre os dois, e o codebook parte de
  um erro menor que o afim). Mexer nos códigos discretos (PV-Tuning) rende mais
  ~0,05–0,1 de ×, ao custo de muito mais computação.
- Só escalas (DWQ, E2E-QP, OmniQuant) recupera cerca de metade do intervalo
  PTQ → melhor resultado nos 7B; no projeto, isso daria ×1,14 → ~×1,07–1,10
  **[estimativa]**. Para passar disso, é preciso treinar os pesos dos experts
  dentro do bloco (Block-AP) ou os índices do codebook.
- A degradação relativa cresce com a qualidade do modelo base (Llama-3-70B a 2
  bits ×2,24 contra Llama-2-70B ×1,39). Qwen3-235B e Kimi K2, treinados com
  muitos tokens, tendem ao comportamento do Llama-3, o que torna 2 bits puros
  mais caros em qualidade do que as tabelas do Llama-2 sugerem; 2,5–3 bits é a
  faixa segura.
- O DWQ do projeto com ~20 mil tokens não é comparável ao uso típico: o padrão
  do mlx-lm é 1024 × 2048 ≈ 2 M tokens, e os cards de MoE de 230B usam ~1,1 M.

### Gaps

- Nenhum resultado publicado de QAT ou PV-Tuning em 2–3 bits para modelos MoE
  de 235B+ (todas as tabelas de 70B são densas; ParetoQ para em 8B).
- Não achei ppl antes/depois para OmniQuant/AffineQuant/CBQ a 2 bits em 70B
  além do número do EfficientQAT (6,55) — e nada de CBQ.
- O model card do Kimi K2 Thinking e o blog do LMSYS não dão tokens, GPU-h ou
  memória do QAT INT4.

## Pergunta 2 — Quantos tokens são necessários a 2–2,5 bits, e MoE precisa de menos?

### Takeaway

Há dois regimes: o bloco a bloco com destilação local satura com ~10–20 M
tokens (EfficientQAT: 4096 × 2048 no Block-AP e 4096 × 4096 no E2E-QP), e o QAT
fim a fim do ParetoQ satura em ~10 B tokens a 3–4 bits e ~30 B tokens a 1–2
bits (em modelos de até 8B). Não encontrei nenhuma evidência publicada sobre MoE
precisar de menos tokens por expert; a aritmética aponta o contrário.

### Cited Findings

- ParetoQ: com orçamento de 100 B tokens, o ótimo é ~90% em pré-treino fp e
  ~10% em QAT; o ajuste fino QAT satura em ~10 B tokens a 3–4 bits e ~30 B
  tokens a 1/1,58/2 bits; o Llama-3-8B a 1,58 bits usou 30 B tokens (contra 100
  B do "1-bit Era"); 120 mil iterações para ≤2 bits e 40 mil para 3–4 bits —
  [ParetoQ](https://arxiv.org/html/2502.02631v2).
- EfficientQAT: Block-AP com 4096 amostras × 2048 (≈8,4 M tokens), batch 2, 2
  épocas por bloco; E2E-QP com 4096 × 4096 (≈16,8 M tokens), batch 32, 1 época
  (a contagem de tokens é aritmética minha sobre os números da tabela) —
  [EfficientQAT](https://arxiv.org/html/2407.11062v3).
- PV-Tuning: dados de RedPajama; 7B cabe em 1 GPU; 70B "exige um servidor com
  8×A100 ou offload para RAM"; "a corrida mais longa levou 2 dias em 8 GPUs
  para superar todas as baselines e 8 dias para convergir totalmente" (o texto
  não diz explicitamente que é o 70B) — [PV-Tuning](https://arxiv.org/html/2405.14852v2).
- AQLM: no Llama-2-7B a ppl melhora conforme o número de amostras de calibração
  sobe de 128 para 4096 — [AQLM](https://arxiv.org/pdf/2401.06118).
- BitDistiller: 4 mil Alpaca + 2 mil WikiText (2:1) para linguagem geral; "a
  autodestilação usa só uma pequena parte dos datasets" —
  [BitDistiller](https://arxiv.org/html/2402.10631v1).
- OmniQuant: 128 amostras — [OmniQuant](https://arxiv.org/abs/2308.13137v3).
- DWQ (mlx-lm): 1024 amostras × 2048 por padrão; "mais amostras podem melhorar
  a qualidade mas aumentam o tempo" — [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md).

### Inferences

- Num MoE, cada expert roteado vê k/E dos tokens: 8/128 = 1/16 no
  Qwen3-235B e 8/384 = 1/48 no Kimi K2. Para dar a cada expert o mesmo número
  de tokens que uma camada densa recebe no EfficientQAT (8,4 M), seriam ~134 M
  tokens no 235B e ~400 M no Kimi **[estimativa]**. A hipótese "experts precisam
  de menos" não tem apoio na literatura que achei; o que vale é o número de
  tokens *por expert*, e experts raros ficam subtreinados (daí o overfit com 20
  mil tokens, que dá ~1,2 mil tokens por expert no Qwen3-30B).
- Por outro lado, os FLOPs por token são só os dos 22B/32B ativos, então 134 M
  tokens no 235B custam, em FLOPs, o mesmo que ~12 M tokens num denso de 235B.
- O regime do ParetoQ (10–30 B tokens) é QAT de verdade, com todos os pesos; é
  o que uma empresa faz (Kimi K2 Thinking) e não é necessário para "escalas +
  codebooks" ou bloco a bloco. Para o projeto, 10–500 M tokens é a faixa
  relevante.

### Gaps

- Nenhuma curva "tokens × ppl" publicada para 2–3 bits em 70B+ nem em MoE; os
  pontos de saturação do ParetoQ vêm de modelos de 125M–8B.
- Nenhuma fonte sobre eficiência de dados por expert em QAT de MoE.

## Pergunta 3 — Memória, infraestrutura e custo para 235B e 1T

### Takeaway

Bloco a bloco (EfficientQAT Block-AP / DWQ por camada) nunca precisa do modelo
inteiro na GPU: um bloco do Kimi tem ~16 B parâmetros (~33 GB em bf16), e o
resto fica em disco; o custo é dominado por I/O e pelo cache de ativações
(centenas de GB a ~2 TB em NVMe). Estimo **US$ 1–3 mil para 235B e US$ 3–8 mil
para o Kimi** num nó de 8 GPUs por alguns dias, ou um único H200 por mais
tempo. QAT fim a fim no estilo ParetoQ (10–30 B tokens) custa ~4–5 mil GPU-h
H100 no 235B (US$ 25–40 mil) e ~5–20 mil GPU-h no Kimi (US$ 40–150 mil), e
exige vários nós só para caber. Preços de out. 2026: p5 US$ 5,97–6,88 /
GPU-h, p5en (H200) 7,90, p6-b200 14,2; RunPod H200 3,59–5,29; Lambda H100
2,69–4,29.

### Cited Findings

**Preços (out. 2026)**

| Oferta | Preço | Fonte |
|---|---|---|
| AWS Capacity Blocks p5.48xlarge (8×H100) | US$ 47,76/h = 5,97/acelerador-h | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| AWS Capacity Blocks p5e.48xlarge (8×H200) | 54,92/h = 6,87/acel.-h | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| AWS Capacity Blocks p5en.48xlarge (8×H200) | 63,16/h = 7,90/acel.-h | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| AWS Capacity Blocks p6-b200.48xlarge (8×B200) | 113,67/h = 14,21/acel.-h | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| AWS Capacity Blocks u-p6e-gb200x72 | 761,90/h = 10,58/acel.-h | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/) |
| AWS on-demand p5 / p5en / p6-b200 (terceiros) | ~55,0 / 63,3 / 113,9 por nó-h | [packet.ai](https://packet.ai/blog/aws-gpu-pricing-vs-packet-ai-blackwell), [holori](https://calculator.holori.com/aws/ec2/p6-b200.48xlarge) |
| Capacity Blocks subiram ~15% em jan. e ~20% em jul. 2026 | — | [spheron](https://www.spheron.network/blog/aws-capacity-blocks-pricing-2026-b200-b300-hike/) |
| RunPod H100 SXM / H200 / B200 (pods) | 3,99 / 5,29 / 7,99; community 2,69 / 3,59 / 5,98; secure 3,49 / 4,59 / 6,79 (página atualizada 8/10/2026) | [RunPod](https://www.runpod.io/pricing) |
| RunPod cluster H200 SXM (até 64 GPUs) | 4,31/h | [RunPod](https://www.runpod.io/pricing) |
| Lambda H100 / B200 | 2,69 (1 sem.–3 meses) e 3,79 (B200, semanal) na página própria; agregadores mostram 3,99–4,29 (H100) e 6,69–6,99 (B200) | [Lambda](https://lambda.ai/service/gpu-cloud/1-click-clusters), [computeprices](https://computeprices.com/providers/lambda/gpus/b200), [costbench](https://costbench.com/software/ai-gpu-cloud/lambda-labs/) |

**Custos reportados nos papers**

| Método | Modelo | Hardware | Tempo | Memória | Fonte |
|---|---|---|---|---|---|
| EfficientQAT Block-AP | Llama-2-70B | 1×A100-80G | 26,6 h | 29,9 GB | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| EfficientQAT E2E-QP | Llama-2-70B | 1×A100-80G | 14,3 h | 48,4 / 42,0 / 34,2 GB (4/3/2 bits) | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| EfficientQAT total | Llama-2-70B 2 bits | 1×A100-80G | 41 h (outros métodos: ~300–900 h) | — | [EfficientQAT](https://arxiv.org/html/2407.11062v3) |
| PV-Tuning | Llama-2-70B | 8×A100 (ou offload p/ RAM) | 2 dias p/ superar baselines, 8 dias p/ convergir | sharding + offload de otimizador | [PV-Tuning](https://arxiv.org/html/2405.14852v2) |
| OmniQuant | Llama-2 7–70B | 1×A100-40G | 1–16 h | — | [OmniQuant](https://arxiv.org/abs/2308.13137v3) |
| BitDistiller | WizardCoder-7B | 1×A100-80G | 3,0 h (1,47 gerar dados, 0,63 init, 0,92 QAT) | — | [BitDistiller](https://arxiv.org/html/2402.10631v1) |
| LLM-QAT | WizardCoder-7B | 1×A100-80G | ~281 GPU-h (100 mil amostras) | — | [BitDistiller](https://arxiv.org/html/2402.10631v1) |
| ParetoQ | até Llama-3-8B | 16 GPUs (8×8 nos experimentos de agenda) | não informado | — | [ParetoQ](https://arxiv.org/html/2502.02631v2) |
| Cache-aware router (joint) | Qwen3-30B-A3B | 8 aceleradores × 140 GB | não informado (4 épocas) | — | [2609.04895](https://arxiv.org/html/2609.04895v2) |

- Truques que evitam carregar o modelo inteiro: o EfficientQAT treina bloco a
  bloco com entradas/saídas do bloco pré-calculadas (por isso 70B cabe em 30
  GB), e o E2E-QP só treina os passos de quantização —
  [EfficientQAT](https://arxiv.org/html/2407.11062v3). O DWQ sugere um
  professor de 8 bits em vez de 16 para cortar memória —
  [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md).
  O LMSYS observa que INT4 reduz o Kimi a ~500 GB, que cabe "num único
  servidor" (multi-GPU) — [LMSYS](https://www.lmsys.org/blog/2026-01-26-int4-qat).

**Estimativas de custo [estimativa — aritmética minha sobre as fontes acima]**

Premissas: FLOPs de treino ≈ 6 × parâmetros ativos × tokens (4× quando só
escalas são treináveis, porque não há GEMM de gradiente de peso); H100/H200 a
~300 TFLOPS efetivos (≈30% de utilização); parâmetros ativos 22 B (Qwen3-235B)
e 32 B (Kimi K2); uma camada do Kimi ≈ 1T/61 ≈ 16 B parâmetros (~33 GB bf16, ~8
GB a 4 bits); hidden 4096 (Qwen3-235B) e 7168 (Kimi).

| Cenário | Tokens | GPU-h (H100/H200) | Memória / infra | Custo (p5en 7,90; RunPod H200 ~4,3–5,3) |
|---|---|---|---|---|
| (a1) 235B, bloco a bloco (Block-AP + escalas), dados do EfficientQAT | 8,4 M × 2 épocas + 17 M | ~10–40 (limitado por I/O: ler 470 GB bf16 ou 118 GB q4 por bloco; cache de ativações 8,4 M × 4096 × 2 B ≈ 69 GB por bloco) | 1 GPU de 80–141 GB + ~1 TB NVMe | US$ 100–300 |
| (a2) 235B, bloco a bloco com 16× tokens (134 M, ~8 M por expert) | 134 M | ~300–900 | cache por bloco ≈ 1,1 TB em NVMe; 1 nó de 8 GPUs em 2–5 dias | US$ 2–6 mil (1 nó p5en 2–5 dias ≈ 3–7,5 mil; RunPod ≈ 1,5–4 mil) |
| (a3) 235B, só escalas/codebooks fim a fim (DWQ / E2E-QP), professor q4 ou 8 bits | 17–134 M | ~20–200 | aluno q2,5 (~75 GB) + professor q4 (118 GB) ou 8 bits (235 GB) + ativações: 1 nó 8×H100 (640 GB) ou 8×H200 (1,1 TB) | US$ 200–1,6 mil |
| (a4) 235B, QAT fim a fim estilo ParetoQ | 10–30 B | ~3,7–11 mil (6 × 22 B × 30 B = 4e21 FLOPs) | todos os pesos treináveis: ~16 B/param ≈ 3,8 TB → 4–6 nós de 8×H200; teacher extra | US$ 30–90 mil |
| (b1) Kimi 1T, bloco a bloco, dados do EfficientQAT | 8,4 M × 2 + 17 M | ~30–100 (I/O: 578 GB q4 ou 2 TB bf16 lidos uma vez por passada; cache 8,4 M × 7168 × 2 B ≈ 120 GB por bloco) | 1 GPU de 141 GB (bloco de 33 GB bf16 + Adam dos experts do bloco) + ~2 TB NVMe | US$ 250–800 |
| (b2) Kimi 1T, bloco a bloco com 48× tokens (400 M, ~8 M por expert) | 400 M | ~1–3 mil | cache por bloco ≈ 5,7 TB (ou subamostrar) | US$ 8–25 mil |
| (b3) Kimi 1T, só escalas/codebooks fim a fim, professor = o próprio INT4 oficial | 17–400 M | ~50–600 | aluno q2,5 (~360 GB) + professor INT4 (578 GB) + ativações ≈ 1–1,2 TB → 1 nó 8×H200 (1,13 TB) no limite, ou 2 nós; com logits do professor pré-computados, 1 nó | US$ 400–5 mil |
| (b4) Kimi 1T, QAT fim a fim estilo ParetoQ | 10–30 B | ~5,3–16 mil (6 × 32 B × 30 B = 5,8e21) | pesos mestres bf16 + Adam ≈ 16 TB → 16+ nós de 8×H200; professor 2 TB bf16 à parte | US$ 40–130 mil só de GPU |
| Roteador só (ReMoE-like), 235B | 33 M (2000 passos × 8 × 2048) | ~10–30 (4 × 22 B × 33 M ≈ 3e18 FLOPs; o custo real é o forward/backward completo) | pesos congelados q4 (118 GB) + ativações → 1 nó 8×H100 | US$ 100–300 |
| Roteador só, Kimi 1T | 33 M | ~15–50 | pesos congelados INT4 (578 GB) + ativações → 1 nó 8×H200 | US$ 150–500 |

### Inferences

- O professor em bf16 (2 TB no Kimi) nunca precisa existir na GPU: (i) bloco a
  bloco, o "professor" é a saída do próprio bloco em precisão alta, calculada
  uma vez e guardada; (ii) o Kimi K2 Thinking é nativamente INT4 — o checkpoint
  de 4 bits *é* o modelo de referência, então destilar de q4 não perde nada
  frente ao original; para o Qwen3-235B, o q4 (ou 8 bits, como o DWQ sugere)
  é um professor adequado dado que a meta é "chegar perto do q4".
- O gargalo do bloco a bloco em 1T não é FLOPs, é I/O: cada bloco precisa das
  ativações de entrada de todos os tokens de calibração. Com 400 M tokens e
  hidden 7168, são 5,7 TB por bloco em f16; na prática, subamostrar por bloco
  ou treinar vários blocos por passada.
- Um nó de 8×H200 (p5en, US$ 63/h; RunPod cluster US$ 34/h) por 2–5 dias cobre
  (a2), (a3) e (b1)/(b3). O QAT fim a fim (a4/b4) é outra ordem de grandeza e
  só faz sentido para quem publica o modelo.
- O resultado de 70B do EfficientQAT (41 h em A100) sugere que uma passada de
  qualidade razoável no 235B cabe numa única GPU de 141 GB em ~2–4 dias se o
  I/O for bem feito.

### Gaps

- Nenhuma fonte dá GPU-h para QAT/DWQ em modelos de 235B+ (nem o Kimi, nem o
  LMSYS, nem os cards de DWQ); todas as linhas da tabela de custo são
  extrapolações de 7–70B.
- Preços on-demand de p5/p5en/p6 vêm de agregadores (packet.ai, holori); só
  confirmei na AWS os preços de Capacity Blocks. Preços spot de p5 não
  encontrados.

## Pergunta 4 — Ajuste fino só do roteador para roteamento ciente de cache

### Takeaway

Existe literatura recente (2026) e ela diz duas coisas: (1) treinar só o gate
com os experts congelados, com um termo de reuso + âncora KL ao roteador
original, dá +25–30% de reuso de experts e +3–5 pontos de hit rate em LRU com
perda de ≤1 ponto nos benchmarks (ReMoE, DeepSeek-V2-Lite e Qwen1.5-MoE); (2)
para ganhos grandes de hit rate (61% → 73–91%) é preciso treinar o backbone
junto, e aí a acurácia cai 0–3 pontos (Qwen3-30B-A3B). Um estudo pré-registrado
em modelos pequenos conclui que redução de misses e qualidade são fortemente
acopladas (nenhuma configuração passou num gate de ≤1% de ppl), e que o
reroteamento ciente de cache em tempo de inferência (o `bias` do projeto)
combinado com treino chega a ~80% menos misses com ≤3,4% de ppl.

### Cited Findings

- ReMoE (ICML 2026; DeepSeek-V2-Lite 15,7B/2,4B ativos, 64 experts top-6, e
  Qwen1.5-MoE-A2.7B): treina **só os pesos do gate**; embeddings, atenção e
  experts congelados; perda = CE + λ·KL(roteador congelado como âncora) +
  α·L_loc (reuso, suavidade, inércia, working set); balanceamento de carga
  desligado; dados OpenHermes-2.5 (100 mil amostras), 2000 passos, AdamW 5e-5,
  seq 2048, micro-batch 1, acumulação 8 (≈33 M tokens, aritmética minha).
  Qualidade (baseline → ReMoE): GSM8K 38,89 → 38,13; HumanEval 26,83 → 29,27;
  MMLU 57,72 → 57,81; IFEval strict 17,19 → 16,08. Reuso: overlap de experts
  27,3% → 34,5% (+26%); simulador LRU com cache de 6: hit rate único 0,319 →
  0,369; vLLM com offload numa RTX 3090 3,58 → 3,88 tok/s (+8,4%); Jetson Orin
  NX com llama.cpp TPOT −43,6–49,8%. Chama o reroteamento em tempo de inferência
  (Mixture of Cache-Conditional Experts, Skliar et al. 2025) de complementar;
  não há comparação direta — [ReMoE](https://arxiv.org/html/2605.27081),
  [ICML](https://icml.cc/virtual/2026/poster/60585).
- Cache-Aware Joint Router Adaptation (Huawei et al., 2609.04895;
  Qwen3-30B-A3B-Instruct-2507 e GPT-OSS-20B; treino em GSM8K/MATH/CSQA, 4
  épocas, nó de 8 aceleradores × 140 GB): modo "só roteador auxiliar"
  (backbone congelado) mantém a acurácia (GSM8K 85,44 → 85,44; MATH 58,22 →
  57,82; CSQA 87,39 → 86,16) e sobe o hit rate 61,2 → 73,1% (GSM8K) e 56,4 →
  91,6% (CSQA), cortando os bytes carregados por token de 1407 → 974 MB (GSM8K)
  e 1579 → 304 MB (CSQA); mas num ablation com s_w = 0,1, só-auxiliar deu 63,0%
  contra 61,2% do baseline (ganho "modesto"). O modo conjunto (treina o backbone)
  chega a 90,6 / 88,4 / 95,6% de hit, com acurácia 83,40 / 57,66 / 84,11 (−2,0 /
  −0,6 / −3,3 pontos). Conclusão deles: "treinar só o auxiliar preserva a
  acurácia mas dá ganhos modestos de cobertura; pós-treino conjunto remodela a
  distribuição de seleção" — [2609.04895](https://arxiv.org/html/2609.04895v2).
- "Cacheable by Design?" (2608.18261; MoEs de 137M e 340M treinados com perdas
  auxiliares de localidade; traces do Qwen3-30B; Qwen3-235B Q4_K_M de 134 GB
  numa GPU de 8 GB para medir banda): perdas de localidade reduzem misses em até
  60%, mas **toda configuração falhou o gate pré-registrado de ≤1% de ppl**; a
  perda não diminui com a escala (sobe um pouco em 340M); StickyMoE
  (concorrente) relata custo quase nulo só em modelos de domínio único <25M;
  reroteamento ciente de cache sem treino + localidade treinada: ~80% menos
  misses com ≤3,4% de ppl; LRU com 13,4% dos experts serve 66% dos pedidos no
  trace do Qwen3-30B; prefetch por domínio não ajudou —
  [2608.18261](https://arxiv.org/abs/2608.18261).
- melinoe (2602.11192): perda auxiliar que penaliza transferências de cache;
  1,2–3× de throughput "mantendo ou melhorando" o desempenho (só vi o resumo da
  busca) — [2602.11192](https://arxiv.org/pdf/2602.11192).
- Kimi K2 Thinking: 384 experts, top-8 + 1 compartilhado; parâmetros do gate
  por camada = 7168 × 384 ≈ 2,75 M (aritmética minha sobre o card) —
  [model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

### Inferences

- O `bias1` do projeto (≈83% de acerto com 16 GB, ×1,057) é exatamente o
  "reroteamento em tempo de inferência" que o 2608.18261 considera a base
  forte; os papers sugerem que o treino do gate soma pouco por cima dele em
  hit rate, mas pode recuperar parte da qualidade perdida pelo bias (a âncora
  KL do ReMoE é, na prática, uma destilação do roteador original).
- O caminho mais promissor para o projeto: treinar só o gate com perda = KL
  para a distribuição completa do modelo (ou do roteador original, como no
  ReMoE) + termo de residência (dado o estado do cache LFU do próprio motor),
  com os experts congelados; ~30 M tokens bastaram no ReMoE. O custo é um
  forward/backward completo através do modelo congelado (não dos pesos), o
  que num 235B em q4 cabe num nó de 8×H100 e, em tese, num Mac de 256–512 GB
  **[estimativa]**.
- O resultado de 2608.18261 é um aviso: a ppl sobe junto com a localidade em
  modelos pequenos; o ReMoE e o 2609.04895 mostram que em 15–30B a perda cabe
  em ~1 ponto quando só o gate muda. Não há dado em 235B+.

### Gaps

- Nenhum paper treina o gate de um MoE de 235B+ para cache; não há medição de
  custo (GPU-h) de treino do gate em nenhum deles.
- Nenhuma comparação direta "bias em inferência" versus "gate treinado" em
  igualdade de hit rate.

## Pergunta 5 — Apple Silicon (128–512 GB) consegue destilar bloco a bloco um 235B?

### Takeaway

Memória não é o problema (um bloco do 235B em bf16 tem ~5 GB; do Kimi ~33 GB),
e os cards comunitários indicam que o DWQ do mlx-lm já roda em MoEs de ~230B
com ~1 M tokens. O limite é computação: o M3 Ultra tem ~65 TFLOPS fp16
teóricos (sem número oficial da Apple), ~15× menos que uma H100 — bloco a bloco
com ~10–20 M tokens é questão de dias; QAT fim a fim de bilhões de tokens é
inviável.

### Cited Findings

- M3 Ultra 512 GB: Apple não publica FLOPS; valores teóricos derivados de 2× o
  M3 Max: 32,8 TFLOPS fp32, 65,5 fp16, 131 INT8; 819 GB/s; teto de ~384 GB
  (75%) para a GPU; LoRA em 70B 16 bits e QLoRA em 70B+ "viáveis", treino
  completo prático só em 7–8B (estimativas do guia, não medições) —
  [apxml](https://apxml.com/hardware/apple-m3-ultra-512gb). Um fórum estimou
  ~115 TFLOPS fp16, especulativo — [hada.io](https://news.hada.io/topic?id=19617).
- DWQ em MiniMax-M2 (MoE ~230B) com ~1,1 M tokens, batch 1, mlx-lm 0.28–0.30:
  ppl 4 bits 6,581 → 6,431; 3 bits 7,802 → 7,434 (hardware e tempo não
  constam no trecho que vi) — [card](https://huggingface.co/catalystsec/MiniMax-M2-4bit-DWQ/blob/main/README.md).
- DWQ recomenda professor em 8 bits, seq 512 e batch 1 para economizar memória
  — [LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md).
- Não achei nenhum número publicado de tokens/s de treino do MLX num M3 Ultra
  (só inferência, ex.: MiniMax-M2.5 8 bits a 14,6 tok/s de geração) —
  [oMLX benchmarks](https://omlx.ai/benchmarks?page=969).

### Inferences

- **[estimativa]** A ~20–25 TFLOPS efetivos (30–40% dos 65 teóricos), o bloco a
  bloco do 235B com os dados do EfficientQAT (≈6 × 22 B × 25 M ≈ 3,3e18 FLOPs)
  leva ~40 h; com 134 M tokens (~8 M por expert), ~2–3 semanas. O Kimi, 1,5×
  isso. Em memória: bloco (5–33 GB) + Adam para os experts do bloco (se
  treinar pesos: ×4 do bloco) + cache de ativações em SSD (69 GB a 1,1 TB por
  bloco para 235B) — cabe em 128 GB para o 235B só com escalas/codebooks, e
  em 256–512 GB para o Kimi.
- Treino só de escalas (sem Adam sobre os experts) é o cenário mais leve e o
  mais compatível com o `engine/` do projeto, que já lê os experts por camada
  do SSD.
- O QAT fim a fim de 10 B tokens no 235B (1,3e21 FLOPs) levaria ~600 dias a 25
  TFLOPS: fora de questão.

### Gaps

- Nenhuma medição pública de throughput de treino (tokens/s, MFU) de MLX em
  M3/M4 Ultra com modelos de 100B+; nem relatos de DWQ no Qwen3-235B.
