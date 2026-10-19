# Quantização MoE e precisão mista para MoEs muito grandes (Kimi K2, DeepSeek-V3/R1, Qwen3-235B)

Notas de pesquisa (2026-10-13). Tamanhos em GB são os que a fonte publica; quando a fonte usa GiB (ubergarm), a tabela traz GiB e a conversão (×1,0737) entre parênteses. "bpw" = bits por peso sobre o modelo inteiro (inclui atenção, embeddings etc.); os bits só dos experts são menores que o bpw total nos quants que guardam a atenção em q8_0.

## 1. Quants "Dynamic 2.0" da Unsloth: tamanhos e qualidade

### Takeaway
Para o Kimi K2 (1T), a família UD vai de 244 GB (UD-TQ1_0) a 587 GB (UD-Q4_K_XL); para o DeepSeek-R1-0528 (671B), de 162 GB a 384 GB. A Unsloth publica KLD só para modelos pequenos (Gemma 3) e Aider Polyglot para o DeepSeek-V3.1; para o Kimi K2 e o Qwen3-235B não há número de qualidade por quant nos cards. No DeepSeek-V3.1, o Aider cai de 71,6% (BF16) para 69,7% (Q4_K_XL, 387 GB), 65,8% (Q2_K_XL, 255 GB), 61,2% (IQ2_XXS, 225 GB) e 50,7% (TQ1_0, 170 GB).

### Cited Findings

**Tamanhos (Kimi-K2-Instruct, Kimi-K2-Thinking, DeepSeek-R1-0528, Qwen3-235B-A22B-Instruct-2507)**

| Quant | Kimi-K2-Instruct | Kimi-K2-Thinking | DeepSeek-R1-0528 | Qwen3-235B-A22B-2507 |
|---|---|---|---|---|
| UD-TQ1_0 | 244 GB | 247 GB | 162 GB | — |
| UD-IQ1_S | 280 GB | 285 GB | 185 GB | — |
| UD-IQ1_M | 304 GB | 309 GB | 200 GB | — |
| UD-IQ2_XXS | 329 GB | 335 GB | 217 GB | — |
| UD-IQ2_M | 347 GB | 354 GB | 229 GB | — |
| Q2_K | 373 GB | 374 GB | 245 GB | 85,7 GB |
| UD-Q2_K_XL | 382 GB | 386 GB | 251 GB | 88,8 GB |
| UD-IQ3_XXS | 417 GB | 422 GB | 273 GB | — |
| Q3_K_S | 442 GB | 443 GB | 290 GB | 101 GB |
| UD-Q3_K_XL | 452 GB | 455 GB | 296 GB | 104 GB |
| IQ4_XS | 547 GB | 547 GB | 358 GB | 125 GB |
| IQ4_NL | 578 GB | 579 GB | 379 GB | — |
| UD-Q4_K_XL | 587 GB | 646 GB | 384 GB | 134 GB |
| Q4_K_M | 621 GB | 621 GB | 405 GB | 142 GB |
| Q8_0 | 1,09 TB | 1,09 TB | 713 GB | 250 GB |
| BF16 | 2,05 TB | 2,05 TB | 1,34 TB | 470 GB |

- Kimi-K2-Instruct-GGUF: tabela acima (o card não diz quais camadas ficam em mais bits nem traz KLD; os benchmarks do card são do modelo completo) — [unsloth/Kimi-K2-Instruct-GGUF](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF)
- Kimi-K2-Thinking-GGUF: tabela acima; o card diz que o modelo-fonte é "native INT4" (QAT nos componentes MoE) mas não explica como os GGUFs se relacionam com o checkpoint INT4; não há benchmark por quant — [unsloth/Kimi-K2-Thinking-GGUF](https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF)
- DeepSeek-R1-0528-GGUF: tabela acima; sem KLD nem benchmark por quant — [unsloth/DeepSeek-R1-0528-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)
- Qwen3-235B-A22B-Instruct-2507-GGUF: o card não lista UD-TQ1_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_M/IQ3_XXS; só os quants da tabela; sem KLD; os benchmarks (MMLU-Pro 83,0, Aider 57,3) são do modelo completo — [unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF)

**DeepSeek-R1 original (671B), os quants "1,58-bit"**

| Quant | Bits médios | Tamanho | Experts MoE | `down_proj` dos experts | Rótulo da Unsloth |
|---|---|---|---|---|---|
| UD-IQ1_S | 1,58 | 131 GB (listagem de arquivos: 140 GB) | 1,56 bit em tudo | mistura 2,06/1,56 | "Fair" |
| UD-IQ1_M | 1,73 | 158 GB (169 GB) | 1,56 bit | 2,06 bit | "Good" |
| UD-IQ2_XXS | 2,22 | 183 GB (196 GB) | 2,06 bit | mistura 2,5/2,06 | "Better" |
| UD-Q2_K_XL | 2,51 | 212 GB (227 GB) | 2,5 bit | mistura 3,5/2,5 | "Best" |

- Tabela acima e a recomendação de `--min-p 0.05` "especialmente para o 1,58-bit", para conter tokens raros — [unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF); mesma tabela em [docs Unsloth, DeepSeek-R1 how to run locally](https://unsloth.ai/docs/models/tutorials/deepseek-r1-how-to-run-locally.md)
- O card diz que os quants 1,58 e 2 bit são "selectively quantized, greatly improving accuracy over standard 1-bit/2-bit", mas não dá os bits da atenção, embeddings, primeiras camadas densas ou experts compartilhados — [unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)

**Como o Dynamic 2.0 escolhe os bits**
- "Dynamically adjust[s] the quantization type of every possible layer"; esquema específico por modelo; calibração com >1,5M tokens de dados "hand-curated" voltados a chat; funciona em MoE e não-MoE (a versão 1.0 só em MoE) — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- A página de Aider recomenda manter `attn_k_b` (e tensores afins do MLA) em precisão maior — [Unsloth, Dynamic GGUFs on Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-ggufs-on-aider-polyglot)

**KLD publicado (só para Gemma 3 27B, modelo denso; referência de magnitude por tipo de quant)**

| Quant | KLD (Dynamic 2.0) | GB | MMLU 5-shot |
|---|---|---|---|
| IQ1_S | 0,9729 | 6,06 | 41,87 |
| IQ1_M | 0,8000 | 6,51 | 48,10 |
| IQ2_XXS | 0,5210 | 7,31 | 59,20 |
| IQ2_M | 0,2582 | 8,96 | 66,47 |
| Q2_K_XL | 0,2209 | 9,95 | 68,70 |
| IQ3_XXS | — | 10,07 | 68,27 |
| Q3_K_XL | 0,0806 | 12,76 | 70,87 |
| Q4_K_XL | 0,0237 | 15,64 | 71,47 |
| Q8_0 | — | 26,74 | 71,60 |

- Tabela acima (modelo denso de 27B, não MoE) — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)

**Aider Polyglot, DeepSeek-V3.1 (671B), modo não-pensante, pass-2**

| Quant (UD) | GB | Aider % | Concorrente de tamanho parecido |
|---|---|---|---|
| TQ1_0 | 170 | 50,7 | — |
| IQ1_M | 206 | 55,7 | — |
| IQ2_XXS | 225 | 61,2 | 43,6 (164 GB) |
| IQ2_M | 235 | 64,3 | 56,6 (215 GB) |
| Q2_K_XL | 255 | 65,8 | 64,0 (239 GB) |
| IQ3_XXS | 279 | 66,8 | 65,6 (268 GB) |
| Q3_K_XL | 300 | 68,4 | 65,2 (293 GB) |
| IQ4_XS | 357 | 69,2 | 66,3 (360 GB) |
| Q4_K_XL | 387 | 69,7 | — |
| Q4_K_M | 405 | 69,7 | 67,7 (409 GB) |
| Q5_K_XL | 484 | 70,7 | 68,9 (478 GB) |
| BF16 | — | 71,6 | — |

- Tabela acima; modo pensante: 3-bit 75,6%, 2-bit 66,7%, 1-bit 57,8% contra 76,1% em BF16; ~3 rodadas por ponto, mediana; a maioria rodada por um contribuidor da comunidade (David Sluys) — [Unsloth, Dynamic GGUFs on Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-ggufs-on-aider-polyglot)
- Na mesma página, Kimi K2 (completo) 59,1% e Qwen3-235B-A22B 59,6% no Aider não-pensante; não há resultado por quant desses modelos — [mesma fonte](https://unsloth.ai/docs/basics/unsloth-dynamic-ggufs-on-aider-polyglot)

### Inferences
- A perda do DeepSeek-V3.1 no Aider é quase plana de Q4 a Q3_K_XL (69,7→68,4) e só cai forte abaixo de ~2,5 bits médios (65,8 em Q2_K_XL, 61,2 em IQ2_XXS, 50,7 em TQ1_0). Para um motor que já usa q4 nos picks mais pesados, a faixa 2,2–2,5 bits nos experts é onde a comunidade para de ver "quase sem perda".
- Nos quants 1,58/1,73-bit do R1 a Unsloth deixa o `down_proj` dos experts com mais bits que o gate/up (2,06 vs 1,56), repetindo o padrão do ubergarm (seção 2) e do MxMoE (seção 4): o `down_proj` é o mais sensível dentro do expert.
- O KLD do Gemma 3 serve só como escala: um IQ2_M/Q2_K_XL fica em KLD ~0,22–0,26, um Q4_K_XL em ~0,024, ou seja, ~10× de diferença entre 2 e 4 bits em modelo denso. Não transferir o valor absoluto para o Kimi.

### Gaps
- O blog original "DeepSeek-R1 1.58-bit" (unsloth.ai/blog/deepseekr1-dynamic) devolveu HTTP 403; não consegui confirmar pela fonte os bits da atenção/primeiras camadas densas/experts compartilhados, a fração de parâmetros nos experts nem os escores do teste Flappy Bird. As páginas que abriram não trazem esses números.
- Nenhum KLD/MMLU por quant para Kimi K2, Kimi K2 Thinking, DeepSeek-R1-0528 ou Qwen3-235B nos cards da Unsloth. O KLD do Dynamic 3.0 está só em gráficos.
- A Unsloth lista dois tamanhos para o R1 original (tabela vs. listagem de arquivos), com 7–15 GB de diferença; não está claro qual é GB decimal e qual é GiB.

## 2. Tipos de quant do llama.cpp abaixo de 4 bits e os números do ubergarm (ik_llama.cpp)

### Takeaway
No Kimi-K2-Instruct, com atenção, camada densa e expert compartilhado em q8_0, a perplexidade sobe ×1,003 a 4,64 bpw, ×1,024 a 3,60, ×1,078 a 2,92 (down em iq3_ks, gate/up em iq2_kl), ×1,155 a 2,76 (tudo em iq2_kl), ×1,25 a 2,43 e ×1,35–1,43 a 1,84–1,96 bpw. No DeepSeek-R1-0528, 2,8 bpw custa ×1,09 e 1,66 bpw custa ×1,52.

### Cited Findings

**Bits nominais dos tipos**
- IQ1_S 1,56 bpw, IQ2_XXS 2,06, IQ2_XS 2,31 (texto de ajuda do `quantize`); IQ1_M = 1,75 bpw, "halfway between IQ1_S and IQ2_XXS", PR #6302 de ikawrakow — [llama.cpp PR #6302](https://github.com/ggerganov/llama.cpp/pull/6302)
- TQ1_0 = 1,6875 bpw (5 trits por byte, 3^5 = 243 < 256) e TQ2_0 = 2,0625 bpw, PR #8151 de compilade, feitos para modelos ternários (TriLM/BitNet) — [llama.cpp PR #8151](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/8151/overview)
- bpw efetivo sobre um modelo inteiro (Llama-3.1-8B, inclui embeddings/output): IQ1_S 2,00, IQ1_M 2,15, IQ2_XXS 2,38, IQ2_XS 2,59, IQ2_S 2,74, IQ2_M 2,93, IQ3_XXS 3,25, IQ3_XS 3,50, IQ3_S 3,66, IQ3_M 3,76, Q2_K_S 2,97, Q2_K 3,16, Q3_K_S 3,64, Q4_K_M 4,89; o README não traz ppl/KLD — [llama.cpp tools/quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- Os nomes da Unsloth são os mesmos: "1,56-bit", "2,06-bit", "2,5-bit", "3,5-bit" nos experts do R1 — [unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)

**ubergarm, Kimi-K2-Instruct (1T), ik_llama.cpp com imatrix; referência Q8_0 = 1016,6 GiB, 8,504 bpw, PPL 2,9507 ± 0,0147 (wiki.test)**

| Quant | GiB (≈GB) | bpw | PPL | PPL / Q8_0 | ffn_down_exps | ffn_(gate\|up)_exps |
|---|---|---|---|---|---|---|
| IQ4_KS | 554,4 (595) | 4,638 | 2,9584 | ×1,003 | iq5_ks | iq4_ks |
| IQ3_KS | 430,9 (463) | 3,604 | 3,0226 | ×1,024 | iq4_ks | iq3_ks |
| IQ2_KL | 349,4 (375) | 2,923 | 3,1813 | ×1,078 | iq3_ks | iq2_kl |
| smol-IQ2_KL | 329,7 (354) | 2,758 | 3,4086 | ×1,155 | iq2_kl | iq2_kl |
| IQ2_KS | 290,3 (312) | 2,429 | 3,6827 | ×1,248 | iq2_kl | iq2_ks |
| IQ1_KT | 234,1 (251) | 1,959 | 3,9734 | ×1,347 | iq2_kt | iq1_kt |
| smol-IQ1_KT | 219,4 (236) | 1,835 | 4,2187 | ×1,430 | iq1_kt | iq1_kt |

- Tabela acima; em todos os sete quants, atenção (`attn_kv_b`, `attn_k_b`, demais `attn_*`), a primeira camada densa (`ffn_down/gate/up`) e os experts compartilhados ficam em q8_0; `token_embd` iq4_k–iq6_k, `output` iq6_k; o card não dá KLD — [ubergarm/Kimi-K2-Instruct-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF)

**ubergarm, Kimi-K2-Thinking (1T, fonte INT4 QAT); referência Q4_X = 543,6 GiB (584 GB), 4,549 bpw, PPL 2,0818 ± 0,0090**

| Quant | GiB (≈GB) | bpw | PPL | PPL / Q4_X | experts roteados |
|---|---|---|---|---|---|
| Q4_X | 543,6 (584) | 4,549 | 2,0818 | ×1,000 | q4_0 com patch (aproxima o QAT) |
| smol-IQ4_KSS | 485,0 (521) | 4,059 | 2,1343 | ×1,025 | iq4_kss |
| IQ3_K | 459,4 (493) | 3,845 | 2,1456 | ×1,031 | down q4_0, gate/up iq3_k |
| smol-IQ3_KS | 388,3 (417) | 3,249 | 2,2363 | ×1,074 | iq3_ks |
| IQ2_KL | 348,9 (375) | 2,920 | 2,3735 | ×1,140 | down iq3_ks, gate/up iq2_kl |
| smol-IQ2_KL | 329,2 (354) | 2,755 | 2,4550 | ×1,179 | iq2_kl |
| smol-IQ2_KS | 270,1 (290) | 2,261 | 2,9361 | ×1,410 | iq2_ks |
| smol-IQ1_KT | 218,9 (235) | 1,832 | 3,5931 | ×1,726 | iq1_kt |

- Tabela acima; atenção, primeira camada densa e experts compartilhados em q8_0 em todos; o Q4_X "reaches perplexity comparable to a full 1TB Q8_0 test quant" e usa um patch de uma linha no q4_0 para "better fit the original QAT target quantization"; o autor suspeita que a Moonshot usou só 15 dos 16 valores de 4 bits (em discussão com llama.cpp e Moonshot); o Q4_X foi feito sem imatrix e serviu para gerar a imatrix dos demais — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)

**ubergarm, DeepSeek-R1-0528 (671B); referência Q8_0 = 666 GiB, PPL 3,2130 ± 0,0170**

| Quant | GiB (≈GB) | bpw | PPL | PPL / Q8_0 | ffn_down_exps | ffn_(gate\|up)_exps | não-experts |
|---|---|---|---|---|---|---|---|
| IQ4_KS_R4 | 368 (395) | 4,701 | 3,2286 | ×1,005 | iq5_ks_r4 | iq4_ks_r4 | q8_0 |
| IQ3_K_R4 | 301 (323) | 3,847 | 3,2730 | ×1,019 | iq4_ks_r4 | iq3_k_r4 | q8_0 |
| IQ3_KS | 281,5 (302) | 3,598 | 3,2983 | ×1,027 | iq4_ks | iq3_ks | iq5_ks/iq4_ks; attn_k_b q5_0 |
| IQ2_K_R4 | 220 (236) | 2,799 | 3,5069 | ×1,091 | iq3_k_r4 | iq2_k_r4 | atenção iq5_ks; attn_k_b q5_0; compartilhado iq5_ks/iq4_ks |
| IQ1_S_R4 | 130,2 (140) | 1,664 | 4,8805 | ×1,519 | iq1_m_r4 | iq1_s_r4 | atenção/compartilhado/3 densas iq4_ks; attn_k_b q4_0 |

- Tabela acima; o KLD aparece só em gráfico (sem valores no texto) — [ubergarm/DeepSeek-R1-0528-GGUF](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- Confirmação em fórum (mesmo autor): Q8_0 3,2130; IQ3_K_R4 3,2730 (301 GiB); IQ2_K_R4 3,5069 (220 GiB); IQ4_KS_R4 (368 GiB) "very close to full quality"; esses GGUFs não rodam no llama.cpp normal por terem tensores MLA extras — [Level1Techs, DeepSeek deep dive](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/213)

### Inferences
- Comparação direta com o `--lo-all` do projeto (codebook de 2,5 bits em todos os experts, ppl ×1,14): o smol-IQ2_KL do ubergarm (2,755 bpw total, todos os experts em iq2_kl, ~2,7 bits/peso nos experts) custa ×1,155 no Kimi-K2-Instruct e ×1,179 no K2-Thinking. O codebook do projeto está na mesma faixa de custo que o estado da arte do llama.cpp em bits parecidos; não há "almoço grátis" óbvio a copiar.
- O salto de smol-IQ2_KL (×1,155) para IQ2_KL (×1,078) custa só +0,17 bpw e vem inteiro de subir o `ffn_down_exps` de iq2_kl para iq3_ks. É o ganho por bit mais barato de toda a tabela; vale replicar no projeto (down em q3/q4, gate/up em codebook).
- Abaixo de ~2,4 bpw, a ppl do Kimi sobe rápido (×1,25 a 2,43; ×1,35–1,43 abaixo de 2). Os quants de 1,6–2,0 bpw são "roda", não "roda bem".
- No Kimi-K2-Thinking (QAT INT4), os quants 2-bit do ubergarm custam mais (×1,14/×1,18/×1,41) que no K2-Instruct (×1,08/×1,16/×1,25) para os mesmos bpw. Hipótese: requantizar um modelo já em INT4 para 2 bits acumula erro, como o "re-quant floor" do projeto; não há fonte que isole isso.

### Gaps
- Não achei tabela numérica de KLD do ubergarm (está em imagens) nem quants IQ1_S/IQ2 do bartowski com ppl para R1/Kimi.
- Os bpw do ubergarm são sobre o modelo inteiro com atenção em q8_0; o bpw só dos experts não é publicado (dá para estimar, mas não fiz).

## 3. Kimi K2 Thinking: INT4 QAT nativo

### Takeaway
A Moonshot aplicou quantização INT4 só de pesos aos "componentes MoE" com QAT na fase de pós-treino; o checkpoint tem 594 GB (contra 1,03 TB do K2 em FP8), todos os benchmarks são reportados em INT4 e a empresa fala em ~2× de velocidade "lossless". Não encontrei nada da Moonshot abaixo de 4 bits nem o tamanho de bloco.

### Cited Findings
- "INT4 weight-only quantization applied to the MoE components", QAT "during the post-training phase", "roughly 2x generation speed improvement", "All benchmark results are reported under INT4 precision" (HLE 44,9 com ferramentas, AIME25 99,1 com Python, GPQA 84,5, SWE-bench Verified 71,3); formato compressed-tensors; não diz se a atenção foi quantizada nem o tamanho de bloco — [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- 594 GB no Hugging Face vs. 1,03 TB do Kimi K2; "quantization aware trained (qat) at int4"; Awni Hannun rodou em 2 M3 Ultra de 512 GB (1 TB) com mlx-lm, ~15 tok/s com pipeline-parallelism, "no loss in quality"; a versão MLX da comunidade tem 658 GB — [Simon Willison, Kimi K2 Thinking](https://simonwillison.net/2025/Nov/6/kimi-k2-thinking/)
- ubergarm: o Q4_X (q4_0 com patch) aproxima o QAT e dá PPL comparável ao Q8_0 de 1 TB; suspeita de que só 15 dos 16 níveis foram usados — [ubergarm/Kimi-K2-Thinking-GGUF](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF)

### Inferences
- Um checkpoint QAT em 4 bits é o piso prático hoje: ninguém publicou QAT abaixo de 4 bits para modelos desta classe. Tudo abaixo disso (ubergarm 2 bits, Unsloth 1,58) é PTQ sobre o INT4/FP8.
- O "lossless" é afirmação da Moonshot sem comparação a BF16 no card; só o PPL do ubergarm (Q4_X ≈ Q8_0) corrobora de forma independente.

### Gaps
- Tamanho de bloco, se atenção/compartilhados ficam em BF16, e o esquema de QAT (STE? distilação?) não estão no card nem nas notícias que abriram.
- Não há QAT publicado a 2–3 bits para Kimi, DeepSeek ou Qwen3-235B.

## 4. Papers de quantização MoE: QuantMoE-Bench, MxMoE, MoEQuant, EAQuant, MiLo, QMoE

### Takeaway
Os papers são todos em modelos pequenos (Mixtral-8x7B, DeepSeek-MoE-16B, DeepSeek-V2-Lite, Qwen1.5/2-MoE, SwitchTransformer). O consenso: atenção e experts compartilhados precisam de mais bits (+5 a +12 pontos de acurácia por bit médio no QuantMoE-Bench), as primeiras camadas MoE importam mais que as últimas, `down_proj` é mais sensível que gate/up, e alocar bits por frequência de ativação bate aleatório mas o ganho encolhe quando o orçamento é maior. O QMoE chega a 0,8 bit/param, mas num SwitchTransformer e com formato próprio.

### Cited Findings

**QuantMoE-Bench ("Examining PTQ for MoE", arXiv 2406.08155) — Mixtral-8x7B e DeepSeek-MoE-16B-base, média de 6 tarefas (WinoGrande, COPA, OBQA, HellaSwag, PIQA, MMLU)**

| Modelo | 16-bit | GPTQ 4-bit | GPTQ 3-bit | GPTQ 2-bit |
|---|---|---|---|---|
| Mixtral-8x7B | 75,33 | 73,89 | 70,69 | 40,53 |
| DeepSeek-MoE-16B | 67,74 | 66,78 | 64,30 | 49,43 |

- Precisão mista: Mixtral com atenção a 4 bits (2,06 bits médios) 46,00 (+5,47 sobre GPTQ 2-bit); Mixtral atenção + frequência + primeiras camadas (3,51 bits) 72,83; DeepSeek atenção + experts compartilhados (2,12 bits) 61,55 (+12,12); DeepSeek atenção + compartilhados + frequência + primeiras camadas (3,06 bits) 65,35 vs 64,30 do GPTQ 3-bit uniforme — [QuantMoE-Bench HTML](https://arxiv.org/html/2406.08155v2)
- Primeiras vs. últimas camadas MoE a 4 bits (resto a 2): Mixtral 4 blocos 51,12 (primeiros) vs 45,54 (últimos); 8 blocos 58,16 vs 47,36; DeepSeek 4 blocos 59,12 vs 58,06; 8 blocos 60,36 vs 59,38 — [mesma fonte](https://arxiv.org/html/2406.08155v2)
- Frequência vs. aleatório: Mixtral top-2 experts a mais bits (2,54 bits médios) 54,20 por frequência vs 49,10 ± 7,73 aleatório; top-4 (3,03 bits) 63,82 vs 63,70 ± 0,49; DeepSeek top-25 (2,97 bits) 64,66 vs 63,83 ± 0,04 — [mesma fonte](https://arxiv.org/html/2406.08155v2)
- Scorer de outliers por camada linear (25% das lineares FFN a 4 bits, 2,54 bits): Mixtral 56,20 vs 55,25 aleatório; DeepSeek 62,43 vs 59,61 — [mesma fonte](https://arxiv.org/html/2406.08155v2)
- Preditor de importância de bloco MoE (DeepSeek): 2,29 bits: aleatório 58,03, primeiros-k 59,12, predito 60,34; 2,92 bits: 60,97 / 63,31 / 63,71 — [mesma fonte](https://arxiv.org/html/2406.08155v2)
- Abstract: "shared experts ... potentially needing higher precision"; precisão mista fina 65,35% vs 64,30% GPTQ — [arXiv 2406.08155](https://arxiv.org/abs/2406.08155)

**MxMoE (arXiv 2505.05799) — Mixtral-8x7B-Instruct, Qwen1.5-MoE (60+4 experts), Qwen2-MoE-Instruct (64+8), DeepSeek-V2-Lite (64+2, top-6)**
- Bits por bloco linear (gate/up/down), não por expert; sensibilidade = distância euclidiana da saída do bloco MoE com 128 sequências WikiText-2; frequência de ativação + runtime por tile num ILP que minimiza L^r·T^(1−r) sob orçamento de memória; pesos com rotação de Hadamard + GPTQ — [MxMoE HTML](https://arxiv.org/html/2505.05799v1)
- "Within one expert, down_proj needs higher precision than gate_proj"; a sensibilidade varia entre experts (ex.: expert 40 degrada mais que o 37); frequências de ativação dentro de um bloco variam >10× — [mesma fonte](https://arxiv.org/html/2505.05799v1)

| Modelo (ppl WikiText-2) | BF16 | GPTQ 3,25 bit | GPTQ 2,25 bit | MxMoE 3,25 | MxMoE 2,25 |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 5,92 | 6,18 | 8,49 | 6,08 | 7,01 |
| Qwen1.5-MoE | 6,79 | 7,15 | 11,19 | 7,02 | 8,79 |
| Qwen2-MoE | 5,84 | 6,11 | 7,98 | 6,18 | 7,57 |
| Mixtral-8x7B | 3,88 | 4,17 | 5,69 | 4,15 | 5,63 |

- Tabela acima; DeepSeek-V2-Lite acurácia média (7 tarefas): 70,68 BF16, 58,04 GPTQ 2,25, 62,04 MxMoE 2,25, 69,37 MxMoE 3,25; os "2,25/3,25" já incluem o overhead das escalas por grupo — [mesma fonte](https://arxiv.org/html/2505.05799v1); abstract: ppl 2,4 menor que GPTQ a 2,25 bits, até 3,4× de speedup — [arXiv 2505.05799](https://arxiv.org/abs/2505.05799)

**MoEQuant (arXiv 2505.03804)**
- PTQ com Expert-Balanced Self-Sampling (calibração balanceada entre experts) e Affinity-Guided Quantization (peso de cada amostra por expert pela afinidade do roteador); ">10 points accuracy gain in HumanEval for DeepSeekMoE-16B under 4-bit"; o abstract não dá números abaixo de 4 bits — [arXiv 2505.03804](https://arxiv.org/abs/2505.03804)

**EAQuant (arXiv 2506.13329)**
- Três componentes: smoothing agregado para outliers, alinhamento de consistência de roteamento (manter a seleção de experts estável após quantizar) e balanceamento da calibração por expert; ganhos médios de 1,15% a 13,81% em W4A4, W3A4, W3A3 e W2A4, em três arquiteturas MoE não nomeadas no abstract — [arXiv 2506.13329](https://arxiv.org/abs/2506.13329)

**MiLo (arXiv 2504.02658)**
- Compensadores de baixo posto ("mixture of low-rank compensators") sobre MoE a 3 bits, seleção adaptativa de posto, sem dados de calibração, kernels 3-bit para Tensor Cores; o abstract não dá ppl nem modelos — [arXiv 2504.02658](https://arxiv.org/abs/2504.02658)

**QMoE (Frantar & Alistarh, arXiv 2310.16795)**
- SwitchTransformer-c2048 (1,6T params, 3,2 TB em bf16) comprimido para <160 GB, 0,8 bit/param (~20×), "minor accuracy loss", kernels de decodificação com <5% de overhead; roda em 4× A6000 ou 8× 3090; compressão em <1 dia numa GPU — [arXiv 2310.16795](https://arxiv.org/abs/2310.16795)

### Inferences
- Nenhum paper quantiza um MoE de 235B+ abaixo de 4 bits com números; o maior é Qwen2-MoE (~57B) e Qwen3-80B (DynaExq, seção 5). Os resultados comunitários (seção 2) são hoje a única evidência em escala.
- O QMoE é um ponto fora da curva por causa do modelo: o SwitchTransformer-c2048 tem 2048 experts por camada e é notoriamente redundante; 0,8 bit/param não se transfere a Kimi/DeepSeek, cujos quants a 1,6–1,8 bpw custam ×1,4–1,5 de ppl.
- A ordem de prioridade dos papers coincide com a dos quants comunitários: atenção > experts compartilhados > primeiras camadas > down_proj > gate/up > experts raros. O projeto já guarda atenção em q4/q8 (model.safetensors); o próximo lugar barato é o `down_proj`.

### Gaps
- Números de MiLo e MoEQuant em modelos ≥100B não existem; o abstract de MiLo não nomeia modelos.
- Não encontrei nenhum paper que meça a sensibilidade por "posição do pick" (1º vs 8º) no top-k do roteador em modelos grandes.

## 5. Alocação de bits por frequência ou por peso do roteador; precisão escolhida em tempo de execução

### Takeaway
Alocação estática por frequência é o baseline em todos os papers de 2024–2026 e já perde para scorers de sensibilidade (QuantMoE-Bench, MODE, GEMQ). A ideia das "duas resoluções" (precisão escolhida pelo score do roteador por token) aparece em 2025–2026 com os nomes "router-guided precision restoration" (arXiv 2512.17073, com compensadores de baixo posto nos top-n) e "dynamic expert quantization" / DynaExq (arXiv 2511.15015, experts quentes em mais bits por traço de roteamento). Não achei um nome consolidado para a regra "pick de maior peso → mais bits".

### Cited Findings
- QuantMoE-Bench: frequência bate aleatório em todos os orçamentos, mas a margem vai de +5,1 (Mixtral top-2, 2,54 bits) a +0,1 (top-4, 3,03 bits) — [QuantMoE-Bench HTML](https://arxiv.org/html/2406.08155v2)
- MxMoE: a frequência entra no custo de tempo (ILP), e a sensibilidade por bloco linear decide os bits — [MxMoE HTML](https://arxiv.org/html/2505.05799v1)
- GEMQ (arXiv 2605.23078): critica a alocação local por camada e usa um programa linear global para expert-level mixed precision — [GEMQ](https://arxiv.org/pdf/2605.23078)
- MODE (arXiv 2606.17118): "frequency reflects how often an expert is selected, which does not fully align with how much its quantization degrades model performance"; combina frequência com sensibilidade num ILP — [MODE](https://arxiv.org/html/2606.17118v1)
- arXiv 2604.06515: ordena experts pela variação da norma L2 do roteador durante ajuste fino e reporta que isso supera frequência de ativação e pesos de ativação; variante com normas do roteador pré-treinado evita o ajuste — [Efficient Quantization of MoE with Generalization Guarantees](https://arxiv.org/html/2604.06515v1)
- Colla-Q (arXiv 2609.18131): "minimax precision balancing" para não deixar experts fracos degradarem — [Colla-Q](https://arxiv.org/pdf/2609.18131)
- "Bandwidth-Efficient Adaptive MoE via Low-Rank Compensation" (arXiv 2512.17073): "router-guided precision restoration": na inferência, transfere fatores de baixo posto só para os top-n (n<k) experts por token e aplica a compensação neles; os demais ficam em poucos bits; GPU e GPU-NDP; o abstract não dá bits nem ppl — [arXiv 2512.17073](https://arxiv.org/abs/2512.17073); o snippet de busca resume que os "top-n experts by router score carry larger quantization errors" — [busca](https://arxiv.org/pdf/2512.17073)
- DynaExq, "Dynamic Expert Quantization for Scalable MoE Inference" (arXiv 2511.15015): experts "quentes" por traço longo do roteador ficam em mais bits, frios em poucos bits, por camada sob orçamento de HBM, promoções/demoções assíncronas; Qwen3-80B sobe de 73,09% (PTQ estática) para 77,57% sob o mesmo orçamento; até 2,73× de throughput sobre offload/prefetch com batch 32; bits exatos não informados — [arXiv 2511.15015](https://arxiv.org/abs/2511.15015)
- Snippet de busca cita "PMQ" (frequência + estatísticas de peso em programação linear) — [busca, GEMQ](https://arxiv.org/pdf/2605.23078)

### Inferences
- O que o projeto faz (q4 nos picks 1–4, codebook nos 5–8, decidido por token pelo score) é "router-guided precision" por token, mais fino que DynaExq (que é por expert, por janela de tempo) e sem o custo de low-rank do 2512.17073. A evidência comunitária de que o `down_proj` é o tensor mais sensível sugere uma terceira dimensão para as duas resoluções: manter o `down_proj` sempre num bit a mais que gate/up, mesmo nos picks de menor peso.
- Alocação estática por frequência tende a render pouco em modelos com roteamento balanceado (DeepSeek/Kimi com `e_score_correction_bias` forçam balanceamento), o que condiz com o que o projeto já viu ao "fixar experts populares" não funcionar.

### Gaps
- O abstract do 2512.17073 não traz bits, ppl, nem modelos; o PDF não foi lido. Mesma coisa para GEMQ, MODE, Colla-Q e 2604.06515 (só snippets de busca).
- Não achei na literatura um estudo de "pick de maior peso tolera menos bits" com número; a única citação próxima é o snippet do 2512.17073.

## 6. DeepSeek-V3/R1 a 1,58–2,5 bits na prática

### Takeaway
A 2,5 bits médios (UD-Q2_K_XL, 212–255 GB) o R1/V3.1 é tido como "usável": Aider 65,8% vs 71,6% em BF16 no V3.1; ubergarm a 2,8 bpw custa ×1,09 de ppl no R1-0528. A 1,58–1,7 bits (131–140 GB) o modelo responde e passa testes de código simples, mas o ppl sobe ×1,5 (IQ1_S_R4) e o Aider cai para 50,7–55,7% (TQ1_0/IQ1_M no V3.1).

### Cited Findings
- Unsloth, R1 original: UD-IQ1_S 131 GB "Fair", UD-IQ1_M 158 GB "Good", UD-IQ2_XXS 183 GB "Better", UD-Q2_K_XL 212 GB "Best"; `--min-p 0.05` recomendado para o 1,58-bit — [unsloth/DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)
- Unsloth, V3.1 Aider: TQ1_0 (170 GB) 50,7%, IQ1_M (206 GB) 55,7%, IQ2_XXS (225 GB) 61,2%, IQ2_M (235 GB) 64,3%, Q2_K_XL (255 GB) 65,8%, IQ3_XXS (279 GB) 66,8%, BF16 71,6%; em modo pensante, 1-bit 57,8%, 2-bit 66,7%, 3-bit 75,6% vs 76,1% — [Unsloth, Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-ggufs-on-aider-polyglot)
- ubergarm, R1-0528: IQ2_K_R4 (220 GiB, 2,80 bpw) ppl 3,5069 vs 3,2130 Q8_0 (×1,09); IQ1_S_R4 (130 GiB, 1,66 bpw) 4,8805 (×1,52) — [ubergarm/DeepSeek-R1-0528-GGUF](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF)
- DeepSeek-TNG-R1T2-Chimera IQ1_S (132,9 GiB, 1,70 bpw): ppl "em torno de 4,0" para a mistura IQ2_XXS, e o IQ1_S "um pouco menos inteligente" — [Level1Techs](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/213)

### Inferences
- Em três fontes independentes (Unsloth/Aider, ubergarm/ppl no R1, ubergarm/ppl no Kimi), o joelho da curva fica entre 2,4 e 2,8 bpw total (~2,2–2,6 bits nos experts). Abaixo disso o custo por bit removido dobra.
- Para o projeto (cache de 6–16 GB, SSD como gargalo), a faixa que a comunidade valida é exatamente a do codebook de 2,5 bits; descer a ~1,7 bits nos experts de baixo peso só se o ganho de tok/s compensar ×1,3–1,5 de ppl, o que os números da seção 2 desaconselham.

### Gaps
- Nenhuma avaliação de MMLU/GPQA por quant abaixo de 4 bits para R1/V3/Kimi além do Aider da Unsloth; os relatos de comunidade são de ppl ou impressão qualitativa.
- Não localizei um teste reproduzível (com número) do "Flappy Bird" da Unsloth; o blog original estava inacessível (403).
