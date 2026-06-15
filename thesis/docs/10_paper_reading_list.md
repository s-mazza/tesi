# Paper Reading List

This document is the working inventory of papers and research artifacts that
can be useful for the thesis. It is intentionally more concrete than
`04_related_work_map.md`: each entry records why the item matters, where it
fits in the thesis, and whether a local copy already exists.

## How To Use This List

- Use this file while planning Chapter 1 and Chapter 2.
- Keep `04_related_work_map.md` as the conceptual map of the related work.
- Before writing the LaTeX bibliography, fetch BibTeX from DBLP where
  available, then ACL Anthology/arXiv when DBLP is missing.
- Treat software repositories and notebooks as implementation references. Cite
  the corresponding paper when possible, and put repository links in footnotes
  only if the code itself is relevant.
- Do not cite local PDFs directly. Use the official paper URL in the thesis and
  keep the local path here only for navigation.

## Advisor-Provided Items

| Item | Local copy | Thesis group | Why it matters |
| --- | --- | --- | --- |
| [Steered LLM Activations are Non-Surjective](https://arxiv.org/abs/2604.09839) | none found | Natural Language Activations / interpretability caveats | Shows that white-box activation steering can move residual streams outside the set reachable by text prompts. Useful as a warning against assuming every activation-level behavior has a faithful prompt-level explanation. |
| [Negation Neglect: When models fail to learn negations in training](https://arxiv.org/abs/2605.13829) | none found | Logical semantics / negation motivation | Directly supports the thesis motivation that negation can be represented or learned incorrectly even when text appears explicit. Useful for the broader argument around logical faithfulness. |
| [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2510.15511) | `spit/2510.15511v4.pdf` | Hidden-state inversion / SIPIT | Core SIPIT paper. Establishes injectivity of decoder-only LM hidden representations and introduces exact prompt reconstruction from hidden activations. |
| [SIPIT repository](https://github.com/giorgosnikolaou/SIPIT) | `spit/SIPIT/` | Hidden-state inversion / SIPIT | Implementation reference for the SIPIT experiments, baselines, dataset creation scripts, and exact inversion claims. |
| [Prompt Waywardness](https://aclanthology.org/2022.naacl-main.266/) | `2022.naacl-main.266.pdf` | Prompt Waywardness / soft-prompt interpretability | Shows that discrete nearest-neighbor interpretations of continuous prompts can be arbitrary or contradictory while preserving task performance. Useful for framing why textual interpretations of continuous objects require careful validation. |
| [Towards Interpretable Soft Prompts](https://arxiv.org/abs/2504.02144) | `2504.02144v1.pdf` | Prompt Waywardness / soft-prompt interpretability | Defines faithfulness and scrutability criteria for trainable prompts and documents a tradeoff between interpretability and task performance. |
| [TowardsInterpretablePrompts T5 notebook](https://github.com/NikhilNayak-debug/towards_interpretable_softprompts/blob/main/TowardsInterpretablePrompts_T5.ipynb) | `towards_interpretable_softprompts/TowardsInterpretablePrompts_T5.ipynb` | Prompt Waywardness / implementation reference | Notebook reference for discrete prompt optimization with T5 and a perplexity-style regularization term over prompt tokens. Useful because earlier proposer anti-overfit changes were inspired by this line of work. |
| [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) | `natural_language_autoencoders/` | Natural Language Activations | Core NLA reference. Defines AV as activation-to-text and AR as text-to-activation. The released Qwen2.5-7B layer-20 checkpoints match the current base-model family used in experiments. |
| [Do Sparse Autoencoders Capture Concept Manifolds?](https://arxiv.org/abs/2604.28119) | none found | Natural Language Activations / interpretability caveats | Useful background for the risk that representation-level concepts may not be isolated linear directions. It helps contextualize why activation verbalizations may mix multiple geometric factors. |
| [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) | none found | GEPA / prompt optimization | Core optimizer paper. GEPA uses natural-language reflection over trajectories to propose prompt updates and select candidates through a Pareto-style prompt population. |
| [G-EVAL](https://aclanthology.org/2023.emnlp-main.153/) | `2023.emnlp-main.153.pdf` | G-EVAL / LLM-as-a-judge | Core evaluation paper. Provides the benchmark framing, dimensions, datasets, and human-correlation metrics that the GEPA experiments are trying to compare against or reproduce. |

## Local Downloaded Papers

| Item | Local copy | Thesis group | Why it matters |
| --- | --- | --- | --- |
| Embedding Inversion via Conditional Masked Diffusion Language Models | `embedding-inversion-demo/2602.11047v3.pdf` | Embedding inversion / Jina reproduction | Early reproduction target for the embedding-inversion part of the thesis. Frames inversion as conditional masked diffusion with parallel denoising rather than autoregressive correction. |
| Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack | `2305.03010v1.pdf` | Embedding inversion | GEIA paper. Treats sentence-embedding inversion as a generation problem and reconstructs ordered text, not just unordered keywords. |
| Universal Zero-shot Embedding Inversion | `2504.00147v1.pdf` | Embedding inversion | ZSinvert paper. Zero-shot, query-efficient inversion method that avoids training a separate model per embedding. Useful as a contrast to trained inversion models. |
| Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings | `2602.01757v2.pdf` | Embedding inversion | Training-free cross-domain inversion using LLM priors plus online alignment. Relevant for privacy background and for contrasting training-free inversion with SIPIT/NLA. |
| This is not a Dataset: A Large Negation Benchmark to Challenge Large Language Models | `thisisnotdataset_paper.pdf` | Logical semantics / negation dataset | Local paper for the negation benchmark used in the earlier logical dataset work. Useful for motivating why standard similarity metrics can miss negation failures. |
| MIBench: A Comprehensive Benchmark for Model Inversion Attack and Defense | `3143_MIBench_A_Comprehensive_B.pdf` | Model inversion background | Broader model-inversion benchmark, mostly useful for privacy/evaluation background. Less central because it is not focused on text hidden-state inversion. |
| Higher Embedding Dimension Creates a Stronger World Model for a Simple Sorting Task | `2510.18315v1.pdf` | Mechanistic interpretability background | Tangential local paper on embedding dimension and interpretable internal world models in a sorting task. Keep as low-priority background unless the thesis needs a short note on representation capacity. |

## Duplicate Local Copies

These copies are content duplicates of canonical local PDFs and should not be
listed twice in the bibliography:

| Canonical local copy | Duplicate copy |
| --- | --- |
| `2022.naacl-main.266.pdf` | `prompt-waywardness/2022.naacl-main.266.pdf` |
| `2023.emnlp-main.153.pdf` | `gepa-experiments/2023.emnlp-main.153.pdf` |
| `2504.02144v1.pdf` | `towards_interpretable_softprompts/2504.02144v1.pdf` |
| `thisisnotdataset_paper.pdf` | `embedding-inversion-demo/thisisnotdataset_paper.pdf` |

## Local PDFs Excluded From This List

The PDFs under `tesi_t_simoneMazzacano/` are previous-thesis reference
material and figure assets, not new literature items for this thesis. Keep them
available for structure, style, and LaTeX conventions, but do not merge them
into the paper reading list unless a specific cited source is extracted from
that thesis.

## Immediate Bibliography Tasks

- Fetch BibTeX for the core set: SIPIT, NLA, GEPA, G-EVAL, Prompt
  Waywardness, Towards Interpretable Soft Prompts, and the main embedding
  inversion papers.
- Decide which local-only or low-priority papers deserve actual thesis
  citations versus being kept only as background reading.
- For G-EVAL, record the exact datasets, dimensions, and metrics in
  `06_experimental_setup_spec.md` once the final experimental matrix is frozen.
- For NLA, record the exact checkpoint, layer, token-selection strategy, and
  verbalization format in `05_method_spec.md` after the current experimental
  branch stabilizes.
