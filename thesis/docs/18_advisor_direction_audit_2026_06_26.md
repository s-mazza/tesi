# Advisor Direction Audit - 2026-06-26

Purpose: check the broad direction requested by the advisor, not only the
line-level Overleaf comments.

Source: `thesis/fixes.txt`, especially the final Teams message about method
writing quality, narrative clarity, method positioning, and thesis readability.

## High-Level Requests

| Advisor direction | Thesis response | Status |
|---|---|---|
| Prioritize the method chapter before polishing results/related work. | Chapter 3 was rewritten around the actual proposed method: Semantic-Fidelity Corpus, judge task, Latent-GEPA, feedback variants, and soft-prompt diagnostic. | Resolved locally |
| Stop framing the thesis as a list of things it does not do. | Defensive paragraphs in Chapter 3 were replaced with positive method statements. Remaining boundary statements are mostly in results/limitations, where they support interpretation. | Resolved locally |
| Avoid the "branch" framing. | Chapter 3 now uses components, diagnostics, feedback variants, and conditions. A remaining "Branch is diagnostic" in Chapter 5 was replaced with a neutral diagnostic caveat. | Resolved locally |
| Make the method formal and descriptive. | The method now defines input objects, feedback signals, model roles, selection boundary, and artifact requirements. GEPA is described as a validation-driven prompt search, not an informal set of runs. | Resolved locally |
| Keep SIPIT mainly in related work. | SIPIT is explained in related work. In Chapter 3 it is only referenced as a readout-style tool for the soft-prompt diagnostic, because that experiment needs to say what readout is applied. | Resolved locally |
| Treat NLA as a tool, not as a thesis method on the same level as Latent-GEPA. | NLA technical details remain in related/setup/results. Chapter 3 describes only how NLA verbalizations become proposer feedback inside Latent-GEPA. | Resolved locally |
| Put GEPA/Latent-GEPA at the center of the method. | Chapter 3 now has a dedicated `Latent-GEPA Prompt Optimization Loop` section with seed prompt, minibatch evaluation, reflection data, proposer, candidate pool, Pareto-style frontier, and final-test boundary. | Resolved locally |
| Explain GEPA mini-batch proposal and Pareto-style selection. | Both are now in the method prose and pseudocode. The text explains why the prompt trajectory matters and why the final prompt is selected from validation behavior. | Resolved locally |
| Add the soft-prompting plus SIPIT analysis if missing. | Chapter 3 contains a concise soft-prompt diagnostic method. Chapter 5 contains task deltas, SIPIT-style readout, and controls separating discrete-token SIPIT from continuous soft-prompt targets. | Resolved locally |
| Remove Python/file/path references from thesis prose. | Method/results prose was rewritten to avoid local script/cache path references. Provenance is described methodologically instead. | Resolved locally |
| Make the thesis clearer even if it becomes shorter. | Some implementation-history material was moved from method to setup/results, or archived if it was useful but distracting. The current chapter organization is more reader-facing. | Resolved locally |

## Conceptual Check

The advisor's main point was not only that some sections had bad titles. The
underlying issue was that the thesis looked like a chronology of experiments
instead of a clean method. The current version fixes that by making
`Latent-GEPA` the main proposed pipeline and by treating the other components as
either:

- prior methods used as diagnostic tools;
- feedback sources for Latent-GEPA;
- experimental controls needed to interpret the results.

This distinction is important because it prevents the reader from thinking that
SIPIT, NLA, embedding inversion, soft prompting, and GEPA are five independent
methods proposed by the thesis. The thesis contribution is instead framed as:

1. build a semantic-fidelity evaluation surface;
2. use inversion/verbalization/readout methods to inspect semantic preservation;
3. use latent-derived signals as feedback for GEPA through `Latent-GEPA`;
4. evaluate which signals actually improve or diagnose LLM-as-a-judge prompt
   optimization.

## Remaining Deliberate Choices

- GEPA is still mentioned in related work because it is prior work and belongs
  in the prompt-optimization literature. The detailed algorithmic explanation
  is now in Chapter 3.
- NLA still appears in Chapter 3, but only as `NLA feedback for Latent-GEPA`,
  not as a standalone method section.
- SIPIT appears in Chapter 3 only where the soft-prompt diagnostic needs to name
  the readout tool. The SIPIT method itself is not re-explained there.
- Negative and inconclusive results remain in Chapter 5 and Chapter 6 because
  they are needed for scientific interpretation. They are not used as the
  opening framing of Chapter 3.

## Additional Fixes Made During This Audit

- Rewrote the method readout section from "these diagnostics do not introduce
  new algorithms" to a positive description of prior methods used as measurement
  tools.
- Rewrote the SIPIT related-work bridge from "not satisfied with exactness" to
  "adds a semantic-fidelity perspective".
- Rewrote the Latent-GEPA objective sentence so it says that latent feedback
  keeps GEPA's objective intact.
- Rewrote the feedback-variant paragraph so it explains the ablation map instead
  of saying that variants do not imply usefulness.
- Rewrote the G-EVAL boundary paragraph in related work as a positive
  distinction rather than "this thesis does not treat...".
- Added captions and labels to the two long Chapter 3 listing examples.
- Removed the last `Branch is diagnostic` wording from the final claim matrix.

## Suggested Advisor Message

```text
ho ricontrollato anche il messaggio generale, non solo i commenti puntuali. Ho cercato di sistemare proprio la direzione del metodo: ora non e' piu una cronologia di esperimenti/branch, ma parte dal corpus, definisce il judge task e poi mette LatentGEPA come metodo principale. SIPIT e NLA restano come strumenti/prior work, GEPA invece e' il backbone del metodo e viene spiegato li con minibatch proposal, reflection data, proposer e Pareto-style selection. Ho anche tolto varie frasi in tono "questa tesi non fa..." e le ho riscritte in modo piu formale/positivo.
```
