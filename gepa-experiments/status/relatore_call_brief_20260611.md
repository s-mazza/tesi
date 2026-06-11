# Briefing call relatore - GEPA / G-Eval / NLA

Data: 2026-06-11

Scopo: spiegare lo stato della repo e degli esperimenti dal punto in cui il
relatore era rimasto, circa commit `9997e3a`, fino allo stato corrente.

## Messaggio principale

Prima avevamo una pipeline Topical-Chat engagingness con GEPA, Qwen2.5-7B come
judge/base model, Qwen 35B via llama.cpp come proposer, e feedback di
perplexity. La run PPL long era positiva rispetto alla baseline interna:

```text
Run: gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer
Metriche final-test, n=60

baseline:  pearson=0.551831  spearman=0.547848  agreement=0.761111  mae=0.477778
optimized: pearson=0.632812  spearman=0.619893  agreement=0.788889  mae=0.422222
```

Da quel punto sono state aggiunte tre cose principali:

- supporto piu generale a G-Eval: Topical-Chat, SummEval, QAGS-CNN, QAGS-XSUM;
- metriche paper-aligned: Pearson/Spearman/Kendall tau, oltre a MAE/agreement come diagnostiche;
- NLA come feedback aggiuntivo a GEPA, piu esperimenti diagnostici per capire perche aiuti o peggiori.

La conclusione attuale e sospesa: NLA ha dato un segnale positivo su smoke
Topical-Chat con Qwen35B proposer, ma non abbiamo ancora una claim pulita sulla
long run perche manca il controllo PPL current-code completato (`11913587`).

## Timeline git da raccontare

Commit di partenza indicativo:

- `9997e3a Avoid fixed port collisions in long proposer run`: stato vicino alla prima run PPL che il relatore conosce.

Poi:

- `9ab28b9` / `62897b5`: introdotto task registry e runner generalizzato per piu dataset.
- `be986f0` / `ba1b0db`: aggiunto feedback NLA e precompute reale delle verbalizzazioni.
- `26caa67`: aggiunto auxiliary judge opzionale.
- `de2f561`: aggiunta metrica `kendall_tau`.
- `0d2ecd6`: aggiunta diagnostica NLA vs controllo.
- `93265a1`: migliorata token selection NLA dopo prima root-cause analysis.
- `7f4bcd5`: accettate verbalizzazioni NLA con tag parziali.
- `da63910`: impedito fallback scientificamente pericoloso a dry-run NLA.
- `4d08b14`: registrati risultati long fixed-NLA e bisogno del controllo PPL current-code.
- `907c0bb`: aggiunto report aggregato NLA e prima path per aux judge.
- `4164e0c`: registrato crash del controllo PPL per GPU gia occupata da altri processi.
- Dopo due altri tentativi falliti per GPU proposer non abbastanza libera (`11913482`, `11913557`), il replacement corrente e `11913587`.

## Stato esperimenti principali

### 1. PPL long nota al relatore

Cartella:

```text
gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer
```

Config chiave:

```text
dataset legacy Topical-Chat / label Engaging
train/val/test contexts: 40/10/10
seed: 42
judge: Qwen/Qwen2.5-7B-Instruct
proposer: Qwen35B via llama.cpp
perplexity_feedback: true
instruction_proposer: generalizing
```

Risultato:

```text
baseline:  pearson=0.551831  spearman=0.547848  agreement=0.761111  mae=0.477778
optimized: pearson=0.632812  spearman=0.619893  agreement=0.788889  mae=0.422222
```

Interpretazione: PPL ha dato feedback utile a GEPA e l'optimized prompt ha
migliorato le metriche sul final test.

### 2. Prima NLA long

Cartella:

```text
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b
```

Risultato:

```text
baseline:  pearson=0.723364  spearman=0.721559  agreement=0.769444  mae=0.461111
optimized: pearson=0.511144  spearman=0.490815  agreement=0.691667  mae=0.616667
runtime:   28355.7s, circa 7h52m
trajectory rows: 631
seed words: 192
optimized words: 339
```

Interpretazione: NLA peggiora molto rispetto al controllo PPL vecchio. Dopo
diagnostica, questa run non va letta come fallimento definitivo di NLA, ma come
test di una condizione NLA debole: token selection su primi token semantici e
verbalizzazioni ripetitive/generiche.

### 3. Fixed-NLA smoke Qwen35B

Cartella:

```text
gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_llamacpp35b_smoke
```

Controllo PPL smoke:

```text
gepa-experiments/results/geval_gepa_engaging_qwen25_ppl_llamacpp35b_smoke
```

Metriche optimized, n=12:

```text
PPL smoke:       pearson=0.536400  spearman=0.527410  kendall=0.459933  agreement=0.638889  mae=0.722222
Fixed-NLA smoke: pearson=0.674979  spearman=0.674693  kendall=0.606407  agreement=0.763889  mae=0.472222
```

Interpretazione: primo segnale positivo. Non basta per la tesi perche n=12, ma
giustifica una long run fixed-NLA.

### 4. Fixed-NLA long

Cartella:

```text
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b
```

Risultato:

```text
baseline:  pearson=0.681158  spearman=0.677076  kendall=0.571150  agreement=0.752778  mae=0.494444
optimized: pearson=0.681158  spearman=0.677076  kendall=0.571150  agreement=0.752778  mae=0.494444
runtime:   30519.012s, circa 8h28m
trajectory rows: 740
seed == optimized: true
```

Confronto con vecchia PPL long:

```text
fixed-NLA optimized minus old PPL optimized:
pearson   +0.048346
spearman  +0.057183
agreement -0.036111
mae       +0.072222  (peggio)
```

Interpretazione: non e una claim pulita che NLA migliori GEPA. Il seed prompt
era gia forte e GEPA non ha selezionato un prompt diverso. Serve il controllo
PPL current-code, cioe `11913587`, per capire se il vantaggio Pearson/Spearman
dipende davvero da NLA o da cambi di seed/codepath.

### 5. Candidate-only NLA

Cartelle:

```text
gepa-experiments/results/experimental_nla_candidate_content_6_topical_chat_smoke
gepa-experiments/results/experimental_nla_candidate_content_10_topical_chat_smoke
```

Candidate-content-10 optimized, n=12:

```text
pearson=0.402090
spearman=0.371727
kendall=0.310087
agreement=0.625000
mae=0.750000
```

Interpretazione: togliere source/reference e usare solo candidate tokens non
basta. Anche con 0% duplicati, il feedback resta poco metric-aligned e le
metriche peggiorano. Quindi il problema non e solo la duplicazione.

### 6. Dataset smoke

SummEval consistency e l'unico smoke dataset-level abbastanza leggibile:

```text
PPL optimized, n=32:
pearson=0.701281  spearman=0.790860  kendall=0.716853  agreement=0.723958  mae=1.104167

PPL+NLA optimized, n=32:
pearson=0.618512  spearman=0.696582  agreement=0.718750  mae=1.125000
```

Interpretazione: direzione negativa per NLA. QAGS-CNN e QAGS-XSUM sono solo
plumbing checks perche hanno final-test slice `n=2`, quindi non vanno usati per
claim scientifiche.

## Stato cluster attuale

```text
11913415: fallito dopo 53s prima di GEPA
causa: vLLM ha trovato solo 7.22 GiB liberi sulla GPU assegnata
origine: processi di altri utenti su faretra, non nostri container/zombie

11913482: replacement PPL long current-code, fallito in startup llama.cpp
causa: proposer GPU con circa 7.4 GiB liberi, ma Qwen35B richiedeva circa 20.6 GiB

11913557: secondo tentativo, stesso fallimento llama.cpp per proposer GPU non abbastanza libera

11913587: replacement PPL long current-code corrente
stato ultimo check: RUNNING
startup: llama.cpp proposer ready, vLLM ready, perplexity precompute avviato
```

Questo job e il pezzo mancante piu importante per la call se dovesse finire in
tempo. Se non finisce, dire chiaramente che la claim NLA long resta sospesa.

## Mappa cartelle da aprire in VS Code

### 1. Risultati e diagnostiche

```text
gepa-experiments/results
gepa-experiments/results/diagnostics
```

File piu utili:

```text
gepa-experiments/results/diagnostics/nla_evidence_deep_dive_20260611.md
gepa-experiments/results/diagnostics/nla_fixed_long_vs_ppl_long_20260611.md
gepa-experiments/results/diagnostics/nla_vs_ppl_fixed_smoke_20260610.md
gepa-experiments/results/diagnostics/nla_candidate_content_10_vs_ppl_smoke_20260611.md
gepa-experiments/results/diagnostics/nla_summeval_consistency_vs_ppl_smoke_20260611.md
```

### 2. Config

```text
gepa-experiments/config
```

File chiave:

```text
gepa-experiments/config/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer.env
gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b.env
gepa-experiments/config/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control.env
gepa-experiments/config/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke.env
```

Nota: la config aux judge va spiegata come esperimento separato/da rifinire,
perche far vedere NLA all'aux judge cambia la domanda sperimentale.

### 3. Codice

```text
gepa-experiments/geval_gepa
gepa-experiments/scripts
```

Code pointer:

```text
runner.py:77       create_metric_fn: costruisce score e feedback per GEPA
runner.py:131      NLA feedback entra nel feedback testuale
runner.py:268      configure_lms: separa judge/proposer
runner.py:497      seed_prompt da dataset/dimensione
runner.py:618      optimizer = GEPA(...); optimizer.compile(...)

prompts.py:36      seed_instructions
metrics.py:16      EvaluationMetrics include kendall_tau
metrics.py:101     Kendall tau-b
tasks.py:41        TaskSpec
tasks.py:377       registry Topical-Chat/SummEval/QAGS
nla_feedback.py:29 NlaFeedbackProvider
nla_precompute.py:77 token selection
nla_precompute.py:452 activation summary senza salvare raw vectors
aux_judge.py:29    AuxJudgeFeedbackProvider

scripts/diagnose_nla_run.py:331     report controllo vs NLA
scripts/analyze_nla_evidence.py:50  registry delle run nel report aggregato
```

## Spiegazione pipeline attuale

Schema:

```text
dataset examples
  -> seed prompt G-Eval
  -> base judge Qwen2.5-7B produce score/rationale
  -> metric_fn confronta score con target umano
  -> feedback GEPA:
       - errore metrica
       - opzionale perplexity sul Qwen2.5-7B
       - opzionale NLA verbalization dal Qwen2.5-7B
       - opzionale aux judge 35B come feedback extra
  -> proposer Qwen35B propone nuovo prompt
  -> GEPA valuta candidati su validation
  -> prompt selezionato viene testato sul final test
```

Da sottolineare:

- GEPA ottimizza sul validation set, non sul final test.
- Le metriche finali sono calcolate dopo, sul final test.
- Qwen35B non sostituisce il judge base: e proposer, e opzionalmente aux judge.
- PPL e NLA sono calcolati sul modello base Qwen2.5-7B.

## Domande probabili e risposte brevi

### "NLA ha migliorato?"

Risposta: per ora non abbiamo una risposta positiva solida. Ha migliorato nello
smoke Topical-Chat Qwen35B, ma nella fixed-NLA long GEPA ha scelto il seed
invariato. Serve il controllo current-code PPL `11913587`.

### "Perche la prima NLA long e peggiorata?"

Token selection debole e verbalizzazioni ripetitive/generiche. Era soprattutto
un test di plumbing/condizione NLA non ancora buona, non una confutazione
definitiva di NLA.

### "La token selection fixed ha risolto?"

Ha migliorato coverage e qualita tecnica: token status ok, NLA reale,
verbalizzazioni piu corte, candidate tokens piu presenti. Pero molte
verbalizzazioni restano completion-like, cioe non spiegano direttamente come
correggere la rubrica.

### "Perche non basta candidate-only?"

Perche candidate-content-10 elimina i duplicati ma peggiora comunque. Quindi il
problema principale e l'allineamento semantico del feedback, non solo la
duplicazione source/reference.

### "Serve una run piu lunga?"

Possibile, ma non e la spiegazione principale. Nella fixed-NLA long il seed era
il best score fin dalla riga 0 e nessun candidato lo ha superato. Prima di
spendere 12-16h su raw NLA, serve il controllo PPL current-code o uno smoke in
cui NLA venga trasformata in feedback piu semantico.

### "A cosa serve aux judge?"

Serve a testare se un modello 35B puo produrre feedback semantico utile al
proposer, potenzialmente comprimendo PPL/NLA/esempio/errore in una regola di
rubrica. Va separato in modo scientifico:

```text
ppl_nla_auxjudge_no_nla_to_aux
ppl_nla_auxjudge_nla_context
```

Se l'aux judge riceve NLA, non e piu solo "aggiungo aux judge": e "uso il 35B
per interpretare/comprimere NLA".

### "Quali metriche sono paper-aligned?"

```text
Topical-Chat: Pearson, Spearman
SummEval: Spearman, Kendall tau
QAGS-CNN/XSUM: Pearson, Spearman, Kendall tau
```

Agreement e MAE restano utili per debug, ma non sono il confronto principale
con il paper.

## Cosa mostrare se hai poco tempo

Ordine consigliato:

1. `nla_evidence_deep_dive_20260611.md`: tabella Run Metrics e Observations.
2. `metrics_20260605T161254Z.csv`: PPL long nota al relatore.
3. `metrics_20260610T163111Z.csv`: fixed-NLA long.
4. `nla_vs_ppl_fixed_smoke_20260610.md`: smoke positivo.
5. `nla_candidate_content_10_vs_ppl_smoke_20260611.md`: candidate-only negativo.
6. `runner.py` e `nla_feedback.py`: dove entra il feedback in GEPA.

## Stato da dire alla fine

```text
Abbiamo implementato la matrice G-Eval, le metriche mancanti e NLA reale.
Il primo PPL long era positivo.
La prima NLA long era negativa per feedback debole.
La fixed-NLA smoke era positiva.
La fixed-NLA long e inconclusiva perche GEPA ha tenuto il seed.
Candidate-only NLA non basta.
Il controllo PPL current-code 11913587 e il prossimo risultato necessario.
Aux judge e il prossimo strumento per capire se NLA deve essere trasformata in feedback piu semantico.
```
