# Briefing call relatore - GEPA / G-Eval / NLA

Data aggiornata: 2026-06-17

Scopo: spiegare lo stato della repo e degli esperimenti dal punto in cui il
relatore era rimasto, circa commit `9997e3a`, fino allo stato corrente.

## Messaggio principale

Prima avevamo una pipeline Topical-Chat engagingness con GEPA, Qwen2.5-7B come
judge/base model, Qwen35B via llama.cpp come proposer, e feedback di
perplexity. La run PPL long iniziale era positiva:

```text
Run: gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer
Final test, n=60

baseline:  pearson=0.551831  spearman=0.547848  agreement=0.761111  mae=0.477778
optimized: pearson=0.632812  spearman=0.619893  agreement=0.788889  mae=0.422222
```

Da lì sono state aggiunte tre cose:

- supporto più generale a G-Eval: Topical-Chat, SummEval, QAGS-CNN, QAGS-XSUM;
- metriche paper-aligned: Pearson/Spearman/Kendall tau, più MAE/agreement come diagnostiche;
- NLA come feedback aggiuntivo a GEPA, con diagnostiche per capire quando aiuta o peggiora.

Conclusione attuale: NLA non ha ancora una claim forte di miglioramento. La
fixed-NLA long è leggermente migliore del controllo PPL current-code su tutte
le metriche, ma entrambe le run hanno tenuto lo stesso seed prompt byte-identico.
Quindi il delta è debole: 2 esempi final-test migliorati su 60, 0 peggiorati,
58 invariati. La direzione più sensata ora è usare un auxiliary judge Qwen35B
per comprimere NLA in feedback semantico/rubric-conditioned invece di passare
verbalizzazioni raw al proposer.

Aggiornamento del 2026-06-17: l'aux-judge smoke è stato eseguito ma non è
scientificamente valido perché Qwen35B via llama.cpp ha prodotto solo
`reasoning_content` e `content` vuoto fino a `finish_reason=length` per 36/36
feedback. GEPA ha comunque prodotto metriche finali, ma il feedback ausiliario
è stato marcato errore al 100% e il job lungo dipendente è stato cancellato.
Sono stati anche completati i job soft-prompt/SIPIT e uno sweep NLA token-position
smoke; quest'ultimo ha rivelato un problema di wiring/naming perché i job
avevano nomi diversi ma la strategia effettiva loggata era sempre
`candidate_middle_1`.

## Timeline git da raccontare

- `9997e3a`: stato vicino alla prima run PPL nota al relatore.
- `9ab28b9` / `62897b5`: task registry e runner generalizzato per più dataset.
- `be986f0` / `ba1b0db`: feedback NLA e precompute reale delle verbalizzazioni.
- `26caa67`: auxiliary judge opzionale.
- `de2f561`: Kendall tau.
- `0d2ecd6`: diagnostica NLA vs controllo.
- `93265a1`: token selection NLA fixed.
- `7f4bcd5`: parser più robusto per tag NLA parziali.
- `da63910`: impedito fallback dry-run NLA nei run scientifici.
- `91e5431`: registrato replacement PPL current-code.
- Stato attuale: `11914197` aux-judge smoke completato ma invalidato dal guard,
  `11914211`-`11914222` sweep NLA smoke completato con problema di wiring,
  `11914226`/`11914232` soft-prompt e `11914237`/`11914239` SIPIT completati.

## Risultati principali

### 1. PPL long iniziale

```text
gepa-experiments/results/geval_gepa_engaging_qwen25_8h_ppl_llamacpp35b_proposer

baseline:  pearson=0.551831  spearman=0.547848  agreement=0.761111  mae=0.477778
optimized: pearson=0.632812  spearman=0.619893  agreement=0.788889  mae=0.422222
```

Interpretazione: GEPA+PPL aveva migliorato il seed in questo setting.

### 2. Prima NLA long

```text
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b

baseline:  pearson=0.723364  spearman=0.721559  agreement=0.769444  mae=0.461111
optimized: pearson=0.511144  spearman=0.490815  agreement=0.691667  mae=0.616667
runtime:   28355.7s, circa 7h52m
trajectory rows: 631
seed words: 192
optimized words: 339
```

Interpretazione: negativa. Non è una confutazione definitiva di NLA perché la
token selection era debole e produceva verbalizzazioni raw ripetitive/generiche.

### 3. Fixed-NLA smoke Qwen35B

```text
PPL smoke optimized, n=12:
pearson=0.536400  spearman=0.527410  kendall=0.459933  agreement=0.638889  mae=0.722222

Fixed-NLA smoke optimized, n=12:
pearson=0.674979  spearman=0.674693  kendall=0.606407  agreement=0.763889  mae=0.472222
```

Interpretazione: segnale positivo, ma solo smoke su 12 esempi.

### 4. Fixed-NLA long

```text
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b

baseline:  pearson=0.681158  spearman=0.677076  kendall=0.571150  agreement=0.752778  mae=0.494444
optimized: pearson=0.681158  spearman=0.677076  kendall=0.571150  agreement=0.752778  mae=0.494444
runtime:   30519.012s, circa 8h28m
trajectory rows: 740
seed == optimized: true
```

Confronto con il controllo PPL current-code:

```text
PPL current-code optimized, n=60:
pearson=0.658218  spearman=0.658203  kendall=0.555309  agreement=0.741667  mae=0.516667

Fixed-NLA optimized minus PPL current-code optimized:
pearson   +0.022940
spearman  +0.018872
kendall   +0.015841
agreement +0.011111
mae       -0.022222  (meglio)
```

Interpretazione: confronto 1-to-1 pulito lato config/split. Però i prompt finali
sono identici al seed in entrambe le run, quindi non possiamo dire che GEPA con
NLA abbia trovato un prompt migliore. È un weak positive: 2 esempi migliorati,
0 peggiorati, 58 invariati.

### 5. Candidate-only NLA

```text
candidate_content_10 optimized, n=12:
pearson=0.402090  spearman=0.371727  kendall=0.310087  agreement=0.625000  mae=0.750000
```

Interpretazione: togliere source/reference e usare solo candidate tokens non
basta. Anche con duplicati quasi eliminati, il feedback resta poco metric-aligned.

### 6. Dataset smoke

SummEval consistency è l’unico dataset smoke leggibile direzionalmente:

```text
PPL optimized, n=32:
pearson=0.701281  spearman=0.790860  kendall=0.716853  agreement=0.723958  mae=1.104167

PPL+NLA optimized, n=32:
pearson=0.618512  spearman=0.696582  agreement=0.718750  mae=1.125000
```

Interpretazione: direzione negativa per NLA. QAGS-CNN e QAGS-XSUM hanno final
test `n=2`, quindi sono solo plumbing checks.

### 7. Aux-judge smoke Qwen35B

```text
job: 11914197
artifact: gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke
final test, n=12

baseline:  pearson=0.603136  spearman=0.590879  kendall=0.531588  agreement=0.722222  mae=0.555556
optimized: pearson=0.693240  spearman=0.691967  kendall=0.632890  agreement=0.569444  mae=0.861111

aux feedback rows: 36
aux status: error=36/36
finish_reason: length=36/36
errore: message content vuoto, reasoning_content presente
```

Interpretazione: il numero Pearson migliora nello smoke, ma non va usato come
evidenza per l'aux-judge perché il feedback ausiliario non è entrato davvero in
modo valido. Il guard finale ha bloccato il job con:

```text
Aux judge feedback success rate is below threshold: 0.000 < 0.950
```

Il job lungo dipendente `11914198` era quindi in `DependencyNeverSatisfied` ed è
stato cancellato il 2026-06-17. Prossimo fix necessario: cambiare configurazione
o parsing dell'aux judge per evitare risposte solo in `reasoning_content`
(`max_tokens` più alto, modalità no-thinking se disponibile, o parser che accetta
reasoning solo se produce feedback finale utilizzabile).

### 8. Soft-prompt e SIPIT sui soft token

```text
soft_prompt_topical_chat_engagingness_long
train rows: 240, tokenized: 234/240, max_seq_len=1024

validation: pearson 0.537111 -> 0.548315  delta +0.011204
test:       pearson 0.756774 -> 0.723530  delta -0.033244
```

```text
soft_prompt_topical_chat_engagingness_long_2048
train rows: 240, tokenized: 240/240, max_seq_len=2048

validation: pearson 0.537111 -> 0.599639  delta +0.062528
test:       pearson 0.756774 -> 0.708456  delta -0.048318
```

Interpretazione: max sequence length 2048 risolve la perdita di esempi in
training e migliora molto di più la validation, però peggiora il final test.
Questo è utile come artifact di explainability, ma non come miglioramento del
judge G-Eval.

Contesto tecnico: questi job seguono l'idea del notebook soft-prompting citato
dal relatore. Il modello base è sempre `Qwen/Qwen2.5-7B-Instruct`, congelato; gli
unici parametri allenati sono 16 virtual tokens PEFT/prompt-tuning. Il task è lo
stesso G-Eval Topical-Chat engagingness usato da GEPA: il modello deve produrre
lo score 1/2/3 e la loss viene calcolata solo sui token target dello score, non
su tutto il prompt. Gli split sono 40/10/10 gruppi, cioè 240/60/60 righe. La run
`max_seq_len=1024` perde 6 esempi di train perché troppo lunghi, mentre
`max_seq_len=2048` mantiene 240/240 esempi.

La parte SIPIT è una diagnostica, non una claim di inversione esatta già riuscita.
Lo script fa due letture diverse dei soft token:

1. nearest-token baseline: per ogni vettore soft prende il token del vocabolario
   con embedding più vicino in L2. Questo produce:

```text
nearest_text in entrambi: "You are a careful, impartial evaluator. Rate the candidate output according to the rub"
nearest_mean_l2 long:      2.178889
nearest_mean_l2 long_2048: 2.113662
```

Questo significa che, proiettando ogni soft vector sul token reale più vicino, il
soft prompt assomiglia ancora molto al testo di inizializzazione. Non vuol dire
che il modello "abbia imparato quella frase" in senso forte: vuol dire che la
proiezione discreta più vicina resta ancorata all'inizializzazione testuale.

2. bounded SIPIT-style recovery: lo script usa i vettori continui del soft prompt
   come target nascosto e prova a trovare una sequenza discreta di token che
   riproduca quegli hidden states. Qui la verifica non passa:

```text
long:      all_positions_verified=false, nearest_mean_l2=2.178889, elapsed=2034.9s
long_2048: all_positions_verified=false, nearest_mean_l2=2.113662, elapsed=2046.7s
nearest_text in entrambi: "You are a careful, impartial evaluator. Rate the candidate output according to the rub"
```

`all_positions_verified=false` significa che nessuna delle 16 posizioni soft è
stata verificata come ricostruzione discreta esatta. Questo non è sorprendente:
i soft prompt sono vettori continui addestrati, quindi possono stare fuori dalla
manifold degli embedding di token reali. Inoltre il job usava recovery bounded
con 4-bit quantization, ultimo layer risolto a layer 28 e `max_iters_per_token=500`;
queste scelte rendono l'esperimento economico ma non massimizzano la probabilità
di recovery esatta. I `timesteps=500` su tutte le posizioni indicano che il
budget massimo è stato consumato per ogni token senza verificare la soluzione.

Il recovered text ottimizzato contiene caratteri non interpretabili/non italiani
e non inglesi, quindi non è utile semanticamente. La lettura corretta è:

- il soft prompt sembra migliorare la validation ma peggiorare il test, quindi
  può star overfittando lo split di validation;
- il nearest-token decode recupera soprattutto il prompt di init, non una regola
  nuova chiara;
- SIPIT non ha ancora dimostrato una verbalizzazione affidabile dei soft token;
- per trarre una conclusione servono controlli: token reali noti, embedding del
  prompt di init, precisione più alta o più iterazioni, e random-vector controls.

### 9. Sweep NLA token positions

Sono stati lanciati 12 job smoke `11914211`-`11914222` per testare posizioni
diverse dei token da verbalizzare. Tutti risultano `COMPLETED`, ma il risultato
non è utilizzabile come confronto scientifico tra strategie perché i log dicono:

```text
job name intended: candidate_first_1 / candidate_last_1 / candidate_fml_3 / ...
actual logged strategy: candidate_middle_1
output dir effettiva: experimental_nla_candidate_middle_1_topical_chat_smoke
```

In più ci sono metriche finali per 10/12 job; `11914211` e `11914222` risultano
completati da Slurm ma non hanno un `runtime_manifest`/metrics finale distinto.
Sui 10 artifact recuperati, il migliore per Pearson ottimizzato è:

```text
intended=candidate_quintile_5, actual=candidate_middle_1, job=11914215, n=12
baseline:  pearson=0.638111  spearman=0.607320  kendall=0.522651  agreement=0.750000  mae=0.500000
optimized: pearson=0.670151  spearman=0.688091  kendall=0.599501  agreement=0.680556  mae=0.638889
delta:     pearson=+0.032040 spearman=+0.080771 kendall=+0.076850 agreement=-0.069444 mae=+0.138889
```

Il peggiore è:

```text
intended=reference_fml_3, actual=candidate_middle_1, job=11914218, n=12
baseline:  pearson=0.638111
optimized: pearson=0.223384
delta:     pearson=-0.414727
```

Interpretazione: questi numeri sono utili come controllo di plumbing e come
segnale che la pipeline può generare prompt diversi, ma non rispondono ancora
alla domanda del relatore "quali token NLA conviene verbalizzare?". Prima va
fixato il wiring della strategia e salvato sempre il nome della strategia
effettiva in `run_config`.

## Job necessari per soddisfare il messaggio del relatore

Obiettivo esatto del messaggio: verificare se NLA sta verbalizzando i token
giusti, provando più posizioni in modo naive ma controllato; e usare il notebook
soft-prompting per allenare lo stesso task, poi capire a cosa corrispondono i
soft token tramite SIPIT/random-token controls. Aggiungo anche aux-judge perché
nelle conversazioni successive è emerso che va comparato come feedback
rubric-conditioned al proposer.

### A. Fix e validation prima di rilanciare job lunghi

```text
A1. nla_strategy_wiring_probe
    scopo: verificare che EXPERIMENTAL_NLA_TOKEN_STRATEGY arrivi davvero dentro
           il container, dentro run_config e nei nomi artifact.
    configurazione: 1 sola strategia non-default, es. candidate_first_1;
                    train/val/test minimale; MAX_FULL_EVALS=1.
    successo atteso: run_config.strategy=candidate_first_1, output_dir dedicata,
                     manifest/precomputed dedicati, nessun fallback a candidate_middle_1.
    priorità: bloccante per tutto lo sweep NLA.

A2. aux_judge_content_probe
    scopo: verificare che Qwen35B via llama.cpp produca `content` parseabile,
           non solo `reasoning_content`.
    configurazione: poche istanze, niente GEPA long, AUX_JUDGE_MAX_TOKENS alto
                    o modalità no-thinking se disponibile.
    successo atteso: success_rate aux feedback >= 0.95 e feedback breve leggibile.
    priorità: bloccante per aux-judge smoke/long.

A3. sipit_hard_token_control
    scopo: controllare che la recovery SIPIT funzioni quando il target è una
           sequenza di token reali nota.
    configurazione: stesso modello Qwen2.5-7B, stessi layer/precisione se possibile.
    successo atteso: all_positions_verified=true o failure spiegabile.
    priorità: bloccante per interpretare seriamente i fallimenti sui soft prompt.
```

### B. Sweep NLA sulle posizioni token

Questi job vanno rilanciati solo dopo `A1`, perché la prima sweep non ha
applicato davvero strategie diverse.

```text
B1. experimental_nla_candidate_first_1_topical_chat_smoke
B2. experimental_nla_candidate_middle_1_topical_chat_smoke
B3. experimental_nla_candidate_last_1_topical_chat_smoke
B4. experimental_nla_candidate_fml_3_topical_chat_smoke
B5. experimental_nla_candidate_quintile_5_topical_chat_smoke
B6. experimental_nla_candidate_even_8_topical_chat_smoke
B7. experimental_nla_source_fml_3_topical_chat_smoke
B8. experimental_nla_reference_fml_3_topical_chat_smoke
B9. experimental_nla_balanced_fml_9_topical_chat_smoke
B10. experimental_nla_prompt_tail_6_topical_chat_smoke
B11. experimental_nla_evaluation_tail_3_topical_chat_smoke
B12. experimental_nla_hybrid_context_dedup_8_topical_chat_smoke
```

Output obbligatori per ogni job:

```text
metrics.csv
run_config.json con strategy effettiva
runtime_manifest.json
nla_manifest.jsonl
nla_precomputed.jsonl
nla_verbalizations.jsonl
prompt_trajectory.jsonl
gepa_viz_run.json
baseline/optimized predictions
slurm log + dependency manifest
```

Decisione dopo B1-B12: scegliere top 2-3 strategie non solo da Pearson, ma anche
da salute del feedback: coverage, duplicati, token positions, verbalizzazioni
semanticamente leggibili, delta sugli esempi peggiorati/migliorati. Se una
strategia ha segnale positivo e feedback più sano, lanciare:

```text
B13. top_strategy_medium_repeat_seed42
B14. top_strategy_medium_repeat_seed43
B15. top_strategy_medium_repeat_seed44
B16. best_strategy_long_8h_or_12h
```

### C. Soft-prompt e SIPIT

Job già completati e da tenere come baseline diagnostica:

```text
C1. soft_prompt_topical_chat_engagingness_long         job 11914226, max_seq_len=1024
C2. soft_prompt_topical_chat_engagingness_long_2048    job 11914232, max_seq_len=2048
C3. soft_prompt_sipit_topical_chat_engagingness_long   job 11914239
C4. soft_prompt_sipit_topical_chat_engagingness_2048   job 11914237
```

Job ancora necessari per interpretare quei risultati:

```text
C5. sipit_init_prompt_embedding_control
    scopo: usare embedding del testo di init come target; deve recuperare una
           verbalizzazione vicina al prompt iniziale.
    motivazione: distingue "SIPIT non funziona nel setup" da "i soft token
                 addestrati non sono token reali verbalizzabili".

C6. sipit_random_hard_token_control
    scopo: token reali random, noti e recuperabili.
    motivazione: è il controllo richiesto nel filone SIPIT/random tokens.

C7. sipit_random_continuous_vector_control
    scopo: vettori random con norma simile ai soft prompt.
    motivazione: se fallisce come i soft prompt, conferma che il problema è
                 off-manifold; se riesce, il problema è specifico dei soft token.

C8. soft_prompt_2048_repeat_seed43
C9. soft_prompt_2048_repeat_seed44
    scopo: capire se validation up / test down è overfit casuale o pattern stabile.

C10. sipit_soft_prompt_2048_precision16_or_more_iters
    scopo: verificare se il `verified=false` dipende dal budget economico
           4-bit/500 iters o dalla natura off-manifold dei soft prompt.
```

### D. Aux-judge come feedback al proposer

Questi job servono per comparare la parte "llm-as-a-judge ausiliario" descritta
dal relatore. Vanno lanciati dopo `A2`, perché il primo tentativo ha fallito con
36/36 feedback vuoti.

```text
D1. aux_judge_fixed_smoke_ppl_nla
    config: Topical-Chat engagingness, Qwen2.5-7B judge, Qwen35B proposer,
            PPL + NLA + aux-judge, n=12 final test, min_success_rate>=0.95.
    scopo: verificare che il feedback ausiliario funzioni davvero.

D2. matched_no_aux_smoke_ppl_nla
    config: identica a D1 ma senza aux-judge.
    scopo: ablation diretta per isolare il contributo aux-judge.

D3. aux_judge_only_smoke_ppl_aux_no_nla
    config: PPL + aux-judge, senza NLA.
    scopo: capire se il miglioramento viene dall'aux judge o dall'interazione con NLA.

D4. aux_judge_fixed_long_ppl_nla
    config: stessa di D1, durata 8-12h se D1 passa.
    scopo: run principale per claim "NLA compresso da aux judge aiuta GEPA".

D5. matched_no_aux_long_ppl_nla
    config: stessa durata/split/seed di D4 ma senza aux-judge.
    scopo: confronto scientifico minimo per dire se aux-judge aggiunge valore.
```

Output obbligatori aggiuntivi per D1-D5:

```text
aux_judge_feedback.jsonl
aux success_rate e status counts
raw_response/finish_reason per debug
feedback finale passato al proposer
prompt trajectory completa
GEPA viz artifact
tempo medio per feedback aux
```

Ordine pratico consigliato: A1, A2, A3; poi B1-B12 e C5-C7 in parallelo; poi
D1-D3; poi C8-C10; infine solo se il segnale è sano B13-B16 e D4-D5.

## Stato cluster

Al check del 2026-06-17:

```text
squeue: nessun job utente visibile
11914197: FAILED lato guard aux-judge, artifact recuperati localmente
11914198: PENDING DependencyNeverSatisfied da afterok:11914197(failed), cancellato
11914211-11914222: COMPLETED, sweep NLA smoke, artifact recuperati localmente
11914226/11914232: COMPLETED, soft-prompt long e long_2048
11914237/11914239: COMPLETED, SIPIT recovery sui soft prompt
```

Il monitor Telegram è stato patchato e deployato su faretra e moro232:

```text
gepa-experiments/slurm/telegram_monitor.py
```

Fix applicati:

- logga anche gli invii Telegram riusciti, non solo gli errori;
- segnala job pendenti con ragioni terminali come `DependencyNeverSatisfied`;
- test manuale da faretra: `telegram send ok`;
- test su `11914198`: inviati start, state change, job cannot start, monitor exiting.

Crash recenti non dovuti a noi:

```text
11913415: vLLM startup fallito perché la GPU assegnata aveva solo ~7.22 GiB liberi
11913482/11913557: llama.cpp startup fallito perché la GPU proposer aveva solo ~7.4 GiB liberi
```

## File da mostrare

Risultati:

```text
gepa-experiments/results/diagnostics/nla_evidence_deep_dive_20260612.md
gepa-experiments/results/diagnostics/nla_fixed_long_vs_current_ppl_long_20260612.md
gepa-experiments/results/diagnostics/nla_vs_ppl_fixed_smoke_20260610.md
gepa-experiments/results/diagnostics/nla_candidate_content_10_vs_ppl_smoke_20260611.md
gepa-experiments/results/diagnostics/nla_summeval_consistency_vs_ppl_smoke_20260611.md
gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke/aux_judge_feedback_20260617T100402Z.jsonl
gepa-experiments/results/geval_gepa_topical_chat_engagingness_ppl_nla_auxjudge_llamacpp35b_smoke/metrics_20260617T100402Z.csv
gepa-experiments/results/soft_prompt_topical_chat_engagingness_long/metrics.json
gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048/metrics.json
gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_sipit/sipit_recovery.json
gepa-experiments/results/soft_prompt_topical_chat_engagingness_long_2048_sipit/sipit_recovery.json
gepa-experiments/results/experimental_nla_candidate_middle_1_topical_chat_smoke/metrics_20260617T*.csv
```

Prompt/artifact:

```text
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_llamacpp35b_current_control
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_fixed_nla_llamacpp35b
gepa-experiments/results/geval_gepa_topical_chat_engagingness_8h_ppl_nla_llamacpp35b
```

Codice:

```text
gepa-experiments/geval_gepa/runner.py
gepa-experiments/geval_gepa/nla_feedback.py
gepa-experiments/geval_gepa/nla_precompute.py
gepa-experiments/geval_gepa/aux_judge.py
gepa-experiments/geval_gepa/metrics.py
gepa-experiments/geval_gepa/tasks.py
gepa-experiments/scripts/diagnose_nla_run.py
gepa-experiments/scripts/analyze_nla_evidence.py
```

## Pipeline attuale

```text
dataset examples
  -> seed prompt G-Eval
  -> Qwen2.5-7B judge produce score/rationale
  -> metric_fn confronta score con target umano
  -> feedback GEPA:
       - errore metrica
       - opzionale perplexity su Qwen2.5-7B
       - opzionale NLA verbalization da Qwen2.5-7B
       - opzionale aux judge Qwen35B
  -> Qwen35B proposer propone nuovo prompt
  -> GEPA valuta candidati su validation
  -> prompt selezionato viene testato sul final test
```

Da sottolineare:

- GEPA ottimizza su validation, non su final test.
- Le metriche finali sono calcolate sul final test.
- Qwen35B è proposer e opzionalmente aux judge, non sostituisce il judge base.
- PPL e NLA sono calcolati sul modello base Qwen2.5-7B.

## Risposte probabili

### NLA ha migliorato?

Risposta breve: non abbastanza per una claim forte. C’è un weak-positive nella
long matched current-code, ma senza prompt improvement. NLA raw sembra utile da
studiare, ma non ancora affidabile come feedback diretto a GEPA.

### Perché la prima NLA long è peggiorata?

Perché il feedback NLA era rumoroso: token selection debole, molte
verbalizzazioni ripetitive/completion-like, e GEPA ha probabilmente ottimizzato
su segnali non allineati alla metrica G-Eval.

### La fixed token selection ha risolto?

Ha risolto parte del problema tecnico: token status ok, coverage buona,
verbalizzazioni più corte, più candidate tokens. Non ha risolto il problema
semantico: il testo NLA resta spesso una continuazione/associazione latente,
non una ragione rubric-aligned.

### Perché serve aux judge?

Per testare se Qwen35B può trasformare PPL+NLA+errore in una regola breve di
rubrica utile al proposer. Questo è diverso dal passare raw NLA direttamente.
Il primo smoke però non ha prodotto feedback valido: va fixato prima di lanciare
una long run aux-judge.

### Serve una run più lunga?

Non è la priorità. Le long run hanno già girato circa 8h. Il problema ora non è
solo convergenza: nella fixed-NLA long il prompt migliore è rimasto il seed.
Prima di 12-16h raw NLA conviene provare feedback semanticamente compresso.

## Cosa manca per scrivere la tesi

- Decidere con il relatore se la claim è positiva su NLA o diagnostica/negativa su raw NLA.
- Fixare e rilanciare l’aux judge smoke Qwen35B: il primo tentativo ha fallito
  perché 36/36 risposte avevano `content` vuoto e solo `reasoning_content`.
- Fixare lo sweep NLA token-position: i job sono partiti con nomi diversi ma la
  strategia effettiva loggata era sempre `candidate_middle_1`; inoltre vanno
  salvati strategy name e artifact finali per ogni job.
- Se si vuole claim positiva, ottenere una run dove GEPA seleziona davvero un prompt migliore con NLA/aux-NLA.
- Espandere la matrice full paper-aligned o dichiarare chiaramente che i dataset extra sono smoke/pilot. La matrice completa aggiornata è in `gepa-experiments/status/full_matrix_execution_plan_20260613.md`.
- Tenere separata la nuova pipeline joint-prompt multi-dimensione: può produrre metriche per tutte le dimensioni in una sola run per dataset, ma non sostituisce i risultati single-dimension paper-aligned.
- Nuovo vincolo operativo: provare a completare i job selezionati entro il 28 giugno incluso; per questo i `base_gepa` sono stati messi in fondo e un primo joint-prompt Topical-Chat `ppl` è stato alzato di priorità come benchmark.
- Prima delle run lunghe finali, aggiungere timing per fase oltre al runtime totale, così si possono riportare anche i costi di processing.
- Preparare tabella finale con n, split, modello proposer, feedback variant, Pearson/Spearman/Kendall, MAE/agreement, prompt changed yes/no, artifact path.
- Recuperare prompt rappresentativi per appendice: seed, PPL optimized, old NLA optimized, fixed-NLA final, candidate-only failed prompt.

## Stato da dire alla fine

```text
Abbiamo implementato la pipeline G-Eval multi-dataset, le metriche mancanti e NLA reale.
PPL long iniziale era positivo.
Prima NLA long era negativa per feedback debole.
Fixed-NLA smoke era positiva.
Fixed-NLA long è weak-positive contro PPL current-code, ma non è una claim forte perché il prompt resta il seed.
Candidate-only NLA non basta.
Il primo aux-judge smoke ha mostrato il problema giusto da fixare: Qwen35B ha ragionato ma non ha prodotto content parseabile.
Lo sweep sulle posizioni NLA va ripetuto dopo fix del wiring perché le strategie non sono state applicate davvero.
Il prossimo test utile resta aux judge / compressed NLA feedback con Qwen35B, ma prima serve rendere affidabile il feedback ausiliario.
```
