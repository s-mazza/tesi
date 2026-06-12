# Briefing call relatore - GEPA / G-Eval / NLA

Data aggiornata: 2026-06-12

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
- Stato attuale: `11913587` completato e confrontato con fixed-NLA long.

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

## Stato cluster

Al check del 2026-06-12:

```text
squeue: nessun job utente visibile su faretra o moro232
sacct: non affidabile, SlurmDB restituisce Connection refused
11913587: completato, artifact recuperati localmente
```

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

### Serve una run più lunga?

Non è la priorità. Le long run hanno già girato circa 8h. Il problema ora non è
solo convergenza: nella fixed-NLA long il prompt migliore è rimasto il seed.
Prima di 12-16h raw NLA conviene provare feedback semanticamente compresso.

## Cosa manca per scrivere la tesi

- Decidere con il relatore se la claim è positiva su NLA o diagnostica/negativa su raw NLA.
- Eseguire o rimandare esplicitamente l’aux judge smoke Qwen35B.
- Se si vuole claim positiva, ottenere una run dove GEPA seleziona davvero un prompt migliore con NLA/aux-NLA.
- Espandere la matrice full paper-aligned o dichiarare chiaramente che i dataset extra sono smoke/pilot.
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
Il prossimo test utile è aux judge / compressed NLA feedback con Qwen35B.
```
