# Advisor Comment Replies - 2026-06-26

Use these as copy-paste replies on Overleaf/Teams after porting the matching
LaTeX changes. The tone is intentionally concise and first-person.

## Overall Reply

```text
ho dato priorità al metodo come dicevi. Ho riscritto il capitolo 3 con una narrativa più lineare: prima corpus semantico, poi task judge, poi Latent-GEPA come metodo principale, poi feedback PPL/NLA/aux e soft prompt readout. Ho tolto la struttura a branch dove creava confusione, ho spostato SIPIT/NLA fuori dal metodo dettagliato e li ho lasciati solo come strumenti/readout già introdotti nei related. Ho anche aggiunto il capitolo 6 e ripulito i riferimenti troppo locali a file/script nel testo finale.
```

## Comment-Level Replies

| Comment / topic | Copy-paste reply |
|---|---|
| Non cancellare i commenti Overleaf | `ok, li tengo aperti e rispondo sotto. Le modifiche le sto facendo in locale con commit piccoli così poi posso riportarle su Overleaf senza perdere il contesto dei commenti.` |
| Metodo troppo frammentato / difensivo | `fatto, ho riscritto il capitolo metodo con una struttura unica e positiva. Ora non parte più dicendo cosa la tesi non fa, ma introduce direttamente cosa propone: Semantic-Fidelity Corpus e Latent-GEPA.` |
| Troppi “branch” | `fatto, ho tolto “branch” dal testo finale e l'ho sostituito con componenti, diagnostiche, condizioni o experiment family a seconda del caso, così non sembra che SIPIT/NLA/GEPA siano tre metodi proposti sullo stesso piano.` |
| Riferimenti a file Python / script / path locali | `fatto, nel testo finale ho tolto riferimenti a file o cache locali tipo nomi di script e path. Dove serviva mantenere la provenance l'ho riscritta a livello metodologico, cioè spiegando cosa è stato generato e confrontato senza legarlo al nome del file locale.` |
| Sezione “Why GEPA fits this thesis” | `fatto, l'ho rimossa come sezione standalone. Ho tenuto solo una frase utile nei related per dire che GEPA è il prior work su cui poi costruisco Latent-GEPA nel metodo.` |
| Intro del metodo diceva cosa il capitolo non mostra | `fatto, l'introduzione del capitolo 3 ora dice direttamente cosa definisce il metodo e qual è il contributo principale, invece di partire con esclusioni o limiti.` |
| Frasi difensive su GEPA / metodo | `fatto, ho riformulato tutto attorno a Latent-GEPA. Ora GEPA non è presentato come “un branch” ma come il backbone di prompt optimization che viene esteso con feedback latente.` |
| Titolo “Canonical” poco chiaro | `fatto, ho rinominato la sezione in Semantic-Fidelity Corpus, che descrive meglio il ruolo del dataset nella tesi.` |
| Dataset poco spiegato | `fatto, ho aggiunto costruzione del corpus, metadata, blocchi A/B/C, split, conteggi reali ricontrollati e una tabella con esempi rappresentativi.` |
| Downstream usage descritta troppo presto nel dataset | `fatto, ho tolto l'elenco metodo-per-metodo dalla sezione dataset. Ora il corpus viene descritto come oggetto comune, mentre l'uso specifico viene spiegato nelle sezioni metodo/setup/risultati dove serve.` |
| Embedding inversion nel metodo era troppo lunga | `fatto, ho ridotto il capitolo 3 a una sezione breve di readout diagnostics. I dettagli di reproduction boundary, Jina task adapters e risultati negativi sono rimasti nel capitolo risultati/setup dove sono più naturali.` |
| Riferimento a Python file nella figura embedding inversion | `fatto, quella parte non è più nel metodo. Nei risultati ho lasciato solo una spiegazione leggibile della provenance senza citare script o nomi file locali.` |
| Sezione reproduction boundary poco chiara / troppo da risultati | `fatto, non è più una sezione del metodo. La reproduction boundary ora è nel capitolo risultati, perché serve a interpretare perché embedding inversion non è una reproduction pulita del paper.` |
| SIPIT dovrebbe stare nei related | `fatto, ho tolto la sezione standalone SIPIT dal metodo. Nel capitolo 3 SIPIT resta solo come readout-style tool per i soft prompt, mentre spiegazione e risultati stanno in related/setup/results.` |
| NLA dovrebbe stare nei related o richiamato brevemente | `fatto, ho tolto la sezione autonoma NLA dal metodo. Nel metodo NLA compare solo come feedback source di Latent-GEPA; dettagli tecnici, checkpoint e validazione stanno negli altri capitoli.` |
| Judge task posizionata male | `fatto, ora il G-EVAL-style judge task viene introdotto prima del loop Latent-GEPA, così il lettore capisce subito cosa sta ottimizzando GEPA.` |
| GEPA va spiegato meglio | `fatto, ho rinominato la sezione in Latent-GEPA Prompt Optimization Loop e ho aggiunto minibatch, reflection data, proposer, candidate pool e Pareto-style frontier nel testo e nello pseudocodice.` |
| Prompt / pseudocodice nel formato richiesto | `fatto, lo pseudocodice e i prompt restano dentro box figure con caption e label, seguendo il formato tcolorbox che mi avevi indicato.` |
| Caption per figure/listing/tabelle | `fatto, ho ricontrollato che gli oggetti mostrati nel metodo abbiano caption/label. Dove il contenuto era troppo da artifact o troppo locale l'ho spostato o tolto.` |
| Risultati Jina / public model non chiari | `fatto, ho chiarito che “public Jina model” significa solo encoder pubblico, non inverter. Quindi il public encoder non sostituisce il training/reproduction dell'inverter, serve solo a generare gli embedding in input.` |
| Jina task adapters non spiegati | `fatto, ho aggiunto una spiegazione pratica: sono modalità task-specifiche dell'encoder che cambiano lo spazio vettoriale, per esempio retrieval/classification/text matching. Questo spiega perché una mismatch lì renderebbe diverso il training dell'inverter.` |
| Tabella hardware troppo dettagliata | `già sistemato, ora la tabella riporta solo tipo/numero GPU e VRAM utile agli esperimenti, senza indirizzi o dettagli interni del cluster.` |
| Chapter summary troppo generiche | `fatto, ho rimosso la summary generica del capitolo 3. Nel capitolo 4 l'ho trasformata in Setup Audit Checklist, quindi non è una chiusura generica ma una checklist di riproducibilità.` |
| Capitolo 6 mancante | `fatto, ho aggiunto Conclusions and Future Work con conclusioni, limiti, future work e acknowledgements. Ho tenuto claim controllati: NLA non viene presentato come miglioramento conclusivo perché i risultati non lo supportano ancora.` |
| Acknowledgements | `fatto, ho aggiunto i ringraziamenti e ho ringraziato prima il prof Moro come da linee guida.` |
| Numeri dataset | `fatto, ho ricontrollato i conteggi dagli artifact: 2080 righe totali, 40/720/1320 per blocchi A/B/C e split 962/290/828 train/validation/test.` |
| Underfull warning | `non li ho trattati come errori bloccanti. La compilazione passa; gli underfull rimasti sono soprattutto dovuti a tabelle strette. Li sistemerei solo dove si vede layout brutto nel PDF.` |

## Notes For Manual Overleaf Porting

- Port first the Chapter 3 structural rewrite, because many comments depend on
  that.
- Then port the Chapter 4/5 wording cleanup, especially Jina provenance and
  `branch` replacements.
- Then port Chapter 6.
- After each port, reply to the relevant Overleaf comment with the matching row
  above rather than deleting the comment.
