# Advisor Comment Replies - 2026-06-26

Risposte da usare su Overleaf / Teams dopo aver portato le relative modifiche
LaTeX. Sono volutamente scritte in modo meno "perfetto", cosi sembrano piu da
me e non una lista generata.

## Overall Reply

```text
ho dato priorita al metodo come dicevi. Ho riscritto il cap 3 con una narrativa piu lineare: prima corpus semantico, poi task judge, poi GEPA/LatentGEPA come metodo principale, poi i feedback ppl/nla/aux e alla fine il readout soft prompt. Ho tolto la roba a branch dove confondeva, e SIPIT/NLA non sono piu messi come metodi nuovi miei, ma come strumenti gia spiegati nei related e richiamati solo dove servono. Ho anche fatto un passaggio apposta sul tono generale, quindi ho tolto varie frasi in stile "la tesi non fa X" e le ho riscritte in modo piu formale/positivo. Ho anche aggiunto il cap 6, ripulito i riferimenti troppo locali a file/script e ricontrollato caption/label sugli oggetti mostrati.
```

## Comment-Level Replies

| Comment / topic | Copy-paste reply |
|---|---|
| Messaggio generale Teams / direzione del metodo | `ho ricontrollato anche il messaggio generale, non solo i commenti puntuali. Ho cercato di sistemare proprio la direzione: ora il metodo non e' una cronologia di esperimenti/branch, ma parte dal corpus, definisce il judge task e mette LatentGEPA come metodo principale. SIPIT e NLA restano strumenti/prior work, GEPA invece e' il backbone spiegato nel metodo.` |
| Non cancellare i commenti Overleaf | `ok, da ora li tengo aperti e rispondo sotto. Sto facendo le modifiche in locale con commit piccoli, cosi poi posso riportarle su Overleaf senza perdere il contesto dei tuoi commenti.` |
| Metodo troppo frammentato / difensivo | `fatto, ho riscritto il metodo con una struttura piu unica e meno difensiva. Ora parte dicendo cosa propongo, cioe corpus semantico + LatentGEPA, invece che dire continuamente cosa la tesi non fa.` |
| Troppi “branch” | `fatto, ho tolto branch come framing del metodo. Ora parlo di componenti, diagnostiche, condizioni sperimentali o feedback variants, a seconda del punto. Cosi non sembra che SIPIT/NLA/GEPA siano tre metodi proposti allo stesso livello.` |
| Riferimenti a file Python / script / path locali | `fatto, ho tolto dal testo finale i riferimeti a file python, nomi di cache e path locali. Dove serviva mantenere la provenance l'ho riscritta spiegando cosa e' stato generato/confrontato, non il nome del file.` |
| Sezione “Why GEPA fits this thesis” | `fatto, l'ho rimossa come sezione standalone. Ho tenuto solo il concetto utile nei related, cioe GEPA come prior work di prompt optimization su cui poi costruisco LatentGEPA.` |
| Intro del metodo diceva cosa il capitolo non mostra | `fatto, l'intro del cap 3 ora e' positiva: dice cosa definisce il capitolo e qual e' il contributo. Ho tolto la parte tipo "questa tesi non fa..." perche effettivamente suonava male.` |
| Frasi difensive su GEPA / metodo | `fatto, ho riformulato attorno a LatentGEPA. GEPA ora e' il backbone di prompt optimization, e la mia modifica e' presentata come feedback latente dato al proposer, non come una cosa separata/giustificata a posteriori.` |
| Titolo “Canonical” poco chiaro | `fatto, ho rinominato la sezione in Semantic-Fidelity Corpus. Mi sembra piu chiaro perche dice subito il ruolo del dataset invece di usare "canonical", che era poco informativo.` |
| Dataset poco spiegato | `fatto, ho esteso molto la sezione: costruzione del corpus, blocchi A/B/C, metadata, esempi concreti, split e statistiche ricontrollate. Ho messo anche le lunghezze medie cosi e' piu formale.` |
| Downstream usage descritta troppo presto nel dataset | `fatto, ho tolto l'elenco metodo-per-metodo da li. Ora il dataset viene presentato come oggetto comune, mentre l'uso specifico sta dopo nelle parti metodo/setup/results dove serve.` |
| Embedding inversion nel metodo era troppo lunga | `fatto, nel cap 3 resta solo il framing diagnostico che serve a capire la semantic fidelity. I dettagli su reproduction boundary, Jina adapters e risultati deboli sono spostati in setup/results, dove hanno piu senso.` |
| Riferimento a Python file nella figura embedding inversion | `fatto, quella cosa non e' piu nel metodo. Dove parlo della provenance nei risultati, non cito piu file/script locali ma solo il setting pratico dell'esperimento.` |
| Sezione reproduction boundary poco chiara / troppo da risultati | `fatto, non e' piu una sezione del metodo. L'ho spostata nei risultati perche serve a interpretare perche non e' una reproduction pulita del paper, non a definire il metodo.` |
| SIPIT dovrebbe stare nei related | `fatto, SIPIT non e' piu spiegato come metodo nel cap 3. Rimane nei related; nel metodo viene solo richiamato come readout-style tool per analizzare i soft prompt, perche li serve a capire l'esperimento.` |
| NLA dovrebbe stare nei related o richiamato brevemente | `fatto, NLA non e' piu una sezione metodo autonoma. Nel cap 3 compare solo come sorgente di feedback per LatentGEPA; checkpoint, dettagli tecnici e validazione sono negli altri capitoli.` |
| Judge task posizionata male | `sistemato il senso della sezione. Non l'ho buttata tutta in experimental setup perche serve prima del loop GEPA per capire cosa sta ottimizzando il prompt; pero ho lasciato nel cap 4 i dettagli sperimentali veri, split/config ecc.` |
| GEPA va spiegato meglio | `fatto, ho messo GEPA/LatentGEPA come blocco centrale del metodo. Ho aggiunto seed prompt, minibatch proposal, reflection data, proposer, candidate pool e selezione tipo Pareto/frontier nello pseudocodice e nel testo.` |
| Prompt / pseudocodice nel formato richiesto | `fatto, prompt e pseudocodice ora sono in box/figure con caption e label, seguendo il formato tcolorbox che mi avevi suggerito. Ho evitato snippet inline buttati nel testo.` |
| Caption per figure/listing/tabelle | `fatto, ho ricontrollato gli oggetti mostrati e aggiunto caption/label anche ai listing lunghi del feedback PPL+NLA/proposer. Dove il contenuto era troppo locale l'ho tolto o spostato.` |
| Risultati Jina / public model non chiari | `fatto, ho chiarito meglio il punto: "public Jina model" vuol dire solo encoder pubblico usato per produrre embedding, non inverter gia pronto. Quindi non sostituisce il training/reproduction dell'inverter.` |
| Jina task adapters non spiegati | `fatto, ho aggiunto una spiegazione pratica: gli adapter sono modalita task-specific dell'encoder, tipo retrieval/classification/text matching, e cambiano lo spazio vettoriale. Quindi se non matchano cambia proprio l'input dell'inverter.` |
| Tabella hardware troppo dettagliata | `sistemato, ora la tabella non contiene indirizzi o dettagli interni. Riporta solo tipo/numero di GPU e VRAM, che e' quello utile per capire gli esperimenti.` |
| Chapter summary troppo generiche | `fatto, la summary generica del cap 3 e' stata rimossa. Nel cap 4 invece ho lasciato una checklist di setup/riproducibilita, perche li serve davvero e non e' una chiusura generica.` |
| Capitolo 6 mancante | `fatto, ho aggiunto Conclusions and Future Work con conclusioni, limiti, sviluppi futuri e acknowledgements. Ho tenuto claim controllati, soprattutto su NLA, per non promettere un miglioramento che i dati ancora non mostrano.` |
| Acknowledgements | `fatto, ho aggiunto i ringraziamenti e ho messo prima il prof Moro come da linee guida.` |
| Numeri dataset | `fatto, li ho ricontrollati dagli artifact: 2080 righe totali, blocchi 40/720/1320 e split 962/290/828 train/validation/test. Ho aggiornato anche la sezione con la lunghezza media.` |
| Underfull warning | `qui non ho trattato gli underfull come errori bloccanti. La build compila; molti warning vengono da tabelle strette. Li sistemerei solo se nel pdf si vede proprio un layout brutto, altrimenti rischiamo di perdere tempo su rumore.` |

## Notes For Manual Overleaf Porting

- Prima portare il rewrite del Chapter 3, perche molte risposte dipendono da
  quello.
- Poi portare i fix Chapter 4/5, soprattutto Jina provenance, hardware table e
  rimozione dei riferimenti locali.
- Poi portare Chapter 6.
- Dopo ogni port rispondere al commento Overleaf usando la riga corrispondente
  sopra, senza cancellare il commento.
