# Exact Advisor Comments With Replies - 2026-06-26

Uso: cerca nel file il testo del commento del relatore, poi copia la risposta
subito sotto. I commenti sono riportati come li ho salvati in `thesis/fixes.txt`.

## Messaggio Teams

### Commento

```text
Ok sono riuscito ora a dare il secondo check alla tesi. In generale per piacere non cancellare i miei commenti. Piuttosto fai tu un sotto-commento e scrivi "risolto". Senno non riesco a capire se hai fatto o meno una cosa. Su alcune ho visto che le hai fatte. Ma poi i commenti sono andati a finire in parti strane. Non più attached alle parti dove li avevo messi. Magari hai fatto un git commit da locale. Ma evita perché senno ci perdiamo dove sono le cose.
```

Risposta da dare:

```text
ok hai ragione, da ora non cancello piu i commenti. Porto le modifiche a pezzi e rispondo sotto, cosi resta chiaro cosa e' stato risolto e non perdiamo il riferimento.
```

### Commento

```text
In generale sul metodo c'è ancora tanto da lavorare. Non mi convince per nulla la narrativa. Sembra tutto troppo spezzettato. E soprattutto è un continuo "la mia tesi NON fa X". Usa piuttosto un tono più formale e descrittivo dove con chiarezza dici quali sono i metodi e gli obiettivi.
```

Risposta da dare:

```text
fatto, ho provato a rimettere ordine nel metodo. Ora non e' piu una lista di cose separate: parte dal Semantic-Fidelity Corpus, poi introduce il task di judge, e poi mette LatentGEPA come pipeline principale.
```

### Commento

```text
Ci sono cose come riferimenti a file python o funzioni che assolutamente non vanno usate. Solo se hai del codice estremamente custom. E in quel caso in dei box appositi per il codice. Non in-line con il testo. Tanto per farti un esempio ed essere più chiari:
For each example, the cache stores two aligned objects:
decoder token ids in files named like \texttt{token\_ids\_*.npy}, and the
corresponding Jina-v3 embedding vectors in files named like
\texttt{embeddings\_*.npy}. These vectors were therefore generated locally, not
downloaded as ready-made embeddings from the paper authors.
...
Both the raw
\texttt{AutoModel(...)} forward path without adapter and \texttt{encode(...)}
without an explicit task produced mean cosine similarity 1.000000 against the
stored \texttt{data\_jinav3} embeddings.
Poi ci sono diverse cose non chiarissime in giro. Tipo proprio quest'ultima frase che ho incollato. Sembra molto strana. Quasi "allucinata". Ora non ti sto dicendo che non puoi fare init dell'organizzazione del paper con claudia-codice o simili. Però una volta fatta serve riscrivere tutto in maniera chiara. Meglio scrivere meno cose ma ben fatte. Cioè il progetto l'ho seguito fin dall'inizio e su alcune parti ho difficoltà a comprendere. Figurati un revisore tesi che non sa nulla. Ci sono proprio parti che io toglierei. Tipo "\section{Chapter Summary}" o "\section{Why GEPA fits this Thesis}".
```

Risposta da dare:

```text
fatto, ho tolto i riferimenti a file python/cache/path dal testo finale. La parte Jina/provenance l'ho quasi tutta tagliata perche alla fine appesantiva e non era centrale. Ho lasciato solo il punto importante, cioe che quella parte resta una reproduction diagnostica e non una baseline pulita del paper.
```

### Commento

```text
Non vedo ancora la parte Conclusions ma magari non hai ancora fatto in tempo, ci sta. Te lo faccio notare solo in caso ti sei dimenticato di quella parte.
```

Risposta da dare:

```text
fatto, ho aggiunto il capitolo Conclusions and Future Work con conclusioni, limiti, sviluppi futuri e ringraziamenti. Ho tenuto i claim abbastanza cauti, soprattutto su NLA, perche dai risultati non possiamo ancora dire che migliori in modo netto.
```

### Commento

```text
Ora dai priorità alla scrittura del metodo. Poi torni a fixare results, related etc. Quella parte deve essere ben fatta perchè è quasi l'unica su cui ti valutano. Ora mi sembra davvero confusionaria. Come ti ho scritto nei commenti, non parlare di branch
Quella è una roba che il lettore capisce implicitamente leggendo. Cioè che c'è un'analisi preliminare delle tecniche di emb inversion e poi una in cui alcune di quelle tecniche sono usate.
```

Risposta da dare:

```text
fatto, ho dato priorita al cap 3. Ho tolto il framing a branch e ho provato a farlo leggere come metodo unico: prima analisi/diagnostiche sui segnali latenti, poi uso di alcuni di quei segnali dentro LatentGEPA.
```

### Commento

```text
Non sono ancora convinto dal posizionamento dei vari NLA, SIPIT, GEPA. Cioè tu li hai SIA in related che in method. Dobbiamo essere più ordinati. Sicuramente SIPIT deve andare SOLO in related. Non ha senso metterlo in method. Anche perchè non fai modifiche a quel metodo. La questione di fare semantic similarity invece che hard token assignment negli ultimi esperimenti che hai fatto è più una roba da experimental setup/results. Non da metodo.
```

Risposta da dare:

```text
fatto, SIPIT ora non e' piu presentato come metodo mio nel cap 3. Sta nei related; nel metodo lo richiamo solo quando serve per spiegare il readout dei soft prompt. I dettagli su semantic similarity / hard token assignment li ho lasciati in setup/results.
```

### Commento

```text
NLA alla fine lo usi come tool nel metodo. Quindi anche quello può andare secco solo nei related. E poi brevemente richiamato nel metodo quando discuti delle varie tecniche di "extra-info" in GEPA.
```

Risposta da dare:

```text
fatto, NLA non e' piu una sezione autonoma del metodo. I dettagli stanno nei related/setup/results; nel metodo lo richiamo solo come sorgente di extra-info che viene data al proposer.
```

### Commento

```text
GEPA secondo me è l'unico dei 3 metodi che puoi mettere SOLO nel metodo. Cioè lo citi nei related work quando parli di altre tecniche di prompt optimization. Ma poi la vera descrizione di cosa fa GEPA la metti nel metodo. Questo perchè, come ti dicevo in call, ha più senso descrivere assieme la base e cosa cambi con LatentGEPA. In generale anche la descrizione di GEPA non mi sembra sufficiente. Seppure molto "operativo", GEPA è un algoritmo che ha un paio di step cruciali come ad esempio le mini-batch proposal e la parte di pareto per la selection. E' importante descrivere bene quelle cose. Perché così ti aiuta a far capire al lettore che effetto ha la tua aggiunta e soprattutto dove si inserisce.
```

Risposta da dare:

```text
fatto, GEPA/LatentGEPA ora e' il centro del metodo. Nei related GEPA resta solo come prior work, mentre nel cap 3 spiego meglio seed prompt, minibatch proposal, reflection data, proposer, candidate pool e frontier. Cosi si capisce dove entrano PPL, NLA e aux judge.
```

### Commento

```text
Non mi è parso poi di vedere (ma potrei essermelo perso) la parte di Soft-prompting con SIPIT analysis. Se non c'è ovviamente aggiungila.
```

Risposta da dare:

```text
fatto, ho aggiunto la parte soft-prompting + SIPIT-style readout. Nel metodo la descrivo come diagnostic separata, e nei results metto sia metriche del soft prompt sia controlli/readout per capire se quei vettori continui sono interpretabili.
```

### Commento

```text
In generale, bisogna dare un bel boost sulla scrittura mentre gli esperimenti vanno avanti. Ti ripeto che non serve fare 100 pagine scritte così. Ne bastano 50, guarda. Ma chiare. Te lo dico perché se il membro della commissione non capisce bene, aver messo tanta carne al fuoco in modo non chiaro può solo nuocere alla tua valutazione. È un peccato perché il lavoro è molto valido e meriterebbe senza dubbio un punteggio pieno. Ma serve descriverlo in maniera precisa.
```

Risposta da dare:

```text
si, ho provato a sistemare proprio questo punto. Ho tolto varie parti che sembravano cronologia di esperimenti e le ho riscritte come metodo/setup/results. Dove una parte era troppo locale o confondeva, l'ho spostata o tagliata.
```

### Commento

```text
Appena hai una nuova versione pingaci pure che ridiamo un check.
```

Risposta da dare:

```text
ok, appena porto questi fix su Overleaf vi pingho con la nuova versione.
```

## Commenti Overleaf

### Commento

```text
riga 770 del latex
evita queste sezioni "gippittierose"
```

Risposta da dare:

```text
prima qui avevo messo una mini sezione per giustificare perche GEPA entrava nella tesi. Rileggendola sembrava piu un meta-discorso che related work vero. L'ho tolta come sezione a parte e ho lasciato solo la parte utile: GEPA come riferimento di hard prompt optimization su cui poi costruisco LatentGEPA.
```

### Commento

```text
righe 910-914
Non serve dire cosa questo capitolo NON mostra. Questa intro deve solo descrivere brevemente cosa proponi.
```

Risposta da dare:

```text
prima l'apertura del cap 3 partiva troppo in negativo, quasi come lista di cose che non faccio. L'ho riscritta dicendo direttamente cosa c'e nel capitolo: Semantic-Fidelity Corpus, LatentGEPA e le diagnostiche sui segnali latenti.
```

### Commento

```text
righe 921-923
anche qui. Usa un linguaggio un pò più formale e descrittivo. Non serve dire cosa la tua tesi non è. Basta essere chiari nel metodo e uno lo capisce solo
```

Risposta da dare:

```text
prima il metodo saltava tra dataset, inversione, NLA e GEPA come se fossero tutti pezzi separati allo stesso livello. Ho provato a renderlo piu lineare: il metodo proposto e' LatentGEPA, mentre dataset, diagnostiche e segnali latenti entrano per il ruolo che hanno dentro quella pipeline.
```

### Commento

```text
riga 1194
non mi fa impazzire il titolo della sezione. Cioè perchè "canonical"? Sarebbe meglio dare un nome al dataset e chiare così la sezione.
```

Risposta da dare:

```text
prima usavo "canonical" come nome interno, pero nel testo poteva sembrare un benchmark gia standard o comunque un nome non motivato. L'ho cambiato in Semantic-Fidelity Corpus e ho aggiornato anche figura, tabelle, caption e claim matrix. Ho lasciato solo label latex interne perche non si vedono nel pdf.
```

### Commento

```text
riga 1195
In generale questa sezione deve essere estesa significativamente. Cioè devi essere più dettagliato su come hai creato il dataset. Mettendo anche degli esempi per far capire al lettore la natura del task. Riporta anche varie stats tipo il numero di istanze, lunghezza media dei dati etc. Serve essere precisi e formali qui
```

Risposta da dare:

```text
prima questa sezione spiegava piu lo scopo del dataset che come era stato costruito. Ho aggiunto blocchi A/B/C, metadata salvati per ogni riga, esempi concreti, split e statistiche. Ho ricontrollato i numeri dagli artifact: 2080 righe totali, blocchi 40/720/1320 e split 962/290/828, piu lunghezze medie.
```

### Commento

```text
righe 1256-1260
non descriverlo qui. Ne parli quando arrivi alle rispettive parti di metodo.
```

Risposta da dare:

```text
prima subito dopo il dataset anticipavo gia embedding inversion, SIPIT, NLA e soft prompt, ma in quel punto il lettore non aveva ancora il contesto. Ho tolto quell'elenco per non caricare troppo la sezione: li descrivo solo il corpus, poi gli usi specifici vengono fuori nelle sezioni giuste.
```

### Commento

```text
riga 1262
non mi piace la nomenclatura "branch". Così sembra tutto troppo compartimentato. usa dei titoli e riferimenti che fanno capire cosa quei risultati indicano. Qui per esempio vogliamo valutare la semantic info preservation. Ne parlerei in questi termini.
```

Risposta da dare:

```text
prima usando "branch" sembrava che fossero esperimenti separati messi uno accanto all'altro. Ho cambiato framing: quando misuro la preservation semantica parlo di diagnostic readouts, mentre quando confronto PPL/NLA/aux dentro GEPA parlo di feedback variants di LatentGEPA.
```

### Commento

```text
riga 1361
MAI citare python file e robe di codice
```

Risposta da dare:

```text
prima in alcuni punti spiegavo il risultato citando nomi di script/cache/file locali. Ho tolto quei riferimenti perche facevano vedere al lettore dettagli della repo invece dell'esperimento. Ora dico solo cosa e' stato generato, cosa e' stato confrontato e perche.
```

### Commento

```text
righe 1372-1376
non capisco il motivo di questa sezione
```

Risposta da dare:

```text
prima nel metodo avevo messo una sezione sulla reproduction boundary dell'embedding inversion. La cosa e' utile, pero li sembrava parte del metodo proposto. L'ho tolta come blocco a parte dal metodo e l'ho spostata nei risultati/setup, dove serve per spiegare perche quella parte resta diagnostica.
```

### Commento

```text
righe 1380-1388
non mi è chiaro nemmeno qui cosa vuoi indicare. Ma su due piedi mi sembra una roba da results, non da metodo
```

Risposta da dare:

```text
prima la parte su Jina mischiava chiamate modello, cache locali e reproduction, e rileggendola non era nemmeno cosi centrale per la tesi. L'ho tolta quasi tutta: nei risultati resta solo il punto importante, cioe che l'embedding inversion e' una diagnostic reproduction e non una baseline paper-level pulita.
```

### Commento

```text
righe 1398-1400
anche qui. Mi sembra tutto troppo sulla "difensiva". Non devi dire cosa NON fai. Semplicemente spiega l'analisi e gli obiettivi ma non in questo tono "diff"
```

Risposta da dare:

```text
prima in alcuni punti spiegavo le scelte in stile "non faccio questo, non faccio quello", quindi veniva fuori troppo difensivo. Ho riscritto quei pezzi come obiettivi: usare readout controllati per valutare semantic fidelity e usare segnali latenti come feedback per il proposer di LatentGEPA.
```

### Commento

```text
riga 1428
Direi che SIPIT è più da related work. Idem per NLA. Nel metodo devi tenere il dettaglio del dataset e poi quello di come modifichi GEPA. Soprattutto usa l'acronimo LatentGEPA il più possibile visto che è il nome del tuo metodo.
```

Risposta da dare:

```text
prima SIPIT, NLA e GEPA comparivano nel metodo quasi allo stesso livello, anche se in realta quello che modifico davvero e' GEPA. Ho spostato la spiegazione piena di SIPIT/NLA nei related/setup/results; nel metodo li richiamo solo quando servono come readout o extra-info. GEPA invece ora e' spiegato nel cap 3 insieme a LatentGEPA, con minibatch proposal, reflection data, proposer e frontier.
```

### Commento

```text
riga 1693
qui non capisco il perchè di questa sezione. Il task di judge eval deve essere introdotto dopo aver parlato del metodo. Magari anche direttamente nella parte di exp setup
```

Risposta da dare:

```text
prima il judge task era un po buttato li e rischiava di sembrare solo setup. Ho lasciato nel metodo solo lo schema minimale del record judge, cosi si capisce cosa ottimizza LatentGEPA. Le cose piu sperimentali, cioe dataset, split, scale e metriche, stanno invece nel cap 4.
```

### Commento

```text
riga 1990
metti sempre le caption ad ogni cosa
```

Risposta da dare:

```text
prima alcuni blocchi lunghi, soprattutto l'esempio JSON PPL+NLA e quello proposer-facing, erano praticamente listing senza una vera caption. Li ho messi come listing con caption e label, cosi il lettore capisce subito cosa sta guardando e posso anche richiamarli nel testo.
```

### Commento

```text
righe 3136 - 3162
no ref a codice e script
```

Risposta da dare:

```text
prima nei risultati dell'embedding inversion erano rimasti dettagli troppo locali su file/cache/script e sulla provenance Jina. Ho tolto quella spiegazione dettagliata perche non aggiungeva molto alla tesi e rischiava di distrarre. Ora resta solo il risultato utile: la reproduction e' diagnostica, non una reproduction paper-level completa.
```

## Commenti aggiuntivi utili da lasciare su Overleaf

Questi non corrispondono per forza a un singolo commento originale, ma possono
servire per notificare modifiche fatte dopo il check del relatore. Le righe sono
quelle dell'attuale `thesis/latex/main.tex` locale.

### Riga 981

```text
ho aggiornato anche la figura del metodo: non compare piu "Canonical corpus", ma Semantic-Fidelity Corpus, cosi il nome resta coerente anche li.
```

### Riga 1160

```text
ho rinominato la sezione in Semantic-Fidelity Corpus. Mi sembra piu chiaro perche dice direttamente a cosa serve il corpus, cioe testare la fedelta semantica delle inversioni/verbalizzazioni.
```

### Riga 1876

```text
ho aggiornato anche la tabella dei dataset: ora anche li si chiama Semantic-Fidelity Corpus, non piu canonical.
```

### Riga 2636

```text
ho rinominato anche la sezione risultati in Semantic-Fidelity Corpus Validation, cosi non sembra piu riferita a un dataset "canonical" interno.
```

### Riga 2713

```text
ho tagliato il dettaglio Jina/provenance nei risultati. Era troppo locale e non centrale; ora resta solo la conclusione utile, cioe che l'embedding inversion qui e' diagnostica e non una reproduction paper-level completa.
```

### Riga 3474

```text
ho aggiornato anche la claim matrix: ora parla di Semantic-Fidelity Corpus, coerente con il resto della tesi.
```

### Riga 3594

```text
ho corretto il nome nei ringraziamenti: Gianluca Moro invece di Andrea Moro.
```
