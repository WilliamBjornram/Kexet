# Williams anteckningar till David
- Jag skriver "As meantioned in this study [16] time is a good way..." studierna jag läser skriver dock ungefär så här "As meantioned by Brown at el., 2018 time is a good way..." Hur borde vi skriva? 
- Så här skriver dom i en studie om MCCFR: Figure 1 shows the results of all four algorithms on all four domains, plotting approximation quality
as a function of the number of nodes of the game tree the algorithm touched while computing.
Nodes touched is an implementation-independent measure of computation; however, the results are
nearly identical if total wall-clock time is used instead. Since the algorithms take radically different
amounts of time per iteration, this comparison directly answers if the sampling variants’ lower cost
per iteration outweighs the required increase in the number of iterations. Furthermore, for any
fixed game (and degree of confidence that the bound holds), the algorithms’ average overall regret
is falling at the same rate, O(1/
√
T), meaning that only their short-term rather than asymptotic
performance will differ. Källa: https://mlanctot.info/files/papers/nips09mccfr.pdf
- I en annan studie som upptäckte Deep CFR så jämför dom CFR och Deep med "nodes touches" men eftersom att datorn måste göra beräkningar för neurala nätverket så kan Deep CFR ta längre tid men ha en lägre "nodes touched" alltså tycker jag att nodes touhed blir lite värdelös, eller åtminstone missar att ta med en dimension. Citat: "The figure shows that Deep CFR asymptotically reaches a
similar level of exploitability as the abstraction that uses 3.6
million clusters, but converges substantially faster. Although
Deep CFR is more efficient in terms of nodes touched, neural network inference and training requires considerable
overhead that tabular CFR avoids. However, Deep CFR
does not require advanced domain knowledge. We show
Deep CFR performance for 10,000 CFR traversals per step.
Using more traversals per step is less sample efficient and
requires greater neural network training time but requires
fewer CFR steps." Källa: https://arxiv.org/pdf/1811.00164
- Om vi skulle kunna typ låsa eller kontrollera exakt hur mycket datorkraft programmen får så skulle tiden vara mer relevant att jämföra, om vi kör varje programm typ tre gånger under olika tillfällen så kanske våra resultat skulle kunna bli bättre? Hur ska vi jämföra CFR varianterna? 1. Jämföra deras NashConv efter en viss tid eller tiden det tar att nå en viss NashConv (Beskrivning av NashConv: https://arxiv.org/pdf/1711.00832). 2. Hur mycket datorkraft som används. 3. För MCCFR och Deep CFR så finns det flera olika inställningar, dessa kan vi exprimentera med alternativt så bestämmer vi oss för en specifik inställning.
- ASW game är en bra beskrivning men vi tar inte upp motivering till designen på spelet, detta känns relevant att göra men kanske inte under denna rubrik, men vart borde detta göras?

# Davids anteckningar till William
- Jag är nöjd med sektionerna för RL och MARL som de ser ut nu, du kanske kan kolla om du vill lägga till något? Jag har i stort sett inte tagit bort något, utan endast omformulerat.
- Jag tycker att vi ska använda kursivt i stället för fetstilt när vi vill betona något ord. Vad tycker du?

# Anteckningar till rapporten i allmänhet
- Vad för figurer och tabeller vill vi ha med?
- Vad vill ha för statistik?
- Hur mäter vi våra resultat?
- Ska vi skriva "choke points" eller "chokepoints"? Se över ord så vi skriver samma.
- Vilka källor vi ska ha? Mer studier, böcker osv i metod, teori?
# Anteckningar till rapporten - Introduction
- Intro till hur ubåtar använts i krig, hur tatiken har varit/utvecklas, varför det är relvant för Sverige idag, hur AI har utvecklas så det kan användas för tatiken, hur projektet binder samma all dessa.
# Anteckningar till rapporten - Background
- Här tar vi avstamp i literaturen, skriver hur andra har gjort liknade saker tidigare.
# Anteckningar till rapporten - ASW Game
- Hur spelet fungerar, endast.
# Anteckningar till rapporten - Theory
- Beskriver: RL, MARL, CFR, MCCFR, DeepCFR. Borde kanske beskriva olika utvärdering sätt och hur dessa fungerar?
# Anteckningar till rapporten - Method
- Under metod skulle vi kunna beskriva: Varför spelet ser ut som det gör, hur vi ställde in CFR, MCCFR, Deep CFR, alltså hur många iterationer och sådär. Hur vi utvärderar dom olika och varför vi gör på detta viset.

# Tidigare rapporter
https://cdn.aaai.org/ojs/10051/10051-13-13579-1-2-20201228.pdf

https://www.politesi.polimi.it/retrieve/a81cb05d-2727-616b-e053-1605fe0a889a/MSc_Thesis-FINALE.pdf

