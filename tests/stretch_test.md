# Comprehensive Stretch Test — All Phases

A single test corpus designed to exercise every capability across Phases 1, 2, and 3.
Ingest all six documents in order, then run all question groups.

---

## Corpus

### Document 1 — `biotech_company`

```
Helix Therapeutics is a biotechnology company founded in 2017 by Dr. Elena Vasquez
and Dr. James Osei in Boston, Massachusetts. The company focuses on CRISPR-based gene
editing therapies for rare genetic disorders.

Dr. Elena Vasquez serves as CEO and holds a PhD in molecular biology from Harvard
University. She previously led the gene editing laboratory at the Broad Institute.
Dr. James Osei is Chief Scientific Officer. He earned his doctorate at MIT and spent
eight years as a senior researcher at Novartis before co-founding Helix.

Helix Therapeutics raised $200 million in Series C funding in March 2024, led by
Andreessen Horowitz. The round also included participation from Flagship Pioneering
and ARCH Venture Partners. The company employs 320 scientists and engineers across
its Boston headquarters and a satellite office in San Diego, California.

Helix's lead programme, HX-001, targets a rare form of muscular dystrophy affecting
approximately 40,000 patients globally. The drug candidate entered Phase 2 clinical
trials in January 2024 after achieving positive Phase 1 safety data.
```

---

### Document 2 — `biotech_partnership`

```
Helix Therapeutics announced a research collaboration with Genome Dynamics Institute
in September 2023. Under the agreement, Genome Dynamics will provide computational
protein-folding models to accelerate Helix's target identification pipeline.

The collaboration is overseen by Dr. Marcus Chen, Helix's VP of Research Partnerships,
and Professor Yuki Tanaka, Director of Computational Biology at Genome Dynamics
Institute. Professor Tanaka is also a visiting professor at MIT, where she collaborates
with Dr. Osei's former laboratory.

Genome Dynamics Institute is headquartered in Cambridge, UK and was founded in 2015
by a consortium of European universities. It receives annual funding from the Wellcome
Trust and the European Research Council.
```

---

### Document 3 — `competitor_profile`

```
Cipher Genomics is a direct competitor to Helix Therapeutics in the CRISPR therapy
market. Cipher was founded in San Francisco, California in 2018 by Dr. Robert Kim,
a former postdoctoral researcher at the Broad Institute.

Cipher's lead asset, CG-202, also targets muscular dystrophy but uses a different
delivery mechanism than HX-001. Cipher Genomics raised $180 million in Series B
funding in 2023, led by Sequoia Capital.

Dr. Elena Vasquez commented in a 2024 industry conference that she views Cipher
Genomics as "the most technically credible competitor in this space." Dr. Robert Kim
and Dr. Elena Vasquez served on the same doctoral thesis committee at Harvard in 2012.
```

---

### Document 4 — `contradicting_facts`

```
Industry analysts at BioCapital Research published a report in April 2024 stating that
Helix Therapeutics was founded in 2019, not 2017 as the company claims. The report
alleges that the company restructured its legal entity in 2019 and backdated its
founding narrative for commercial purposes.

Additionally, the BioCapital report states that Helix's Series C round raised only
$175 million, not $200 million as announced by the company. The report cites
discrepancies in SEC filing data.

The BioCapital Research analysis also notes that HX-001 is targeting a patient
population of approximately 35,000, disagreeing with Helix's stated figure of 40,000.
```

---

### Document 5 — `unrelated_domain_triggers_ontology`

```
The international basketball federation announced updated rules for the 2025 season.
The three-point line will be extended by 30 centimetres in all FIBA-sanctioned
competitions. Teams may now carry 15 players on active rosters, up from 12.

The Golden State Warriors signed point guard Darius Cole to a four-year contract
extension worth $120 million. Cole averaged 24.3 points per game last season and
was named to the All-NBA First Team for the third consecutive year.

Historically, the Boston Celtics hold the record for the most NBA championships with
18 titles. The Los Angeles Lakers are second with 17 titles.
```

---

### Document 6 — `cross_domain_connection`

```
Dr. Marcus Chen, who manages the Helix-Genome Dynamics collaboration, was previously
the Chief Strategy Officer at a sports analytics firm called StatEdge. StatEdge
developed predictive models used by several NBA teams, including the Golden State
Warriors, to optimise player contract valuations.

Dr. Chen left StatEdge in 2022 to join Helix Therapeutics. His hire was announced
by CEO Dr. Elena Vasquez, who noted that data-driven decision making is increasingly
central to Helix's research pipeline strategy.
```

---

## Questions by Phase and Capability

---

### Phase 1 — Core Vector Retrieval

**P1-A.** What is Helix Therapeutics' lead drug programme and what disease does it target?
> Expected: HX-001, rare muscular dystrophy affecting ~40,000 patients globally, Phase 2 clinical trials since January 2024.

**P1-B.** What is Dr. James Osei's educational background?
> Expected: Doctorate at MIT, 8 years at Novartis as senior researcher, co-founder of Helix.

**P1-C.** Who are the investors in Helix Therapeutics' Series C round?
> Expected: Andreessen Horowitz (lead), Flagship Pioneering, ARCH Venture Partners. $200M raised March 2024.

**P1-D.** Describe the Helix–Genome Dynamics collaboration.
> Expected: Research collaboration Sep 2023, computational protein-folding models for target identification, overseen by Dr. Marcus Chen and Professor Yuki Tanaka.

**P1-E.** What is Dr. Elena Vasquez's opinion of Cipher Genomics?
> Expected: "Most technically credible competitor in this space" (cited from 2024 industry conference).

---

### Phase 2 — Entity Resolution

**P2-ER-A.** What MIT connection does Dr. Osei have to the Genome Dynamics collaboration?
> Tests that "Dr. James Osei" (doc 1) and "Dr. Osei's former laboratory" (doc 2) resolve to one canonical entity.
> Expected: Dr. James Osei did his doctorate at MIT; Professor Tanaka is a visiting professor there and collaborates with his former lab.

**P2-ER-B.** Who is the CEO of Helix Therapeutics?
> Tests that "Dr. Elena Vasquez" (doc 1), "Elena Vasquez" (doc 3), and "CEO Dr. Elena Vasquez" (doc 6) resolve to one canonical entity.
> Expected: Dr. Elena Vasquez is CEO and co-founder.

**P2-ER-C.** What is the academic relationship between Dr. Vasquez and Dr. Robert Kim?
> Tests cross-document entity linking.
> Expected: Both served on the same doctoral thesis committee at Harvard in 2012 (doc 3).

---

### Phase 2 — Graph Traversal (Relational Queries)

**P2-G-A.** Who does Dr. Marcus Chen report to, and how is he connected to the Genome Dynamics Institute?
> Expected: Reports to CEO Dr. Elena Vasquez (inferred from org structure); manages the Helix–Genome Dynamics collaboration (doc 2 + doc 6).

**P2-G-B.** Who collaborates with Professor Yuki Tanaka?
> Expected: Works with Dr. Osei's former MIT lab (doc 2); managed by Dr. Marcus Chen on the Helix collaboration.

**P2-G-C.** What is the connection between Dr. Robert Kim and the Broad Institute?
> Expected: Dr. Robert Kim was a postdoctoral researcher at the Broad Institute; Dr. Elena Vasquez previously led the gene editing lab there — both have Broad Institute connections.

---

### Phase 2 — Multi-hop Traversal

**P2-MH-A.** Is there a connection between Golden State Warriors and Helix Therapeutics?
> Tests 3-hop traversal: Warriors → StatEdge → Dr. Marcus Chen → Helix Therapeutics.
> Expected: Yes — Dr. Marcus Chen, now at Helix, previously worked at StatEdge which served the Warriors.

**P2-MH-B.** How is Genome Dynamics Institute connected to muscular dystrophy research?
> Tests 2-hop: Genome Dynamics → collaborates with → Helix → HX-001 targets muscular dystrophy.
> Expected: Through the Helix collaboration — Genome Dynamics provides protein-folding models to accelerate Helix's HX-001 (muscular dystrophy) pipeline.

---

### Phase 2 — Conflict Detection

**P2-C-A.** When was Helix Therapeutics founded?
> Expected: CONFLICT — doc 1 says 2017 (company's own claim), doc 4 says 2019 (BioCapital Research report). Both versions should be presented with sources.

**P2-C-B.** How much did Helix raise in its Series C?
> Expected: CONFLICT — doc 1 says $200M, doc 4 says $175M. Both should surface.

**P2-C-C.** How large is the patient population HX-001 addresses?
> Expected: CONFLICT — doc 1 says ~40,000, doc 4 says ~35,000. Both should surface.

---

### Phase 3 — Query Planning and Decomposition

**P3-QP-A.** Compare Helix Therapeutics and Cipher Genomics: who has raised more funding, and how do their programmes differ?
> Tests query decomposition into sub-questions: (1) funding comparison, (2) programme comparison.
> Expected: Helix raised $200M (conflicted: $175M), Cipher raised $180M. Both target muscular dystrophy but HX-001 and CG-202 use different delivery mechanisms.

**P3-QP-B.** Who are all the people connected to MIT, and what are their roles?
> Tests aggregation intent + parallel retrieval.
> Expected: Dr. James Osei (doctorate, co-founder of Helix), Professor Yuki Tanaka (visiting professor at MIT, Genome Dynamics Director).

---

### Phase 3 — Reranking Quality

**P3-RK-A.** What clinical evidence exists for HX-001?
> With reranking enabled, the chunk containing Phase 1 safety data and Phase 2 trial start should rank highest, ahead of generic company description chunks.
> Expected: Positive Phase 1 safety data; entered Phase 2 clinical trials January 2024.

**P3-RK-B.** Who founded the companies mentioned in the knowledge base?
> Tests breadth — requires multiple relevant chunks from different documents.
> Expected: Helix Therapeutics (Dr. Elena Vasquez + Dr. James Osei, 2017/disputed 2019), Genome Dynamics Institute (European university consortium, 2015), Cipher Genomics (Dr. Robert Kim, 2018).

---

### Phase 3 — Authority Scoring

**P3-A-A.** What year was Helix founded, and which source is more credible?
> With authority scoring enabled: the company's own profile (doc 1) should be compared against the analyst report (doc 4). If the analyst report has no corroborating sources, it should score lower authority.
> Expected: The answer should note that doc 1 (primary source) states 2017, while BioCapital Research (doc 4) states 2019, and may note the credibility differential.

---

### Phase 3 — Ontology Evolution

**P3-OE-A.** After ingesting document 5 (basketball), run the ontology evolution engine.
> Expected: Concepts like "Golden State Warriors", "Boston Celtics", "Los Angeles Lakers" should cluster together → proposed type: `SportsTeam` or `BasketballTeam`.
> Concepts like "point guard", "three-point line" may form a `BasketballRule` or `SportsStatistic` cluster.

```bash
# Run ontology evolution check
python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from semantic_vault.storage import VectorStore, EntityIndex
from semantic_vault.ontology_engine import OntologyEngine, SchemaRegistry
from semantic_vault.config import settings
vs = VectorStore(url=settings.qdrant_url)
engine = OntologyEngine(EntityIndex(vs), SchemaRegistry(db_path=settings.db_path))
print('New types:', engine.run_evolution())
"
```

---

## How to run

### Via CLI

```bash
source .venv/bin/activate

# Ingest all six documents
python main.py ingest --file tests/stretch_test.md --name "stretch_combined"

# Or individually (paste each corpus section into the UI, or use --file with split files)

# Run representative questions
python main.py query "When was Helix Therapeutics founded?"
python main.py query "Who does Dr. Marcus Chen report to?"
python main.py query "Is there a connection between the Golden State Warriors and Helix Therapeutics?"
python main.py query "Compare Helix and Cipher Genomics funding and programmes"
```

### Via Gradio UI

```bash
python app.py  # opens http://127.0.0.1:7860
```

Paste each document text into **Add Knowledge**, give it the name shown, click **Add**. Then ask questions in **Ask Questions**.

---

## Scoring guide

| Mark | Meaning |
|---|---|
| ✅ | Correct, cited, complete |
| ⚠️ | Correct but missing detail or citation |
| ❌ | Wrong or hallucinated |
| 🔀 | Conflict correctly surfaced with both sources |
| 🔗 | Graph traversal used (visible in graph_facts field) |
| 🧩 | Multi-hop reasoning correctly followed |
| 🏷️ | Entity resolution merged aliases correctly |
| 📊 | Reranking visibly improved result order |

A full-phase system should score ✅ on all P1 questions, ✅/🔀 on P2 conflict questions, 🔗/🧩 on P2 graph questions, and show improved comprehensiveness on P3 questions versus Phase 1 alone.
