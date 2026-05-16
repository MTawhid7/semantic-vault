# Phase 2 Stretch Test — Corpus & Questions

A manually designed test corpus that exercises every Phase 2 capability:
entity resolution (same entity under multiple names), multi-hop graph traversal,
relationship extraction, conflict detection, and the interplay between vector
and graph retrieval.

Ingest all five documents **in order**, then run the questions.

---

## Corpus

### Document 1 — Company Profile  `(name: "nexus_ai_profile")`

```
Nexus AI is a machine learning infrastructure company founded in 2018 and headquartered
in San Francisco, California. The company was co-founded by Dr. Sarah Chen and Marcus
Webb after they left their research positions at Stanford University. Nexus AI specialises
in large-scale model training pipelines and has raised over $120 million in venture
funding since its inception.

Dr. Sarah Chen serves as Chief Executive Officer. She holds a PhD in Computer Science
from MIT and previously led the distributed systems group at Stanford. Marcus Webb is
Chief Technology Officer and oversees the engineering organisation of roughly 200
engineers. Webb earned his undergraduate degree at Caltech and his doctorate at
Carnegie Mellon University.

The company's flagship product, TrainFlow, is a distributed training orchestration
platform used by over 300 enterprise clients. Nexus AI's primary investors include
Horizon Ventures and Meridian Capital, both based in Silicon Valley.
```

---

### Document 2 — Project Announcement  `(name: "nexus_quantumedge_partnership")`

```
Nexus AI announced a strategic partnership with QuantumEdge Systems in March 2024.
Under the agreement, Nexus AI's TrainFlow platform will integrate directly with
QuantumEdge's hardware accelerator lineup, beginning with the QE-900 chip series.

QuantumEdge Systems was founded in 2020 by Robert Tanaka, a former Intel Fellow,
and Dr. Aisha Okonkwo, who previously directed the semiconductor research lab at
MIT Lincoln Laboratory. The company is headquartered in Austin, Texas, and employs
around 150 engineers.

The integration project is led jointly by Priya Nair, Nexus AI's VP of Partnerships,
and James Okafor, QuantumEdge's Head of Developer Ecosystems. Sarah Chen commented
that the partnership "represents the most significant hardware collaboration in Nexus
AI's history." Robert Tanaka called TrainFlow "the obvious choice for high-throughput
training workloads on next-generation silicon."
```

---

### Document 3 — Research Collaboration  `(name: "mit_nexus_research")`

```
The MIT Computational Intelligence Lab announced a two-year research collaboration
with Nexus AI in January 2024. The programme is directed by Professor Elena Vasquez,
who leads the lab's model efficiency research group. On the Nexus AI side, the
collaboration is sponsored by Dr. S. Chen and managed day-to-day by Dr. Rohan Mehta,
Nexus AI's Director of Research.

The research agenda focuses on three themes: sparse model architectures, efficient
fine-tuning methods, and hardware-aware neural architecture search. The last theme
overlaps directly with the technology developed by QuantumEdge Systems, and the
collaboration agreement explicitly allows QuantumEdge to participate as an
observer in the architecture search workstream.

Professor Vasquez noted that this is the third consecutive year MIT has partnered
with an AI infrastructure company, following similar arrangements with Google DeepMind
in 2022 and Anthropic in 2023.
```

---

### Document 4 — Internal Org Update  `(name: "nexus_org_update_q1_2024")`

```
Nexus AI completed several leadership changes in Q1 2024. Dr. Rohan Mehta was promoted
from Senior Research Scientist to Director of Research, reporting directly to CTO Webb.
Priya Nair was elevated from Senior Director to Vice President of Partnerships,
reporting to CEO Chen.

James Okafor joined Nexus AI as Head of Strategic Accounts in February 2024 after
three years at QuantumEdge Systems. He reports to Priya Nair. The company also hired
Dr. Yuki Tanaka as VP of Product, effective March 2024. Dr. Yuki Tanaka is not related
to QuantumEdge co-founder Robert Tanaka despite sharing a family name.

Marcus Webb stated in the company all-hands that Nexus AI plans to expand its
Austin, Texas office to support the growing QuantumEdge partnership. The Austin office
currently houses 12 employees; the target headcount by end of 2024 is 60.
```

---

### Document 5 — Funding Announcement  `(name: "nexus_series_d")`

```
Nexus AI closed its Series D funding round in April 2024, raising $85 million led by
Horizon Ventures. This brings total funding to over $200 million. The round also
included participation from Meridian Capital and a new investor, Pacific Ridge Partners.

The company was founded in 2019 by Sarah Chen and Marcus Webb. Pacific Ridge Partners
is based in Seattle and made its first enterprise AI investment with this round.
Partner at Pacific Ridge, Daniel Lim, will join the Nexus AI board of directors.

CEO Sarah Chen said the funding will accelerate the build-out of the Austin office
and support the ongoing MIT research collaboration. CTO Marcus Webb added that a
portion of the funds will be used to deepen the integration with QuantumEdge Systems.
```

---

## Test Questions

Questions are grouped by the capability they primarily exercise.
Expected behaviour is described to help assess answer quality.

---

### Group A — Phase 1 Baseline (pure vector retrieval)

These should work well in Phase 1 and remain correct in Phase 2.

**A1.** What does Nexus AI's TrainFlow product do?  
*Expected: distributed training orchestration platform, used by 300+ enterprise clients.*

**A2.** Where is Nexus AI headquartered?  
*Expected: San Francisco, California.*

**A3.** What are the three research themes in the MIT collaboration?  
*Expected: sparse model architectures, efficient fine-tuning, hardware-aware neural architecture search.*

**A4.** How much has Nexus AI raised in total funding?  
*Expected: over $200 million (doc 5 supersedes doc 1's $120M figure — answer should prefer the more recent number and ideally note the discrepancy).*

---

### Group B — Single-hop graph retrieval

These require the graph to answer correctly. Vector search alone would give vague or incomplete answers.

**B1.** Who does Priya Nair report to?  
*Expected: CEO Sarah Chen / CEO Chen. Tests REPORTS_TO edge.*

**B2.** Who does Dr. Rohan Mehta report to?  
*Expected: CTO Marcus Webb. Tests REPORTS_TO edge.*

**B3.** Who are the co-founders of QuantumEdge Systems?  
*Expected: Robert Tanaka and Dr. Aisha Okonkwo.*

**B4.** Which investors participated in the Nexus AI Series D?  
*Expected: Horizon Ventures (lead), Meridian Capital, Pacific Ridge Partners.*

**B5.** What company did James Okafor work at before joining Nexus AI?  
*Expected: QuantumEdge Systems.*

---

### Group C — Multi-hop graph traversal

These require following 2–3 relationship edges. Pure vector search will struggle.

**C1.** Is there a connection between Dr. Rohan Mehta and QuantumEdge Systems?  
*Expected: Yes — Mehta reports to Webb (CTO), Webb co-founded Nexus AI, Nexus AI partnered with QuantumEdge. Also: the MIT collaboration explicitly allows QuantumEdge to observe Mehta's workstream.*

**C2.** How is Daniel Lim connected to Marcus Webb?  
*Expected: Daniel Lim joined the Nexus AI board → Nexus AI's CTO is Marcus Webb. Two hops.*

**C3.** Which MIT researcher has an indirect connection to QuantumEdge Systems?  
*Expected: Professor Elena Vasquez — she directs the MIT collaboration that includes QuantumEdge as an observer.*

**C4.** Who are all the people connected to the Austin, Texas office expansion?  
*Expected: Marcus Webb (announced it), James Okafor (was hired to support it), Sarah Chen (mentioned it in Series D context).*

---

### Group D — Entity resolution

These test whether the system correctly recognises the same entity under multiple surface forms.

**D1.** What role does Sarah Chen hold at Nexus AI?  
*Expected: CEO. The system must merge "Dr. Sarah Chen" (doc 1), "Sarah Chen" (doc 2), "Dr. S. Chen" (doc 3), and "CEO Sarah Chen" (doc 5) into one canonical entity.*

**D2.** Who manages the MIT collaboration on the Nexus AI side?  
*Expected: Dr. Rohan Mehta day-to-day, sponsored by Dr. S. Chen (i.e. Sarah Chen). Tests alias resolution for both.*

**D3.** Are Robert Tanaka and Dr. Yuki Tanaka the same person?  
*Expected: No — doc 4 explicitly states they are not related despite sharing a family name. The graph should have two separate canonical nodes.*

**D4.** What is Marcus Webb's educational background?  
*Expected: Undergraduate at Caltech, doctorate at Carnegie Mellon. Tests that "Marcus Webb", "Webb", and "CTO Webb" all resolve to one node.*

---

### Group E — Conflict detection

These involve facts where two documents give contradictory information.

**E1.** In what year was Nexus AI founded?  
*Expected: The system should surface the conflict — doc 1 says 2018, doc 5 says 2019 — and present both with their sources rather than picking one silently.*

**E2.** What is Nexus AI's total funding?  
*Expected: Doc 1 says $120M raised "since inception"; doc 5 says "over $200M total". The system should explain that the Series D raised an additional $85M bringing the total above $200M — these are not contradictory if read carefully, but a naive system might report $120M.*

---

### Group F — Entity-type filtering

These use the `--filter-type` CLI flag or the `entity_type_filter` parameter.

**F1.** (filter: Organization) Which organisations are mentioned in the knowledge base?  
*Expected: Nexus AI, QuantumEdge Systems, MIT (MIT Computational Intelligence Lab), Horizon Ventures, Meridian Capital, Pacific Ridge Partners, Stanford University, Google DeepMind, Anthropic, Intel, MIT Lincoln Laboratory, Caltech, Carnegie Mellon University.*

**F2.** (filter: Person) Who are all the named individuals?  
*Expected: Sarah Chen, Marcus Webb, Robert Tanaka, Aisha Okonkwo, Priya Nair, James Okafor, Rohan Mehta, Elena Vasquez, Yuki Tanaka, Daniel Lim.*

**F3.** (filter: Location) What locations are mentioned?  
*Expected: San Francisco, Austin Texas, Silicon Valley, Seattle, Stanford, MIT (Cambridge).*

---

## How to Run

### Via CLI

```bash
cd "semantic-vault"
source .venv/bin/activate

# Ingest all five documents
python main.py ingest --file tests/phase2_stretch_test.md --name "stretch_test_full"

# Or ingest each section separately by pasting the text:
python main.py ingest "Nexus AI is a machine learning infrastructure company..." --name "nexus_ai_profile"

# Query examples
python main.py query "Who does Priya Nair report to?"
python main.py query "Is there a connection between Dr. Rohan Mehta and QuantumEdge?"
python main.py query "In what year was Nexus AI founded?"
python main.py query "Are Robert Tanaka and Yuki Tanaka the same person?"

# With entity-type filter
python main.py query "Which organisations are mentioned?" --filter-type Organization
python main.py query "Who are all the named people?" --filter-type Person
```

### Via Gradio UI

1. `python app.py` → open `http://127.0.0.1:7860`
2. Paste each document text into the **Add Knowledge** panel, give it the name shown above, click **Add**
3. Ask the questions in the **Ask Questions** panel

---

## Scoring Guide

| Score | Meaning |
|---|---|
| ✅ Full credit | Correct answer with sources cited |
| ⚠️ Partial | Correct but missing entity aliases or second-hop facts |
| ❌ Wrong | Hallucinated or factually incorrect |
| 🔀 Conflict surfaced | System correctly flags the 2018 vs 2019 founding year discrepancy |
| 🔗 Graph used | Answer explicitly references graph traversal (visible in `graph_facts` field) |

A strong Phase 2 result scores ✅ on all Group A–B questions, ⚠️–✅ on Group C–D,
and surfaces 🔀 on E1. Group F depends entirely on extraction quality.
