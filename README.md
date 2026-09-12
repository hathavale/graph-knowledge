# graph-knowledge

Builds a knowledge graph from support tickets so a future ticket can be routed
to the right people — **who to notify, with what context, and who else to keep
in the loop.**

```
"Checkout is returning 502s from the payment gateway. @raj picked it up
 but said to ask Priya, she wrote the retry logic."
```

becomes

```cypher
(:Ticket {id:"SUP-1042"})-[:ABOUT]->(:Topic {name:"payment gateway"})
(:Person {name:"Raj"})-[:PARTICIPATED_IN {role:"assignee"}]->(:Ticket)
(:Ticket)-[:MENTIONS {role:"expert", excerpt:"ask Priya, she wrote the retry logic",
                      confidence:0.8}]->(:Person {name:"Priya"})
(:Alias {value:"@raj"})-[:ALIAS_OF]->(:Person {name:"Raj"})
```

Ask it who knows about the payment gateway and it answers with the evidence
that nominated them:

```
Priya: 1 expert mention, 0 tickets worked
    "ask Priya, she wrote the retry logic"
Raj:   0 expert mentions, 1 ticket worked
```

## Quickstart

```bash
make install
make up                       # start Neo4j
make test                     # 46 tests; 54 with Neo4j running
make eval-oracle              # harness self-test, free
export ANTHROPIC_API_KEY=...
python -m graph_knowledge.cli "…ticket text…" --ticket-id SUP-1042
```

## Two layers, separated by provenance

The central design decision. Facts about **who someone is** and facts about
**what they know** have very different reliability, and mixing them is how a
router ends up escalating to a manager who doesn't exist.

| | Identity & org | Expertise & involvement |
|---|---|---|
| `REPORTS_TO`, `MEMBER_OF`, `HAS_FUNCTION` | `PARTICIPATED_IN`, `MENTIONS`, `ABOUT` |
| Authoritative when imported from a directory | Only obtainable from tickets |
| Currently **inferred** — treat as a hint | The signal the product rests on |

Every org fact carries `source` (`imported` / `inferred` / `derived`) and a
confidence. An import overrides an inference; **an inference never downgrades
an import** — both directions are tested. When you get a directory export,
import it and the inferred rows are superseded without a migration.

Tickets are a poor source for reporting lines and the only source for "who
actually knows about the retry path".

## Participation vs mention

The distinction the routing depends on:

- **Participation** — someone acted on the ticket. Behavioural evidence.
- **Mention** — someone was referenced. `expert`, `escalation`, `approver`,
  `affected`. A colleague writing *"ask Priya, she wrote the retry logic"* is a
  deliberate claim about expertise, made in context — a stronger signal than
  many tickets touched.

`experts_for_topic()` returns both counts separately rather than folding them
into one opaque score; ranking policy belongs in the routing layer.

## Evidence is not optional

Every inferred fact carries the ticket it came from and the excerpt supporting
it. A routing decision that cannot say *why* someone was chosen is not
reviewable, and an unreviewable notifier gets muted. The excerpt is also what a
draft notification quotes back.

## A vocabulary that evolves without drifting

The model is meant to grow with LLM input. Unconstrained, that yields
`payment gateway`, `payment-gateway` and `billing gateway` as three unrelated
topics within a week, and every routing query silently misses two thirds of its
evidence.

So `vocabulary.py` holds a controlled vocabulary *in the graph*:

- Extraction is constrained to **canonical** terms, which are what the prompt
  lists.
- Anything new comes back as a **proposal**, not a written fact.
- A proposal is promoted once enough **distinct tickets** support it — one
  ticket saying a thing five times is still one piece of evidence.
- `alias()` folds a duplicate spelling into an existing term without rewriting
  history; `reject()` is permanent, so a curation decision is not undone by
  repetition.

A small, stable canonical list is also what keeps the system prompt cacheable.

## Layout

```
src/graph_knowledge/
├── models.py        # Person, Ticket, Topic, Mention, OrgFact, Evidence
├── vocabulary.py    # canonical / proposed / rejected terms + promotion
├── pipeline.py      # extract → observe terms → persist
├── cli.py
├── extraction/      # Extractor protocol + LLM extractor (structured outputs)
└── store/
    ├── _cypher.py   # statements shared by both backends
    ├── neo4j_store.py
    └── embedded_store.py
evals/               # 24 ticket cases, per-fact grader, bounded harness
```

## Eval

Per atomic fact — `person`, `alias`, `topic`, `participation`, `mention`, `org`
— so a regression is attributable. 101 gold facts across 24 cases, eight
categories of three.

```bash
make eval-oracle    # must be 1.00 — gold replayed
make eval-null      # must be 0.00 — grader isn't lenient
make eval-llm       # costs money
```

**20 of 24 cases assert that no org fact should be produced.** Inventing
structure is the failure that messages the wrong person, so the set pushes
against it and the runner prints invented org facts separately.

| slice | cases | facts | 1 rep | 2 reps |
|---|---|---|---|---|
| full | 24 | 101 | ±10 | ±7 |
| train | 16 | 67 | ±12 | ±9 |
| test | 8 | 34 | ±17 | ±12 |

Iterate on `--slice train`, confirm on `--slice test`, and treat a test move
under 12 points as noise. Every category appears on both sides of the split.

## Status

Working and tested: the schema, both backends, the vocabulary gate, extraction
mapping and validation, and the eval harness. **The live API call is still
unverified** — the request shape and generated JSON schema are checked locally,
but no request has reached the API from this repo.

The cases are author-written. Replacing them with real tickets — and using
*who actually participated* as gold for routing — is the biggest upgrade
available, and needs no hand-annotation.

## Next

1. **Retrieval and context packs.** The token argument only pays off when a
   query returns a small ranked subgraph instead of ticket text. Not built yet.
2. **The routing decision** — candidates, ranking, loop-in set, drafted
   message. Recommend-only until precision is measured on real tickets.
3. **Directory import.** The two-layer split is in place and tested; the
   importer is not written.
4. **Recency decay** on expertise. Counts are currently flat, so someone who
   worked a topic two years ago ranks alongside someone who worked it last week.
