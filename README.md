# graph-knowledge

Build a knowledge graph of people, places and activities from free text.

```
"Mary went to the bank. She withdrew some money."
```

becomes

```cypher
(:Person {name:"Mary", gender:"Female"})-[:TRAVEL {datetime:null}]->(:Place {type:"Bank"})
(:Person {name:"Mary", gender:"Female"})-[:PERFORMED]->
    (:Activity {type:"withdraw money", datetime:null})-[:AT]->(:Place {type:"Bank"})
(:Activity)-[:HAS_ATTRIBUTE]->(:Attribute {name:"amount", value:null})
```

Both statements point at the **same** bank node, and `"She"` is what supplies
Mary's gender.

## Quickstart

```bash
make install          # pip install -e '.[dev,embedded]'
make up               # start Neo4j (waits until healthy)
make test             # run the suite
open http://localhost:7474    # Neo4j Browser: user neo4j, password password
```

No Docker? The embedded backend needs no server at all:

```bash
python -m graph_knowledge.cli "Mary went to the bank. She withdrew some money." \
    --backend embedded --path ./data/graph
make test-embedded
```

With the LLM extractor (needs `ANTHROPIC_API_KEY`, or an `ant auth login` profile):

```bash
python -m graph_knowledge.cli "Mary went to the bank. She withdrew some money." \
    --extractor llm
```

## Why activities are nodes, not edges

The obvious modelling is `(:Person)-[:activity {attributes:{...}}]->(:Place)`.
That does not work, for two reasons that apply to every property-graph engine:

1. **No nested map properties.** Property values must be primitives or arrays
   of primitives, so `attributes: {entity:..., property:...}` cannot be stored
   on an edge.
2. **Edges cannot carry edges.** An activity modelled as an edge has nowhere to
   hang its attributes, participants, provenance or confidence later.

So activities are *reified* into nodes, and their attributes become
`(:Attribute)` nodes. This also gives a clean way to say **known to be
unknown**: a property set to null is indistinguishable from an absent
property, but an `Attribute {name:"amount", value:null}` node records that an
amount exists and was not stated.

`TRAVEL` stays a plain edge — it has no sub-structure.

## Entity keys

Every node has a deterministic `key`, and all writes are `MERGE` on it, so
re-ingesting a document changes nothing.

| Entity | Key | Rationale |
|---|---|---|
| Person | `person:mary` | global — the same person across documents is one node |
| Named place | `place:chase-bank` | global |
| Unnamed place | `place:doc:<doc>:bank` | **document-scoped** |
| Activity | `activity:doc:<doc>:<i>:<type>` | events are always document-scoped |
| Attribute | `<activity key>:attr:amount` | owned by its activity |

The document scoping for unnamed places is the subtle one. Within a passage,
"the bank" in sentence one and sentence two must unify. Across unrelated
documents they must not — otherwise every anonymous bank in the corpus
silently collapses into a single node.

## Two backends, one schema

`GraphStore` (`src/graph_knowledge/store/base.py`) is a small protocol with two
implementations that share the reified schema and near-identical Cypher:

- **`Neo4jStore`** — the Docker Compose service. Gets you Neo4j Browser.
- **`EmbeddedStore`** — LadybugDB, a file on disk, no server.

The test suite is parametrised over whichever backends are available, so the
same assertions run against both and the engine choice stays reversible.
`pytest` is green with or without `make up`.

## Docker disk hygiene (macOS)

The Docker VM lives in one sparse file that **grows to a high-water mark and
never shrinks by itself**. `docker system prune` frees space *inside* the VM
while the host file stays large — which is why Docker appears to eat a disk
irreversibly. The setup here is arranged to prevent that:

- **Set a virtual disk limit.** Docker Desktop → Settings → Resources →
  Advanced → Disk image size → 24–32 GB. This is the highest-value change:
  it turns "laptop unusable" into "Docker prints an error". Lowering it
  recreates the disk image, so do it before you have data you care about.
- **Data is bind-mounted**, not in a named volume. `./neo4j/data` lives on the
  host filesystem: visible to `du -sh`, backed up with `cp -r`, invisible to
  `Docker.raw`, and immune to `docker system prune --volumes`.
- **Logs are capped** (`max-size: 10m`, `max-file: 3`) in `docker-compose.yml`.
  Uncapped container logs are a common way to fill the VM.
- **Neo4j query logging is off** — it is on by default in 5.x and writes
  continuously.
- Cap the build cache in `~/.docker/daemon.json` (Settings → Docker Engine).
  Use `reservedSpace`/`maxUsedSpace`; `defaultKeepStorage` is deprecated in
  Engine 28+:

  ```json
  {
    "log-opts": { "max-size": "10m", "max-file": "3" },
    "builder": { "gc": { "enabled": true,
      "policy": [{ "reservedSpace": "4GB", "maxUsedSpace": "8GB" }] } }
  }
  ```

```bash
make df        # what Docker is actually using
make prune     # drop unused images and build cache
make reclaim   # TRIM the VM disk so macOS gets the space back  <-- the missing step
make nuke      # delete the container AND ./neo4j. destructive.
```

## Layout

```
src/graph_knowledge/
├── models.py              # Pydantic domain model + key strategy
├── pipeline.py            # extractor -> store
├── cli.py
├── extraction/
│   ├── base.py            # Extractor protocol
│   └── rule_based.py      # deterministic baseline, no API key needed
└── store/
    ├── base.py            # GraphStore protocol
    ├── neo4j_store.py
    └── embedded_store.py
tests/test_mary.py         # the worked example, run against every backend
```

## Eval

`evals/` measures the extractor on 40 hand-written meeting-notes passages,
scored per atomic fact (person / gender / travel / activity / attribute /
datetime) rather than whole-graph, so a regression is attributable.

```bash
make eval-oracle   # harness self-test: gold replayed, must score 1.00, free
make eval-null     # null baseline: must score 0.00, free
make eval-rule     # offline rule-based baseline, free
make eval-llm      # the LLM extractor -- costs money
```

| variant | P | R | F1 |
|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 |
| rule-based | 0.58 | 0.46 | 0.51 |
| null | 1.00 | 0.00 | 0.00 |

`evals/README.md` covers the fact model, accept-lists, the train/test split,
the noise floor, and the known limitations.

## Status and next steps

The rule-based extractor is a **baseline**, not the destination. It resolves
pronouns to the most recent person and recognises travel/activity clauses from
seed lexicons — enough to exercise the pipeline offline and to make the schema
concrete. It will not generalise to arbitrary prose.

The real path, both of which plug in behind `Extractor` without touching the
store or the tests:

1. **Statistical coreference** — `fastcoref` or `maverick-coref` instead of
   "most recent person". This is where accuracy is won or lost.
2. **An LLM constrained to the `Extraction` schema** for relation and attribute
   extraction, with the Pydantic models as the validation boundary so bad
   extractions fail loudly instead of polluting the graph.

Further out: temporal resolution ("last Tuesday" → a datetime), entity
resolution beyond exact name match, and provenance edges back to the source
sentence.
