# Extraction eval

Measures `LLMExtractor` on meeting-notes / incident-report prose: does it
produce the right graph, and does it avoid producing facts the text does not
support.

```bash
make eval-oracle    # harness self-test, must be 1.00, free
make eval-null      # null baseline, must be 0.00, free
make eval-rule      # offline rule-based baseline, free
make eval-llm       # the real thing -- costs money
```

## What it measures

Both graphs are decomposed into atomic facts and compared as sets, scored per
kind:

| fact | tuple |
|---|---|
| `person` | name |
| `gender` | name, gender |
| `travel` | person, place |
| `activity` | person, type, place |
| `attribute` | person, type, attribute, value |
| `datetime` | person, type, ISO timestamp |

Per-fact rather than whole-graph, so one wrong attribute costs one fact rather
than zeroing an otherwise correct extraction, and a regression is attributable
to a specific kind. `gender` and `datetime` facts exist **only when asserted**,
which is what makes two properties measurable: inventing a gender from a first
name is a false positive, and resolving "last Tuesday" into a date is a false
positive.

## The cases

40 hand-written cases, five per category, in `cases.jsonl`:

| tag | what it stresses |
|---|---|
| `coref` | pronoun resolves to the right person |
| `multi_person` | several actors in one passage |
| `no_gender` | gender is **not** guessed from a first name |
| `attribute` | stated vs unstated values |
| `place` | named, unnamed, and shared across statements |
| `datetime` | absolute resolves; relative and partial stay null |
| `negative` | nothing to extract -- measures invention |
| `hard` | cross-clause reference, multiple hops |

The `negative` cases are not filler. Without them an extractor that emits
facts aggressively scores well, and the eval would only ever push in one
direction.

### Accept-lists

"flag outage" and "flag payment outage" are both right, so exact string
matching on free-form types would measure phrasing rather than extraction.
Every gold activity type, place type and attribute name carries an explicit
`*_accept` list, visible in `cases.jsonl` and reviewable. The runner prints
**near-misses** -- right person and place, wrong type string -- so accept-lists
get tuned from evidence rather than guessed at.

This is the part of the grader most likely to need adjustment. If a run shows
near-misses you consider correct, add them to the accept-list and re-run; the
runner is deterministic given the same extraction.

## Resolution

124 gold facts across 40 cases. Noise floor on the per-fact F1:

| slice | facts | 1 rep | 2 reps |
|---|---|---|---|
| full set | 124 | ±9 pts | ±6 pts |
| train (28 cases) | 90 | ±11 pts | ±7 pts |
| test (12 cases) | 34 | ±17 pts | ±12 pts |

So the full set can see a ~10-point change at 2 reps. **The held-out slice
cannot** -- at ±12 points it is a final sanity check, not a steering signal.
Iterate on `--slice train`, confirm on `--slice test`, and treat a test-slice
move smaller than 12 points as noise.

The split is drawn at random, stratified by primary tag, seeded by
`SPLIT_SEED` so membership never drifts. It is deliberately **not** drawn by
baseline score: picking the worst-scoring cases for training buys regression
to the mean and produces train gains that never appear on held-out data.

## What the harness guarantees

- **Infra failures never score as model failures.** A timeout, an API error
  after the SDK's retries, or a served model that isn't the one requested goes
  to `errors.jsonl`, not into `results.jsonl` as a zero.
- **The served model is asserted** against the requested one on every call.
- **Cost comes from the API's own token counts**, never estimated. Rates are
  in `PRICES` in `run_eval.py` -- check them before quoting a number.
- **Latency times the extraction call only**, so a variant that hit more
  transient errors doesn't look slower.
- **Every attempt writes a full trace** to `traces/`, so a surprising score
  can be investigated without paying to re-run it.
- **`--resume`** skips `(case, rep)` pairs already in `results.jsonl`.

`tests/test_eval_harness.py` enforces the two bounds in CI: oracle scores 1.0,
null scores 0.0.

## Running the LLM variant

Needs credentials — `ANTHROPIC_API_KEY`, or an `ant auth login` profile. The
runner checks before the loop and exits 2 with a message rather than
producing one identical error row per attempt.

```bash
make eval-llm                                    # train slice, 2 reps
python evals/run_eval.py --extractor llm --slice test --reps 2 --out .eval/llm-test
```

Rough spend for the train slice (28 cases x 2 reps = 56 calls), estimated
from character counts rather than `count_tokens`, so treat it as an order of
magnitude. The system prompt (~500 tokens) is cached after the first call;
case texts are ~15 tokens each, so output tokens dominate and adaptive
thinking is the variable that matters:

| output tokens/call | estimated cost |
|---|---|
| ~400 | $0.58 |
| ~900 | $1.28 |
| ~1800 | $2.54 |

The first run prints the real figure from the API's own token counts. Use
`--effort low` to cut thinking tokens if the quality holds.

## Baselines

| variant | overall P | R | F1 |
|---|---|---|---|
| oracle (gold replayed) | 1.00 | 1.00 | 1.00 |
| rule-based | 0.58 | 0.46 | 0.51 |
| null | 1.00 | 0.00 | 0.00 |

The rule-based row is the floor a real extractor has to beat. It is not
saturated and not zero, so the eval discriminates across the range that
matters.

## Known limitations

- **Gold is author-written, not production traffic.** The cases are modelled
  on meeting-notes and incident-report prose but were written for this eval.
  Replacing them with real passages as they become available is the single
  biggest upgrade available.
- **`datetime` has only 2 gold facts.** Its F1 is nearly meaningless on its
  own; read it as a smoke test, not a metric, until more dated cases exist.
- **Unit is not scored.** `attribute` compares name and value only; units are
  reported in traces but excluded to keep the metric from being brittle.
- **Multiplicity is lost.** Facts are sets, so a person doing the same thing
  twice in one passage collapses to one fact.
