"""Run the extraction eval and write results the report tooling can read.

    python evals/run_eval.py --extractor oracle    # harness self-test, free
    python evals/run_eval.py --extractor null      # null baseline, free
    python evals/run_eval.py --extractor rule      # offline baseline, free
    python evals/run_eval.py --extractor llm --reps 2   # costs money

Design notes, each of which exists because the alternative silently corrupts
the headline number:

- An attempt that never produced a scorable output (timeout, API error after
  the SDK's retries, a served model that isn't the one we asked for) goes to
  `errors.jsonl`, never to `results.jsonl` as a zero. Scoring plumbing
  failures as model failures is the most common way an eval lies.
- The model that served each request is read back from the response and
  asserted against the one requested; a silent reroute would otherwise be
  measured as a quality change.
- Latency times the extraction call only, so a variant that hit more
  transient errors does not look slower.
- Cost is derived from the API's own token counts, never estimated.
- Every attempt writes a full trace, so a surprising score can be traced
  without paying to re-run it.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grader import FACT_KINDS, aggregate, score_case, type_mismatches  # noqa: E402

from graph_knowledge.models import (  # noqa: E402
    Activity,
    ActivityEvent,
    Attribute,
    Extraction,
    Person,
    Place,
    Travel,
)

CASES = Path(__file__).parent / "cases.jsonl"
SPLIT_SEED = 20260912  # fixed so train/test membership never drifts
TRAIN_FRACTION = 0.7

# USD per million tokens. Check against current pricing before quoting costs.
PRICES = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
}


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_split(cases: list[dict]) -> dict[str, str]:
    """Stratified random split by primary tag.

    Drawn at random within each stratum, never by baseline score: selecting a
    train slice from the worst-scoring cases buys regression to the mean and
    produces train gains that never appear on held-out data.
    """
    by_tag: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_tag[case["tags"][0]].append(case["id"])

    rng = random.Random(SPLIT_SEED)
    shuffled = {}
    for tag in sorted(by_tag):
        ids = sorted(by_tag[tag])
        rng.shuffle(ids)
        shuffled[tag] = ids

    # Round-robin across strata rather than rounding within each one: with 8
    # strata of 5, per-stratum rounding of 0.7 lands on 32/8 and leaves the
    # held-out slice too thin to confirm anything.
    ordered: list[str] = []
    for index in range(max(len(v) for v in shuffled.values())):
        for tag in sorted(shuffled):
            if index < len(shuffled[tag]):
                ordered.append(shuffled[tag][index])

    cut = round(len(ordered) * TRAIN_FRACTION)
    return {cid: ("train" if i < cut else "test") for i, cid in enumerate(ordered)}


# --- extractors -------------------------------------------------------------


def _place(spec: dict | None) -> Place | None:
    if spec is None:
        return None
    return Place(name=spec.get("name"), type=spec.get("type"))


class OracleExtractor:
    """Replays the gold graph. Must score 1.0 -- if it doesn't, the harness is broken."""

    name = "oracle"

    def __init__(self, cases: list[dict]) -> None:
        self._gold = {case["id"]: case["expected"] for case in cases}

    def extract(self, text: str, doc_id: str) -> Extraction:
        import datetime as dt

        expected = self._gold[doc_id]
        people = {p["name"]: Person(name=p["name"], gender=p.get("gender")) for p in expected["people"]}
        return Extraction(
            doc_id=doc_id,
            text=text,
            people=list(people.values()),
            places=[_place(t["place"]) for t in expected["travels"]],
            travels=[
                Travel(person=people[t["person"]], place=_place(t["place"]))
                for t in expected["travels"]
            ],
            activities=[
                ActivityEvent(
                    person=people[a["person"]],
                    activity=Activity(
                        type=a["type"],
                        datetime=dt.datetime.fromisoformat(a["datetime"]) if a.get("datetime") else None,
                        attributes=[
                            Attribute(name=x["name"], value=x.get("value"), unit=x.get("unit"))
                            for x in a.get("attributes", [])
                        ],
                    ),
                    place=_place(a.get("place")),
                )
                for a in expected["activities"]
            ],
        )


class NullExtractor:
    """Extracts nothing. Must score 0 -- if it doesn't, the grader is too lenient."""

    name = "null"

    def extract(self, text: str, doc_id: str) -> Extraction:
        return Extraction(doc_id=doc_id, text=text)


def build_extractor(kind: str, cases: list[dict], model: str, effort: str | None):
    if kind == "oracle":
        return OracleExtractor(cases)
    if kind == "null":
        return NullExtractor()
    if kind == "rule":
        from graph_knowledge.extraction import RuleBasedExtractor

        return RuleBasedExtractor()
    from graph_knowledge.extraction import LLMExtractor

    return LLMExtractor(model=model, effort=effort)


# --- running ----------------------------------------------------------------


def usage_of(extractor) -> tuple[dict, str | None, str | None]:
    """Pull usage, served model and stop_reason off the last API response."""
    response = getattr(extractor, "last_response", None)
    if response is None:
        return {}, None, None
    usage = getattr(response, "usage", None)
    fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return (
        {f: getattr(usage, f, None) for f in fields} if usage else {},
        getattr(response, "model", None),
        getattr(response, "stop_reason", None),
    )


def cost_of(usage: dict, model: str | None) -> float | None:
    price = PRICES.get(model or "")
    if not price or not usage:
        return None
    return round(
        (usage.get("input_tokens") or 0) * price["input"] / 1e6
        + (usage.get("output_tokens") or 0) * price["output"] / 1e6
        + (usage.get("cache_read_input_tokens") or 0) * price["cache_read"] / 1e6
        + (usage.get("cache_creation_input_tokens") or 0) * price["cache_write"] / 1e6,
        6,
    )


def run(args) -> int:
    cases = load_cases(Path(args.cases))
    split = make_split(cases)
    if args.slice != "all":
        cases = [c for c in cases if split[c["id"]] == args.slice]

    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    results_path, errors_path = out / "results.jsonl", out / "errors.jsonl"

    done = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["case_id"], row["rep"]))

    extractor = build_extractor(args.extractor, load_cases(Path(args.cases)), args.model, args.effort)
    results_file = results_path.open("a" if args.resume else "w")
    errors_file = errors_path.open("a" if args.resume else "w")
    scored: list[dict] = []

    for case in cases:
        for rep in range(args.reps):
            if (case["id"], rep) in done:
                continue

            started = time.perf_counter()
            try:
                extraction = extractor.extract(case["text"], case["id"])
                latency = time.perf_counter() - started
            except Exception as exc:  # infra, not model quality
                errors_file.write(json.dumps({
                    "case_id": case["id"], "rep": rep,
                    "failure_class": type(exc).__name__, "detail": str(exc)[:500],
                }) + "\n")
                errors_file.flush()
                print(f"  ERROR {case['id']} rep{rep}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            usage, served_model, stop_reason = usage_of(extractor)
            if served_model and args.extractor == "llm" and not served_model.startswith(args.model.rsplit("-", 1)[0]):
                errors_file.write(json.dumps({
                    "case_id": case["id"], "rep": rep, "failure_class": "served_model_mismatch",
                    "detail": f"requested {args.model}, served {served_model}",
                }) + "\n")
                errors_file.flush()
                continue

            per_kind = score_case(case["expected"], extraction)
            scored.append(per_kind)
            row = {
                "case_id": case["id"], "rep": rep, "tags": case["tags"],
                "slice": split[case["id"]], "variant": args.variant or args.extractor,
                "model": served_model, "stop_reason": stop_reason,
                "status": "truncated" if stop_reason == "max_tokens" else "ok",
                "latency_s": round(latency, 3), "usage": usage,
                "cost_usd": cost_of(usage, served_model),
                "grade": {k: m.as_dict() for k, m in per_kind.items()},
                "f1": aggregate([per_kind])["overall"]["f1"],
                "missed": [lbl for m in per_kind.values() for lbl in m.missed],
                "spurious": [list(s) for m in per_kind.values() for s in m.spurious],
                "type_mismatches": type_mismatches(case["expected"], extraction),
            }
            results_file.write(json.dumps(row) + "\n")
            results_file.flush()

            (out / "traces" / f"{case['id']}_rep{rep}.json").write_text(json.dumps({
                "case_id": case["id"], "rep": rep, "text": case["text"],
                "expected": case["expected"],
                "extracted": json.loads(extraction.model_dump_json()),
                "grade": row["grade"], "usage": usage, "model": served_model,
            }, indent=2, default=str))

    results_file.close()
    errors_file.close()
    report(results_path, errors_path, args)
    return 0


def report(results_path: Path, errors_path: Path, args) -> None:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if not rows:
        print("no results")
        return

    per_case = [{k: _metrics(v) for k, v in row["grade"].items()} for row in rows]
    totals = aggregate(per_case)

    print(f"\nvariant={rows[0]['variant']}  slice={args.slice}  "
          f"cases={len({r['case_id'] for r in rows})}  reps={args.reps}  attempts={len(rows)}")
    errors = [l for l in errors_path.read_text().splitlines() if l.strip()] if errors_path.exists() else []
    if errors:
        print(f"  {len(errors)} attempt(s) in errors.jsonl -- excluded from scores, not counted as failures")

    print(f"\n{'fact kind':<12}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>5}{'FN':>5}")
    for kind in FACT_KINDS:
        m = totals[kind]
        print(f"{kind:<12}{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}"
              f"{m['tp']:>6}{m['fp']:>5}{m['fn']:>5}")
    o = totals["overall"]
    print(f"{'OVERALL':<12}{o['precision']:>7.2f}{o['recall']:>7.2f}{o['f1']:>7.2f}"
          f"{o['tp']:>6}{o['fp']:>5}{o['fn']:>5}")

    if args.reps > 1:
        by_rep = defaultdict(list)
        for row in rows:
            by_rep[row["rep"]].append(row["f1"])
        means = [statistics.mean(v) for v in by_rep.values() if v]
        if len(means) > 1:
            print(f"\nper-rep mean case F1: {[round(m, 3) for m in means]}  "
                  f"spread {max(means) - min(means):.3f}")

    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    if costs:
        lat = [r["latency_s"] for r in rows]
        print(f"\ncost ${sum(costs):.4f} total, ${statistics.mean(costs):.5f}/case  |  "
              f"latency {statistics.mean(lat):.2f}s mean, {max(lat):.2f}s max")

    mismatches = [m for r in rows for m in r["type_mismatches"]]
    if mismatches:
        print(f"\n{len(mismatches)} activity-type near-miss(es) -- candidates for an accept-list entry:")
        for m in mismatches[:8]:
            print(f"  gold {m['gold']!r} <- predicted {m['predicted_type']!r}")

    worst = sorted(rows, key=lambda r: r["f1"])[:5]
    print("\nweakest cases:")
    for row in worst:
        print(f"  {row['case_id']} [{row['tags'][0]}] F1={row['f1']:.2f}"
              f"{'  missed: ' + ', '.join(row['missed'][:3]) if row['missed'] else ''}")


def _metrics(d: dict):
    from grader import Metrics

    return Metrics(d["tp"], d["fp"], d["fn"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extractor", choices=["llm", "rule", "oracle", "null"], default="oracle")
    p.add_argument("--slice", choices=["train", "test", "all"], default="all")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--variant", help="label for this run in results.jsonl")
    p.add_argument("--cases", default=str(CASES))
    p.add_argument("--out", default=".eval/latest")
    p.add_argument("--resume", action="store_true")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
