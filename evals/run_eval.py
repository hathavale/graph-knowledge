"""Run the ticket-extraction eval.

    python evals/run_eval.py --extractor oracle   # harness self-test, free
    python evals/run_eval.py --extractor null     # null baseline, free
    python evals/run_eval.py --extractor llm --reps 2   # costs money

Design notes, each because the alternative silently corrupts the headline:

- An attempt that never produced a scorable output (timeout, API error after
  the SDK's retries, a served model that isn't the one requested) goes to
  `errors.jsonl`, never to `results.jsonl` as a zero.
- The served model is asserted against the requested one.
- Latency times the extraction call only.
- Cost comes from the API's own token counts, never estimated.
- Every attempt writes a full trace, so a surprising score can be investigated
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

from grader import FACT_KINDS, aggregate, score_case  # noqa: E402

from graph_knowledge.models import (  # noqa: E402
    Evidence, Mention, MentionRole, OrgFact, OrgRelation, Participation,
    ParticipationRole, Person, Source, Ticket, TicketExtraction, Topic,
)
from graph_knowledge.vocabulary import TermKind, Vocabulary  # noqa: E402

CASES = Path(__file__).parent / "cases.jsonl"
SPLIT_SEED = 20260912
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

    Random within each stratum, never by baseline score: a train slice picked
    from the worst cases buys regression to the mean and produces train gains
    that never reach held-out data.
    """
    by_tag: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_tag[case["tags"][0]].append(case["id"])

    rng = random.Random(SPLIT_SEED)
    assignment: dict[str, str] = {}
    for tag in sorted(by_tag):
        ids = sorted(by_tag[tag])
        rng.shuffle(ids)
        # At least one case per category on each side. A held-out slice with a
        # whole category missing cannot confirm anything about it, and with
        # three cases per tag a plain 70/30 cut leaves one category unrepresented.
        train_n = max(1, min(len(ids) - 1, round(len(ids) * TRAIN_FRACTION)))
        for index, case_id in enumerate(ids):
            assignment[case_id] = "train" if index < train_n else "test"
    return assignment


def credentials_available() -> bool:
    """Mirror the SDK's credential resolution order, checked before the loop."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if (Path.home() / ".config" / "anthropic").exists():
        return True
    return all(os.environ.get(v) for v in (
        "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
    )) and bool(os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE")
                or os.environ.get("ANTHROPIC_IDENTITY_TOKEN"))


# --- extractors -------------------------------------------------------------


class OracleExtractor:
    """Replays gold. Must score 1.0 -- otherwise the harness is broken."""

    def __init__(self, cases: list[dict]) -> None:
        self._gold = {c["id"]: c["expected"] for c in cases}

    def extract(self, text: str, ticket) -> TicketExtraction:
        if isinstance(ticket, str):
            ticket = Ticket(id=ticket)
        expected = self._gold[ticket.id]
        people = {
            p["name"]: Person(name=p["name"], aliases=p.get("aliases", []))
            for p in expected["people"]
        }
        topics = {t["name"]: Topic(name=t["name"]) for t in expected["topics"]}
        evidence = Evidence(ticket_id=ticket.id, excerpt="gold", confidence=1.0)
        return TicketExtraction(
            ticket=ticket, text=text,
            people=list(people.values()), topics=list(topics.values()),
            participations=[
                Participation(person=people[p["person"]], role=ParticipationRole(p["role"]))
                for p in expected["participations"]
            ],
            mentions=[
                Mention(person=people[m["person"]], role=MentionRole(m["role"]),
                        topic=topics.get(m["topic"]) if m.get("topic") else None,
                        evidence=evidence)
                for m in expected["mentions"]
            ],
            org_facts=[
                OrgFact(person=people[o["person"]], relation=OrgRelation(o["relation"]),
                        target=o["target"], source=Source.INFERRED, evidence=evidence)
                for o in expected["org_facts"]
            ],
        )


class NullExtractor:
    """Extracts nothing. Must score 0 -- otherwise the grader is lenient."""

    def extract(self, text: str, ticket) -> TicketExtraction:
        if isinstance(ticket, str):
            ticket = Ticket(id=ticket)
        return TicketExtraction(ticket=ticket, text=text)


def build_extractor(kind: str, cases: list[dict], model: str, effort: str | None,
                    vocabulary: Vocabulary):
    if kind == "oracle":
        return OracleExtractor(cases)
    if kind == "null":
        return NullExtractor()
    from graph_knowledge.extraction import LLMExtractor

    return LLMExtractor(vocabulary=vocabulary, model=model, effort=effort)


def seed_vocabulary(cases: list[dict]) -> Vocabulary:
    """Canonical terms the extractor is allowed to use.

    Seeded from gold so the eval measures extraction rather than the
    vocabulary's cold start: on a fresh graph every topic is a proposal, and
    scoring that would mostly measure how empty the vocabulary was.
    """
    vocabulary = Vocabulary(threshold=1)
    for case in cases:
        for topic in case["expected"]["topics"]:
            vocabulary.observe(TermKind.TOPIC, topic["name"], case["id"])
        for fact in case["expected"]["org_facts"]:
            if fact["relation"] == "HAS_FUNCTION":
                vocabulary.observe(TermKind.FUNCTION, fact["target"], case["id"])
    return vocabulary


# --- running ----------------------------------------------------------------


def usage_of(extractor):
    response = getattr(extractor, "last_response", None)
    if response is None:
        return {}, None, None
    usage = getattr(response, "usage", None)
    fields = ("input_tokens", "output_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens")
    return ({f: getattr(usage, f, None) for f in fields} if usage else {},
            getattr(response, "model", None), getattr(response, "stop_reason", None))


def cost_of(usage: dict, model: str | None) -> float | None:
    price = PRICES.get(model or "")
    if not price or not usage:
        return None
    return round(
        (usage.get("input_tokens") or 0) * price["input"] / 1e6
        + (usage.get("output_tokens") or 0) * price["output"] / 1e6
        + (usage.get("cache_read_input_tokens") or 0) * price["cache_read"] / 1e6
        + (usage.get("cache_creation_input_tokens") or 0) * price["cache_write"] / 1e6, 6)


def run(args) -> int:
    if args.extractor == "llm" and not credentials_available():
        print("No API credentials found. Set ANTHROPIC_API_KEY, or run `ant auth login`.\n"
              "The free variants need no credentials: --extractor oracle | null",
              file=sys.stderr)
        return 2

    all_cases = load_cases(Path(args.cases))
    split = make_split(all_cases)
    cases = all_cases if args.slice == "all" else [
        c for c in all_cases if split[c["id"]] == args.slice
    ]

    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    results_path, errors_path = out / "results.jsonl", out / "errors.jsonl"

    done = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["case_id"], row["rep"]))

    extractor = build_extractor(args.extractor, all_cases, args.model, args.effort,
                                seed_vocabulary(all_cases))
    results_file = results_path.open("a" if args.resume else "w")
    errors_file = errors_path.open("a" if args.resume else "w")

    for case in cases:
        for rep in range(args.reps):
            if (case["id"], rep) in done:
                continue

            started = time.perf_counter()
            try:
                extraction = extractor.extract(case["text"], Ticket(id=case["id"]))
                latency = time.perf_counter() - started
            except Exception as exc:
                errors_file.write(json.dumps({
                    "case_id": case["id"], "rep": rep,
                    "failure_class": type(exc).__name__, "detail": str(exc)[:500]}) + "\n")
                errors_file.flush()
                print(f"  ERROR {case['id']} rep{rep}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                continue

            usage, served_model, stop_reason = usage_of(extractor)
            if (served_model and args.extractor == "llm"
                    and not served_model.startswith(args.model.rsplit("-", 1)[0])):
                errors_file.write(json.dumps({
                    "case_id": case["id"], "rep": rep,
                    "failure_class": "served_model_mismatch",
                    "detail": f"requested {args.model}, served {served_model}"}) + "\n")
                errors_file.flush()
                continue

            per_kind = score_case(case["expected"], extraction)
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
                "proposed_terms": list(getattr(extraction, "proposed_terms", [])),
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


def _metrics(d: dict):
    from grader import Metrics

    return Metrics(d["tp"], d["fp"], d["fn"])


def report(results_path: Path, errors_path: Path, args) -> None:
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if not rows:
        print("no results")
        return

    totals = aggregate({k: _metrics(v) for k, v in row["grade"].items()} for row in rows)

    print(f"\nvariant={rows[0]['variant']}  slice={args.slice}  "
          f"cases={len({r['case_id'] for r in rows})}  reps={args.reps}  attempts={len(rows)}")
    errors = [l for l in errors_path.read_text().splitlines() if l.strip()] \
        if errors_path.exists() else []
    if errors:
        print(f"  {len(errors)} attempt(s) in errors.jsonl -- excluded from scores, "
              f"not counted as failures")

    print(f"\n{'fact kind':<15}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>5}{'FN':>5}")
    for kind in FACT_KINDS:
        m = totals[kind]
        print(f"{kind:<15}{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}"
              f"{m['tp']:>6}{m['fp']:>5}{m['fn']:>5}")
    o = totals["overall"]
    print(f"{'OVERALL':<15}{o['precision']:>7.2f}{o['recall']:>7.2f}{o['f1']:>7.2f}"
          f"{o['tp']:>6}{o['fp']:>5}{o['fn']:>5}")

    # Inventing org structure is the failure that sends a notification to the
    # wrong manager, so it gets its own line rather than hiding in the totals.
    org = totals["org"]
    if org["fp"]:
        print(f"\n{org['fp']} invented org fact(s) -- these route people wrongly:")
        for row in rows:
            for item in row["spurious"]:
                if item[0] == "org":
                    print(f"  {row['case_id']}: {item[1]} {item[2]} {item[3]}")

    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    if costs:
        lat = [r["latency_s"] for r in rows]
        print(f"\ncost ${sum(costs):.4f} total, ${statistics.mean(costs):.5f}/case  |  "
              f"latency {statistics.mean(lat):.2f}s mean, {max(lat):.2f}s max")

    proposed = sorted({t for r in rows for t in r.get("proposed_terms", [])})
    if proposed:
        print(f"\nproposed terms (not canonical until enough tickets agree): "
              f"{', '.join(proposed)}")

    worst = sorted(rows, key=lambda r: r["f1"])[:5]
    print("\nweakest cases:")
    for row in worst:
        missed = "  missed: " + ", ".join(row["missed"][:3]) if row["missed"] else ""
        print(f"  {row['case_id']} [{row['tags'][0]}] F1={row['f1']:.2f}{missed}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extractor", choices=["llm", "oracle", "null"], default="oracle")
    p.add_argument("--slice", choices=["train", "test", "all"], default="all")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--variant")
    p.add_argument("--cases", default=str(CASES))
    p.add_argument("--out", default=".eval/latest")
    p.add_argument("--resume", action="store_true")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
