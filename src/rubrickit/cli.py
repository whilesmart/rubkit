from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .judges import estimate_calls, run
from .loading import load_rubric
from .models import Judgement, Result, Subject
from .scoring import new_run, score


def report(result: Result) -> str:
    total = "blocked" if result.score.reported is None else f"{result.score.reported:.3f}"
    lines = [
        f"# {result.subject.id}",
        "",
        f"Score: {total}",
        f"Verdict: {result.verdict or 'unassigned'}",
        "",
        "Weakest metrics:",
    ]
    weakest = sorted(
        ((key, value.n) for key, value in result.metrics.items() if value.n is not None),
        key=lambda item: (item[1], item[0]),
    )[:3]
    lines.extend(f"* {key}: {value:.3f}" for key, value in weakest)
    if result.flags or result.gates:
        lines += ["", "Flags:"]
        lines.extend(f"* {flag}" for flag in result.flags)
        lines.extend(
            f"* {gate['metric']}: {gate['action']}. {gate['message']}" for gate in result.gates
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rubrickit")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("rubric", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("rubric", type=Path)
    execute.add_argument("subjects", type=Path, help="JSON subject or list of subjects")
    execute.add_argument(
        "--judgements", type=Path, help="JSON mapping of subject ids to metric judgements"
    )
    execute.add_argument("--output", type=Path)
    execute.add_argument("--estimate", action="store_true")
    execute.add_argument("--cache", type=Path)
    execute.add_argument(
        "--router-config", type=Path, help="LiteLLM deployment aliases and fallbacks"
    )
    execute.add_argument("--shuffle", action="store_true")
    execute.add_argument("--seed", type=int, default=0)
    summarize = commands.add_parser("report")
    summarize.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            rubric = load_rubric(args.rubric)
            print(f"{rubric.id} {rubric.version}: valid")
        elif args.command == "report":
            data = json.loads(args.result.read_text())
            print(
                "\n".join(
                    report(Result.model_validate(item))
                    for item in (data if isinstance(data, list) else [data])
                ),
                end="",
            )
        else:
            rubric = load_rubric(args.rubric)
            raw = json.loads(args.subjects.read_text())
            subjects = [
                Subject.model_validate(item) for item in (raw if isinstance(raw, list) else [raw])
            ]
            if args.estimate:
                print(json.dumps({"estimated_llm_calls": estimate_calls(rubric, subjects)}))
                return 0
            record = new_run(rubric)
            if args.judgements:
                raw = json.loads(args.judgements.read_text())
                results = [
                    score(
                        rubric,
                        subject,
                        {
                            key: [Judgement.model_validate(j) for j in entries]
                            for key, entries in raw.get(subject.id, {}).items()
                        },
                        run=record,
                    )
                    for subject in subjects
                ]
            else:
                from .router import ModelRouter

                router = ModelRouter.from_file(args.router_config) if args.router_config else None
                results = run(
                    rubric,
                    subjects,
                    cache=args.cache,
                    shuffle=args.shuffle,
                    seed=args.seed,
                    router=router,
                )
            output = (
                json.dumps(
                    [item.model_dump(mode="json", by_alias=True) for item in results], indent=2
                )
                + "\n"
            )
            if args.output:
                args.output.write_text(output)
            else:
                print(output, end="")
        return 0
    except (ValueError, OSError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
