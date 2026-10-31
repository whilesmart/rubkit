from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

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


def _format(path: Path | None, selected: str | None) -> str:
    if selected:
        return selected
    return {".yaml": "yaml", ".yml": "yaml", ".csv": "csv"}.get(path.suffix if path else "", "json")


def _emit(text: str, output: Path | None) -> None:
    if output:
        output.write_text(text)
    else:
        print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rubkit")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("rubric", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("rubric", type=Path)
    execute.add_argument("subjects", type=Path, help="JSON subject or list of subjects")
    input_mode = execute.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--judgements", type=Path, help="JSON mapping of subject ids to metric judgements"
    )
    input_mode.add_argument(
        "--estimate",
        action="store_true",
        help="Count uncached calls without loading LLM dependencies",
    )
    input_mode.add_argument(
        "--dry-run", action="store_true", help="Estimate tokens and cost without evaluation calls"
    )
    execute.add_argument("--shuffle", action="store_true")
    execute.add_argument("--seed", type=int, default=0)
    repeat = commands.add_parser("rerun")
    repeat.add_argument("result", type=Path)
    repeat.add_argument(
        "--rejudge",
        action="store_true",
        help="Call judges again instead of reusing saved judgements",
    )
    for command in (execute, repeat):
        command.add_argument("--cache", type=Path)
        command.add_argument(
            "--router-config", type=Path, help="LiteLLM deployment aliases and fallbacks"
        )
        command.add_argument("--output", type=Path)
        command.add_argument("--format", choices=["json", "yaml", "csv"])
    summarize = commands.add_parser("report")
    summarize.add_argument("result", type=Path)
    export = commands.add_parser("export")
    export.add_argument("result", type=Path)
    export.add_argument("--format", choices=["json", "yaml", "csv"])
    export.add_argument("--output", type=Path)
    arena = commands.add_parser("arena")
    arena.add_argument("results", nargs="+", type=Path)
    arena.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        from .output import export_results, load_results, rerun

        if args.command == "validate":
            rubric = load_rubric(args.rubric)
            print(f"{rubric.id} {rubric.version}: valid")
            return 0
        if args.command == "report":
            print("\n".join(report(result) for result in load_results(args.result)), end="")
            return 0
        if args.command == "export":
            _emit(
                export_results(load_results(args.result), _format(args.output, args.format)),
                args.output,
            )
            return 0
        if args.command == "arena":
            from .trust import ranking

            results = [result for path in args.results for result in load_results(path)]
            ranks = ranking(results)
            by_id = {result.subject.id: result for result in results}
            rows = []
            position = 1
            for subjects in ranks:
                rows.extend(
                    {
                        "rank": position,
                        "subject": name,
                        "n": by_id[name].score.n,
                        "reported": by_id[name].score.reported,
                        "verdict": by_id[name].verdict,
                    }
                    for name in subjects
                )
                position += len(subjects)
            _emit(json.dumps(rows, indent=2) + "\n", args.output)
            return 0
        if args.command == "rerun":
            from .router import ModelRouter

            if args.router_config and not args.rejudge:
                raise ValueError("--router-config requires --rejudge")
            router = ModelRouter.from_file(args.router_config) if args.router_config else None
            results = rerun(
                load_results(args.result), rejudge=args.rejudge, router=router, cache=args.cache
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
            from .router import ModelRouter

            if args.dry_run:
                from dataclasses import asdict

                from .estimates import estimate

                router = ModelRouter.from_file(args.router_config) if args.router_config else None
                _emit(
                    json.dumps(asdict(estimate(rubric, subjects, router=router)), indent=2) + "\n",
                    args.output,
                )
                return 0
            record = new_run(rubric)
            if args.judgements:
                raw = json.loads(args.judgements.read_text())
                results = [
                    score(
                        rubric,
                        subject,
                        {
                            key: [
                                Judgement.model_validate(j).model_copy(update={"cached": True})
                                for j in entries
                            ]
                            for key, entries in raw.get(subject.id, {}).items()
                        },
                        run=record,
                    )
                    for subject in subjects
                ]
            else:
                router = ModelRouter.from_file(args.router_config) if args.router_config else None
                results = run(
                    rubric,
                    subjects,
                    cache=args.cache,
                    shuffle=args.shuffle,
                    seed=args.seed,
                    router=router,
                )
        _emit(export_results(results, _format(args.output, args.format)), args.output)
        return 0
    except (ValueError, OSError, ImportError, yaml.YAMLError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
