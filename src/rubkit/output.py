from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import yaml

from .models import Result, Rubric
from .scoring import new_run, rubric_hash, score


def load_results(path: str | Path) -> list[Result]:
    path = Path(path)
    data = (
        yaml.safe_load(path.read_text())
        if path.suffix in (".yaml", ".yml")
        else json.loads(path.read_text())
    )
    results = [Result.model_validate(item) for item in (data if isinstance(data, list) else [data])]
    for result in results:
        if result.rubric.definition is not None:
            embedded_rubric(result)
    return results


def embedded_rubric(result: Result) -> Rubric:
    rubric = result.rubric.definition
    if rubric is None:
        raise ValueError("Result has no embedded rubric; rerun requires a self-contained result")
    if rubric_hash(rubric) != result.rubric.hash or (rubric.id, rubric.version) != (
        result.rubric.id,
        result.rubric.version,
    ):
        raise ValueError("Embedded rubric does not match its recorded identity and hash")
    return rubric


def export_results(results: list[Result], format: str = "json") -> str:
    data = [result.model_dump(mode="json", by_alias=True) for result in results]
    if format == "json":
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if format == "yaml":
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if format != "csv":
        raise ValueError(f"Unsupported export format: {format}")
    columns = [
        "subject",
        "rubric",
        "version",
        "run",
        "total",
        "verdict",
        "metric",
        "value",
        "n",
        "status",
        "judge",
        "sample",
        "judge_value",
        "confidence",
        "rationale",
        "evidence",
        "model",
        "cached",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for result in results:
        for name, metric in result.metrics.items():
            base = dict(
                subject=result.subject.id,
                rubric=result.rubric.id,
                version=result.rubric.version,
                run=result.run.id,
                total=result.score.reported,
                verdict=result.verdict,
                metric=name,
                value=metric.value,
                n=metric.n,
                status=metric.status,
            )
            for sample, entry in enumerate(metric.judgements):
                writer.writerow(
                    {
                        **base,
                        "judge": entry.judge,
                        "sample": sample,
                        "judge_value": entry.value,
                        "confidence": entry.confidence,
                        "rationale": entry.rationale,
                        "evidence": json.dumps(entry.evidence, ensure_ascii=False),
                        "model": entry.model,
                        "cached": entry.cached,
                    }
                )
            if not metric.judgements:
                writer.writerow(base)
    return stream.getvalue()


def rerun(
    results: list[Result], *, rejudge: bool = False, router=None, cache: Path | None = None
) -> list[Result]:
    from .judges import run

    grouped = {}
    for index, result in enumerate(results):
        rubric = embedded_rubric(result)
        key = (result.rubric.hash, result.run.options.model_dump_json())
        grouped.setdefault(key, (rubric, []))[1].append((index, result))
    output: dict[int, Result] = {}
    for rubric, entries in grouped.values():
        options = entries[0][1].run.options.model_dump()
        if rejudge:
            rescored = run(
                rubric,
                [result.subject for _, result in entries],
                router=router,
                cache=cache,
                **options,
            )
        else:
            record = new_run(rubric)
            record.options = entries[0][1].run.options.model_copy(deep=True)
            rescored = [
                score(
                    rubric,
                    result.subject,
                    {
                        name: [
                            entry.model_copy(update={"cached": True}) for entry in metric.judgements
                        ]
                        for name, metric in result.metrics.items()
                    },
                    reconcile_method=record.options.reconcile_method,
                    disagreement=record.options.disagreement,
                    run=record,
                )
                for _, result in entries
            ]
        for (index, original), result in zip(entries, rescored):
            result.run.source_run_id = original.run.id
            output[index] = result
    return [output[index] for index in range(len(results))]
