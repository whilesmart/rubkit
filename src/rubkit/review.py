from __future__ import annotations

from typing import Any

from .models import Model, Number, Result, Rubric, Unit, Value

GATE_ORDER = {"fail": 0, "cap": 1, "flag": 2}


class Weakness(Model):
    metric: str
    name: str
    value: Value | None
    n: Unit
    lost: Number
    gate: dict[str, Any] | None = None
    rationale: str = ""
    evidence: list[str]
    flags: list[str]


class Change(Model):
    before: Number | Value | None
    after: Number | Value | None
    delta: Number | None


class Comparison(Model):
    rubric: str
    same_definition: bool
    score: Change
    verdict: Change
    metrics: dict[str, Change]
    gates_added: list[str]
    gates_cleared: list[str]


def _definition(result: Result) -> Rubric:
    if result.rubric.definition is None:
        raise ValueError("The result does not carry its rubric definition")
    return result.rubric.definition


def shares(result: Result) -> dict[str, float]:
    """Each scored metric's part of the overall score under a weighted mean: its weight among the
    scored metrics and groups beside it, times its group's part, up to the top."""
    rubric = _definition(result)
    groups = {group.id: group for group in rubric.groups}

    def weight(metric) -> float:
        return metric.weight if metric.weight is not None else rubric.defaults.weight

    def counted(parent: str | None) -> list[tuple[str, float]]:
        members = [
            (metric.id, weight(metric))
            for metric in rubric.metrics
            if metric.group == parent and result.metrics[metric.id].n is not None
        ]
        members += [
            (group.id, group.weight)
            for group in rubric.groups
            if group.parent == parent and result.groups.get(group.id) is not None
        ]
        return [(name, w) for name, w in members if w > 0]

    found: dict[str, float] = {}

    def visit(parent: str | None, part: float) -> None:
        members = counted(parent)
        total = sum(w for _, w in members)
        for name, w in members:
            if name in groups:
                visit(name, part * w / total)
            else:
                found[name] = part * w / total

    visit(None, 1.0)
    return found


def critique(result: Result, top: int | None = 3) -> list[Weakness]:
    """The weaknesses to fix first: failed gates, then capped and flagged ones, then the metrics that
    lose the most points on the report scale."""
    rubric = _definition(result)
    span = rubric.aggregate.report_scale.max - rubric.aggregate.report_scale.min
    part = shares(result)
    gated = {gate["metric"]: gate for gate in result.gates}
    found = []
    for metric in rubric.metrics:
        scored = result.metrics[metric.id]
        if scored.n is None or (scored.n >= 1 and metric.id not in gated):
            continue
        judged = [j for j in scored.judgements if j.status == "scored"]
        found.append(
            Weakness(
                metric=metric.id,
                name=metric.name,
                value=scored.value,
                n=scored.n,
                lost=(1 - scored.n) * part.get(metric.id, 0.0) * span,
                gate=gated.get(metric.id),
                rationale=" ".join(j.rationale for j in judged if j.rationale),
                evidence=[quote for j in judged for quote in j.evidence],
                flags=scored.flags,
            )
        )
    found.sort(
        key=lambda one: (
            GATE_ORDER[one.gate["action"]] if one.gate else len(GATE_ORDER),
            -one.lost,
        )
    )
    return found if top is None else found[:top]


def _change(before: Any, after: Any) -> Change:
    numbers = all(type(v) in (int, float) for v in (before, after))
    return Change(before=before, after=after, delta=after - before if numbers else None)


def compare(before: Result, after: Result) -> Comparison:
    """How a subject's scores moved between two results under the same rubric."""
    if before.rubric.id != after.rubric.id:
        raise ValueError(f"{before.rubric.id} and {after.rubric.id} are different rubrics")
    was = {gate["metric"] for gate in before.gates}
    now = {gate["metric"] for gate in after.gates}
    return Comparison(
        rubric=before.rubric.id,
        same_definition=before.rubric.hash == after.rubric.hash,
        score=_change(before.score.reported, after.score.reported),
        verdict=_change(before.verdict, after.verdict),
        metrics={
            metric: _change(before.metrics[metric].n, after.metrics[metric].n)
            for metric in before.metrics
            if metric in after.metrics
        },
        gates_added=sorted(now - was),
        gates_cleared=sorted(was - now),
    )
