from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

from .models import (
    Binary,
    Judgement,
    Linear,
    Measured,
    MetricResult,
    Numeric,
    Ordinal,
    Result,
    Rubric,
    RubricRecord,
    RunRecord,
    Scale,
    Score,
    Subject,
    UsageSummary,
    Value,
)


def normalize(scale: Scale, value: Value, *, threshold: bool = False) -> float:
    if isinstance(scale, Binary):
        if type(value) is bool:
            n = float(value)
        elif isinstance(value, str) and value in (scale.true_label, scale.false_label):
            n = float(value == scale.true_label)
        else:
            raise ValueError("Binary score must be a boolean or a declared label")
    elif isinstance(scale, Ordinal):
        if value not in scale.levels:
            raise ValueError(f"Unknown ordinal level: {value}")
        n = scale.levels.index(value) / (len(scale.levels) - 1)
    else:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Score must be a finite number")
        if isinstance(scale, Numeric):
            if not scale.min <= value <= scale.max:
                raise ValueError("Numeric score outside scale")
            steps = (value - scale.min) / scale.step
            if not threshold and not math.isclose(steps, round(steps), abs_tol=1e-9):
                raise ValueError("Numeric score does not match step")
            n = (value - scale.min) / (scale.max - scale.min)
        elif isinstance(scale, Measured) and isinstance(scale.map, Linear):
            n = (value - scale.map.worst) / (scale.map.best - scale.map.worst)
        else:
            points = scale.map.points
            n = points[0][1] if value <= points[0][0] else points[-1][1]
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                if x0 <= value <= x1:
                    n = y0 + (value - x0) * (y1 - y0) / (x1 - x0)
                    break
        n = max(0.0, min(1.0, n))
    return 1 - n if scale.direction == "lower_better" else n


def applies(subject: Subject, condition) -> bool:
    if condition is None:
        return True
    current = subject.model_dump()
    found = True
    for key in condition.field.split("."):
        if not isinstance(current, dict) or key not in current:
            found = False
            break
        current = current[key]
    if condition.exists is not None:
        return found == condition.exists
    if not found:
        return False
    return current in condition.in_ if condition.in_ is not None else current == condition.equals


def aggregate(values: list[tuple[float | None, float]], method: str, missing: str) -> float | None:
    active = [(n, w) for n, w in values if w > 0]
    if missing == "block" and any(n is None for n, _ in active):
        return None
    present = [
        (0.0 if n is None else n, w) for n, w in active if n is not None or missing == "zero"
    ]
    if not present:
        return None
    weight = sum(w for _, w in present)
    if method == "min":
        return min(n for n, _ in present)
    if method == "geometric_mean":
        return (
            0.0
            if any(n == 0 for n, _ in present)
            else math.exp(sum(w * math.log(n) for n, w in present) / weight)
        )
    if method == "sum":
        return sum(n for n, _ in present) / len(present)
    return sum(n * w for n, w in present) / weight


def reconcile(values: list[tuple[Value, float]], method: str) -> tuple[Value, float]:
    if method == "majority":
        counts = Counter(json.dumps(v) for v, _ in values)
        top = max(counts.values())
        if list(counts.values()).count(top) > 1:
            raise ValueError("Majority reconciliation tied")
        return next((v, n) for v, n in values if counts[json.dumps(v)] == top)
    fn = statistics.mean if method == "mean" else statistics.median
    n = fn(n for _, n in values)
    raw = [v for v, _ in values]
    if all(type(v) in (int, float) for v in raw):
        return fn(raw), n
    return min(values, key=lambda pair: (abs(pair[1] - n), pair[1]))[0], n


def rubric_hash(rubric: Rubric) -> str:
    canonical = json.dumps(
        rubric.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def summarize_usage(judgements: dict[str, list[Judgement]]) -> UsageSummary:
    entries = [
        entry
        for values in judgements.values()
        for entry in values
        if entry.model is not None or entry.usage is not None
    ]
    active = [entry for entry in entries if not entry.cached]
    totals = {}
    for field in ("prompt_tokens", "completion_tokens", "estimated_cost_usd", "latency_seconds"):
        values = [getattr(entry.usage, field) if entry.usage else None for entry in active]
        totals[field] = None if any(value is None for value in values) else sum(values)
    return UsageSummary(calls=len(active), cache_hits=len(entries) - len(active), **totals)


def new_run(rubric: Rubric) -> RunRecord:
    return RunRecord(
        id="run_" + uuid4().hex,
        started_at=datetime.now(timezone.utc).isoformat(),
        judges=rubric.judges,
    )


def score(
    rubric: Rubric,
    subject: Subject,
    judgements: dict[str, list[Judgement]],
    *,
    reconcile_method: str = "median",
    disagreement: float = 0.4,
    run: RunRecord | None = None,
) -> Result:
    if reconcile_method not in ("mean", "median", "majority"):
        raise ValueError("Unknown reconciliation method")
    if not 0 <= disagreement <= 1:
        raise ValueError("Disagreement threshold must be between 0 and 1")
    unknown = set(judgements) - {m.id for m in rubric.metrics}
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    metrics = {}
    flags = []
    for metric in rubric.metrics:
        entries = judgements.get(metric.id, [])
        if not applies(subject, metric.applies_if):
            metrics[metric.id] = MetricResult(status="na", judgements=entries)
            continue
        scale = rubric.scale_for(metric)
        scored = [j for j in entries if j.status == "scored" and j.value is not None]
        if not scored:
            status = "na" if entries and all(j.status == "na" for j in entries) else "missing"
            metrics[metric.id] = MetricResult(status=status, judgements=entries)
            continue
        by_judge = {}
        for judgement in scored:
            by_judge.setdefault(judgement.judge, []).append(
                (judgement.value, normalize(scale, judgement.value))
            )
        votes = []
        all_ns = []
        for name, values in by_judge.items():
            config = rubric.judges.get(name)
            votes.append(reconcile(values, config.reconcile if config else reconcile_method))
            all_ns.extend(n for _, n in values)
        value, n = reconcile(votes, reconcile_method)
        threshold = min(
            (rubric.judges[j.judge].disagreement for j in scored if j.judge in rubric.judges),
            default=disagreement,
        )
        local_flags = []
        if max(all_ns) - min(all_ns) > threshold:
            local_flags.append(f"judges disagree (spread {max(all_ns) - min(all_ns):.3g})")
            flags.append(f"{metric.id}: {local_flags[0]}")
        metrics[metric.id] = MetricResult(
            value=value, n=n, status="scored", judgements=entries, flags=local_flags
        )

    groups = {}
    group_sums = {}
    blocked = False
    root_sum = None
    missing = rubric.defaults.missing or rubric.aggregate.missing

    def combine(parent, config, counted=True):
        nonlocal blocked, root_sum
        children = []
        for metric in rubric.metrics:
            result = metrics[metric.id]
            if metric.group == parent and result.status != "na":
                children.append(
                    (
                        result.n,
                        metric.weight if metric.weight is not None else rubric.defaults.weight,
                    )
                )
        for group in rubric.groups:
            if group.parent == parent:
                value, included = combine(group.id, group.aggregate, counted and group.weight > 0)
                groups[group.id] = value
                if included:
                    children.append((value, group.weight))
        if config.method == "sum":
            direct = [
                rubric.scale_for(m).model_dump()
                for m in rubric.metrics
                if m.group == parent and metrics[m.id].status != "na"
            ]
            if any(g.parent == parent for g in rubric.groups) or any(
                s != direct[0] for s in direct[1:]
            ):
                raise ValueError("sum requires direct metrics with one shared scale")
            if direct and direct[0]["type"] != "numeric":
                raise ValueError("sum requires a numeric scale")
        policy = (
            config.missing
            if parent is not None and "missing" in config.model_fields_set
            else missing
        )
        if counted and policy == "block" and any(n is None and w > 0 for n, w in children):
            blocked = True
        if config.method == "sum":
            raw_values = []
            for metric in rubric.metrics:
                entry = metrics[metric.id]
                weight = metric.weight if metric.weight is not None else rubric.defaults.weight
                if metric.group != parent or entry.status == "na" or weight == 0:
                    continue
                if entry.value is not None:
                    raw_values.append(float(entry.value))
                elif policy == "zero":
                    scale = rubric.scale_for(metric)
                    raw_values.append(
                        scale.min if scale.direction == "higher_better" else scale.max
                    )
            raw_sum = sum(raw_values) if raw_values else None
            if parent is None:
                root_sum = raw_sum
            else:
                group_sums[parent] = raw_sum
        return aggregate(children, config.method, policy), bool(children)

    n, _ = combine(None, rubric.aggregate)
    if blocked:
        n = None
        flags.append("Result blocked by missing scores")
    gates = []
    failed = False
    for metric in rubric.metrics:
        result = metrics[metric.id]
        if metric.gate and result.n is not None:
            scale = rubric.scale_for(metric)
            threshold = normalize(scale, metric.gate.below, threshold=True)
            raw_order = result.n if scale.direction == "higher_better" else 1 - result.n
            threshold_order = threshold if scale.direction == "higher_better" else 1 - threshold
            if isinstance(scale, Measured):
                triggered = float(result.value) < float(metric.gate.below)
            else:
                triggered = raw_order < threshold_order
            if triggered:
                gates.append({"metric": metric.id, **metric.gate.model_dump()})
                if metric.gate.action == "cap" and n is not None:
                    n = min(n, metric.gate.cap)
                elif metric.gate.action == "fail":
                    failed = True
                elif metric.gate.action == "flag":
                    flags.append(f"{metric.id}: {metric.gate.message}")
    display = rubric.aggregate.report_scale
    reported = None if n is None else display.min + n * (display.max - display.min)
    verdict = (
        None
        if reported is None
        else next(
            (
                b.label
                for b in sorted(rubric.bands, key=lambda b: b.min, reverse=True)
                if reported >= b.min
            ),
            None,
        )
    )
    if failed:
        verdict = "failed"
    record = run or new_run(rubric)
    record = record.model_copy(
        update={
            "options": record.options.model_copy(
                update={"reconcile_method": reconcile_method, "disagreement": disagreement}
            )
        }
    )
    return Result(
        rubric=RubricRecord(
            id=rubric.id,
            version=rubric.version,
            hash=rubric_hash(rubric),
            definition=rubric.model_copy(deep=True),
        ),
        subject=subject,
        run=record,
        metrics=metrics,
        groups=groups,
        group_sums=group_sums,
        score=Score(n=n, reported=reported, raw=root_sum),
        usage=summarize_usage(judgements),
        verdict=verdict,
        gates=gates,
        flags=flags,
    )
