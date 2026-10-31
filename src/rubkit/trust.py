from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

from .models import Judgement, Metric, Result, Rubric, Subject
from .scoring import applies, normalize, score


def cohens_kappa(left: list[Any], right: list[Any]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires equally sized nonempty paired ratings")
    n = len(left)
    a, b = Counter(left), Counter(right)
    observed = sum(x == y for x, y in zip(left, right)) / n
    expected = sum(a[k] * b[k] for k in a.keys() | b.keys()) / n**2
    return None if expected == 1 else (observed - expected) / (1 - expected)


def krippendorffs_alpha(units: list[list[Any | None]], *, level: str = "nominal") -> float | None:
    if level not in ("nominal", "interval"):
        raise ValueError("Alpha level must be nominal or interval")
    usable = [[v for v in unit if v is not None] for unit in units]
    usable = [unit for unit in usable if len(unit) > 1]
    values = [v for unit in usable for v in unit]
    if len(values) < 2:
        return None

    def distance(a, b):
        return float(a != b) if level == "nominal" else (a - b) ** 2

    observed = sum(
        sum(2 * distance(a, b) for a, b in combinations(unit, 2)) / (len(unit) - 1)
        for unit in usable
    ) / len(values)
    expected = sum(2 * distance(a, b) for a, b in combinations(values, 2)) / (
        len(values) * (len(values) - 1)
    )
    return None if expected == 0 else 1 - observed / expected


@dataclass(frozen=True)
class Calibration:
    count: int
    mean_absolute_error: float
    root_mean_squared_error: float
    bias: float


def calibrate(
    rubric: Rubric, examples: list[tuple[Subject, Metric, Judgement]], judge
) -> Calibration:
    errors = []
    for subject, metric, human in examples:
        predicted = judge.judge(subject, metric)
        if predicted.value is None or human.value is None:
            raise ValueError("Calibration requires scored human and judge values")
        scale = rubric.scale_for(metric)
        errors.append(normalize(scale, predicted.value) - normalize(scale, human.value))
    if not errors:
        raise ValueError("Calibration requires examples")
    return Calibration(
        len(errors),
        statistics.mean(abs(e) for e in errors),
        statistics.mean(e**2 for e in errors) ** 0.5,
        statistics.mean(errors),
    )


def ranking(results: list[Result]) -> list[list[str]]:
    if len({r.subject.id for r in results}) != len(results):
        raise ValueError("Ranking requires unique subject ids")
    if len({r.rubric.hash for r in results}) > 1:
        raise ValueError("Ranking requires the same rubric")
    if any(r.score.n is None for r in results):
        raise ValueError("Cannot rank blocked results")
    buckets = {}
    for result in results:
        buckets.setdefault(result.score.n, []).append(result.subject.id)
    return [sorted(buckets[n]) for n in sorted(buckets, reverse=True)]


@dataclass(frozen=True)
class Preference:
    metric: str
    left: str
    right: str
    winner: str | None
    rationale: str = ""


def pairwise(
    rubric: Rubric,
    left: Subject,
    right: Subject,
    judge: Callable[[Subject, Subject, Metric], Preference],
) -> list[Preference]:
    if left.id == right.id:
        raise ValueError("Pairwise subjects must have distinct ids")
    preferences = []
    for metric in rubric.metrics:
        if not applies(left, metric.applies_if) or not applies(right, metric.applies_if):
            continue
        result = judge(left, right, metric)
        if (result.metric, result.left, result.right) != (metric.id, left.id, right.id):
            raise ValueError("Preference does not match requested comparison")
        if result.winner not in (None, left.id, right.id):
            raise ValueError("Preference winner must be a compared subject")
        preferences.append(result)
    return preferences


def pairwise_ranking(subject_ids: list[str], preferences: list[Preference]) -> list[list[str]]:
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Subject ids must be unique")
    points = dict.fromkeys(subject_ids, 0.0)
    for preference in preferences:
        if (
            preference.left not in points
            or preference.right not in points
            or preference.left == preference.right
        ):
            raise ValueError("Invalid comparison subjects")
        if preference.winner is None:
            points[preference.left] += 0.5
            points[preference.right] += 0.5
        elif preference.winner in (preference.left, preference.right):
            points[preference.winner] += 1
        else:
            raise ValueError("Invalid comparison winner")
    buckets = {}
    for subject, total in points.items():
        buckets.setdefault(total, []).append(subject)
    return [sorted(buckets[n]) for n in sorted(buckets, reverse=True)]


def sensitivity(
    rubric: Rubric,
    subjects: list[Subject],
    judgements: dict[str, dict[str, list[Judgement]]],
    percent: float = 10,
) -> list[dict]:
    if not 0 <= percent <= 100:
        raise ValueError("Sensitivity percent must be between 0 and 100")
    baseline = [score(rubric, subject, judgements.get(subject.id, {})) for subject in subjects]
    original_rank = ranking(baseline)
    changes = []
    for kind in ("metrics", "groups"):
        for index, item in enumerate(getattr(rubric, kind)):
            weight = item.weight if item.weight is not None else rubric.defaults.weight
            for factor in (1 - percent / 100, 1 + percent / 100):
                data = rubric.model_dump(mode="json", by_alias=True)
                data[kind][index]["weight"] = weight * factor
                variant = Rubric.model_validate(data)
                results = [
                    score(variant, subject, judgements.get(subject.id, {})) for subject in subjects
                ]
                verdicts = [
                    r.subject.id for r, b in zip(results, baseline) if r.verdict != b.verdict
                ]
                rank_changed = ranking(results) != original_rank
                changes.append(
                    {
                        "kind": kind,
                        "id": item.id,
                        "factor": factor,
                        "verdict_flips": verdicts,
                        "ranking_changed": rank_changed,
                    }
                )
    return changes
