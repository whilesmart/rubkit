from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
Unit = Annotated[Number, Field(ge=0, le=1)]
Weight = Annotated[Number, Field(ge=0)]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
Method = Literal["weighted_mean", "sum", "min", "geometric_mean"]
Missing = Literal["reweight", "zero", "block"]
Reconcile = Literal["median", "mean", "majority"]
Value = float | bool | str
Evidence = Literal["free", "quoted"]


class Model(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False, serialize_by_alias=True)

    @model_validator(mode="after")
    def extensions(self):
        if any(not key.startswith("x-") for key in self.model_extra or {}):
            raise ValueError("Unknown fields must start with x-")
        return self


class Linear(Model):
    type: Literal["linear"]
    best: Number
    worst: Number

    @model_validator(mode="after")
    def endpoints(self):
        if self.best == self.worst:
            raise ValueError("best and worst must differ")
        return self


class Points(Model):
    type: Literal["points"]
    points: list[tuple[Number, Unit]] = Field(min_length=2)

    @model_validator(mode="after")
    def increasing(self):
        if any(a[0] >= b[0] for a, b in zip(self.points, self.points[1:])):
            raise ValueError("Point values must be strictly increasing")
        return self


class ScaleBase(Model):
    direction: Literal["higher_better", "lower_better"] = "higher_better"
    anchors: dict[str | int, str] = Field(default_factory=dict)


class Numeric(ScaleBase):
    type: Literal["numeric"]
    min: Number
    max: Number
    step: Annotated[Number, Field(gt=0)] = 1

    @model_validator(mode="after")
    def bounds(self):
        if self.max <= self.min:
            raise ValueError("max must exceed min")
        return self


class Binary(ScaleBase):
    type: Literal["binary"]
    true_label: str = "true"
    false_label: str = "false"

    @model_validator(mode="after")
    def distinct(self):
        if self.true_label == self.false_label:
            raise ValueError("Binary labels must differ")
        return self


class Ordinal(ScaleBase):
    type: Literal["ordinal"]
    levels: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("Ordinal levels must be unique")
        return self


class Measured(ScaleBase):
    type: Literal["measured"]
    unit: str
    map: Annotated[Linear | Points, Field(discriminator="type")]


Scale = Annotated[Numeric | Binary | Ordinal | Measured, Field(discriminator="type")]


class ReportScale(Model):
    min: Number = 0
    max: Number = 1

    @model_validator(mode="after")
    def bounds(self):
        if self.max <= self.min:
            raise ValueError("Report max must exceed min")
        return self


class Aggregate(Model):
    method: Method = "weighted_mean"
    missing: Missing = "reweight"
    report_scale: ReportScale = Field(default_factory=ReportScale)


class Gate(Model):
    below: Value
    action: Literal["fail", "cap", "flag"]
    cap: Unit | None = None
    message: str = ""

    @model_validator(mode="after")
    def capped(self):
        if self.action == "cap" and self.cap is None:
            raise ValueError("cap action requires cap")
        return self


class Condition(Model):
    field: str
    equals: Any = None
    in_: list[Any] | None = Field(default=None, alias="in")
    exists: bool | None = None

    @model_serializer(mode="wrap")
    def serialize_condition(self, handler):
        data = handler(self)
        for field in ("equals", "in_", "exists"):
            if field not in self.model_fields_set:
                data.pop(field, None)
                if field == "in_":
                    data.pop("in", None)
        return data

    @model_validator(mode="after")
    def one_operator(self):
        if len(self.model_fields_set & {"equals", "in_", "exists"}) != 1:
            raise ValueError("Condition requires exactly one operator")
        return self


class Metric(Model):
    id: Slug
    name: str
    question: str
    guidance: str = ""
    scale: str | Scale | None = None
    anchors: dict[str | int, str] = Field(default_factory=dict)
    weight: Weight | None = None
    group: str | None = None
    judges: list[str] | None = None
    gate: Gate | None = None
    applies_if: Condition | None = None
    evidence: Evidence = "free"


class Group(Model):
    id: Slug
    name: str
    weight: Weight = 1
    aggregate: Aggregate = Field(default_factory=Aggregate)
    parent: str | None = None


class JudgeConfig(Model):
    type: Literal["human", "function", "llm"]
    ref: str | None = None
    provider: str = "router"
    model: str | None = None
    samples: int = Field(default=1, ge=1)
    temperature: Number | None = None
    prompt: str | None = None
    reconcile: Reconcile = "median"
    disagreement: Unit = 0.4

    @model_validator(mode="after")
    def required_config(self):
        if self.type == "function" and not self.ref:
            raise ValueError("Function judge requires ref")
        if self.type == "llm" and (not self.provider or not self.model):
            raise ValueError("LLM judge requires provider and model")
        return self


class Defaults(Model):
    scale: str | Scale | None = None
    weight: Weight = 1
    judges: list[str] = Field(default_factory=list)
    missing: Missing | None = None


class SubjectSpec(Model):
    type: Literal["text", "file", "json", "any"] = "any"
    description: str = ""


class Band(Model):
    min: Number
    label: str


class Rubric(Model):
    rubkit: Literal["0.1", "0.2"]
    id: Slug
    version: Annotated[
        str,
        Field(
            pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
        ),
    ]
    name: str
    description: str = ""
    extends: str | None = None
    subject: SubjectSpec | None = None
    scales: dict[str, Scale] = Field(default_factory=dict)
    judges: dict[str, JudgeConfig] = Field(default_factory=dict)
    defaults: Defaults = Field(default_factory=Defaults)
    groups: list[Group] = Field(default_factory=list)
    metrics: list[Metric] = Field(min_length=1)
    aggregate: Aggregate = Field(default_factory=Aggregate)
    bands: list[Band] = Field(default_factory=list)

    def scale_for(self, metric: Metric) -> Scale:
        scale = metric.scale if metric.scale is not None else self.defaults.scale
        if isinstance(scale, str):
            if scale not in self.scales:
                raise ValueError(f"{metric.id}: unknown scale {scale}")
            return self.scales[scale]
        if scale is None:
            raise ValueError(f"{metric.id}: scale is required")
        return scale

    @model_validator(mode="after")
    def references(self):
        ids = [m.id for m in self.metrics] + [g.id for g in self.groups]
        if len(ids) != len(set(ids)):
            raise ValueError("Metric and group ids must be globally unique")
        groups = {g.id: g for g in self.groups}
        for group in self.groups:
            seen = {group.id}
            parent = group.parent
            while parent is not None:
                if parent not in groups or parent in seen:
                    raise ValueError("Unknown group parent or group cycle")
                seen.add(parent)
                parent = groups[parent].parent
        for metric in self.metrics:
            self.scale_for(metric)
            if metric.group and metric.group not in groups:
                raise ValueError(f"{metric.id}: unknown group")
            for judge in metric.judges if metric.judges is not None else self.defaults.judges:
                if judge not in self.judges:
                    raise ValueError(f"{metric.id}: unknown judge {judge}")
        return self


class Subject(Model):
    id: str
    content: Any = None
    meta: dict[str, Any] = Field(default_factory=dict)


class CallUsage(Model):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Weight | None = None
    latency_seconds: Weight | None = None


class UsageSummary(CallUsage):
    calls: int = 0
    cache_hits: int = 0


class Judgement(Model):
    judge: str
    value: Value | None = None
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: Unit | None = None
    status: Literal["scored", "missing", "na"] = "scored"
    model: str | None = None
    route: str | None = None
    usage: CallUsage | None = None
    cached: bool = False
    prompt: str | None = None
    grounded: bool | None = None


class MetricResult(Model):
    value: Value | None = None
    n: Unit | None = None
    status: Literal["scored", "missing", "na"]
    judgements: list[Judgement] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Score(Model):
    raw: Number | None = None
    n: Unit | None
    reported: Number | None


class RubricRecord(Model):
    id: str
    version: str
    hash: str
    definition: Rubric | None = None


class RunOptions(Model):
    seed: int = 0
    shuffle: bool = False
    reconcile_method: Reconcile = "median"
    disagreement: Unit = 0.4


class RunRecord(Model):
    id: str
    started_at: str
    source_run_id: str | None = None
    options: RunOptions = Field(default_factory=RunOptions)
    judges: dict[str, JudgeConfig] = Field(default_factory=dict)


class Result(Model):
    rubric: RubricRecord
    subject: Subject
    run: RunRecord
    metrics: dict[str, MetricResult]
    groups: dict[str, Unit | None]
    group_sums: dict[str, Number | None] = Field(default_factory=dict)
    score: Score
    usage: UsageSummary = Field(default_factory=UsageSummary)
    verdict: str | None
    gates: list[dict[str, Any]] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
