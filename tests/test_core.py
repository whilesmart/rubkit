import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from rubkit import Judgement, Rubric, Subject, load_rubric, normalize, score
from rubkit.loading import RubricError, schema_for
from rubkit.models import Result, RunRecord, Scale
from rubkit.scoring import aggregate
from rubkit.trust import cohens_kappa, krippendorffs_alpha, sensitivity

FIXTURES = Path(__file__).parents[1] / "spec" / "fixtures"


def rubric(**overrides):
    data = dict(
        rubkit="0.1",
        id="test",
        version="1.0.0",
        name="Test",
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}},
        metrics=[
            {"id": "a", "name": "A", "question": "A?"},
            {"id": "b", "name": "B", "question": "B?"},
        ],
    )
    data.update(overrides)
    return Rubric.model_validate(data)


def votes(**values):
    return {key: [Judgement(judge="human", value=value)] for key, value in values.items()}


@pytest.mark.parametrize(
    "data,value,expected",
    [
        ({"type": "numeric", "min": 0, "max": 5}, 4, 0.8),
        ({"type": "numeric", "min": 0, "max": 5, "direction": "lower_better"}, 4, 0.2),
        ({"type": "binary"}, True, 1),
        ({"type": "binary"}, False, 0),
        ({"type": "binary", "true_label": "yes"}, "yes", 1),
        ({"type": "ordinal", "levels": ["bad", "okay", "good"]}, "okay", 0.5),
        (
            {"type": "measured", "unit": "ms", "map": {"type": "linear", "best": 10, "worst": 100}},
            55,
            0.5,
        ),
        (
            {"type": "measured", "unit": "ms", "map": {"type": "linear", "best": 10, "worst": 100}},
            200,
            0,
        ),
        (
            {
                "type": "measured",
                "unit": "x",
                "map": {"type": "points", "points": [[0, 0], [10, 0.5], [20, 1]]},
            },
            15,
            0.75,
        ),
    ],
)
def test_scales(data, value, expected):
    assert normalize(TypeAdapter(Scale).validate_python(data), value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, 11, 1.5, True, "5", float("nan")])
def test_invalid_numeric(value):
    with pytest.raises(ValueError):
        normalize(rubric().scale_for(rubric().metrics[0]), value)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("weighted_mean", 0.65),
        ("sum", 0.5),
        ("min", 0.2),
        ("geometric_mean", (0.2 * 0.8**3) ** 0.25),
    ],
)
def test_math(method, expected):
    assert aggregate([(0.2, 1), (0.8, 3)], method, "reweight") == pytest.approx(expected)


def test_zero_geometric_and_zero_weight():
    assert aggregate([(0, 1), (0.8, 2)], "geometric_mean", "reweight") == 0
    assert aggregate([(0, 0), (0.8, 2)], "geometric_mean", "reweight") == pytest.approx(0.8)


@pytest.mark.parametrize("policy,expected", [("reweight", 0.8), ("zero", 0.4), ("block", None)])
def test_missing(policy, expected):
    result = score(rubric(aggregate={"missing": policy}), Subject(id="x"), votes(a=8))
    assert result.score.n == expected


def test_nested_groups_and_na():
    r = rubric(
        groups=[
            {"id": "outer", "name": "Outer"},
            {"id": "inner", "name": "Inner", "parent": "outer"},
        ],
        metrics=[
            {"id": "a", "name": "A", "question": "?", "group": "inner"},
            {
                "id": "b",
                "name": "B",
                "question": "?",
                "applies_if": {"field": "meta.brand", "equals": True},
            },
        ],
    )
    result = score(r, Subject(id="x"), votes(a=8))
    assert result.score.n == 0.8
    assert result.groups == {"inner": 0.8, "outer": 0.8}
    assert result.metrics["b"].status == "na"


@pytest.mark.parametrize(
    "action,expected,verdict", [("cap", 0.3, "Low"), ("fail", 0.5, "failed"), ("flag", 0.5, "High")]
)
def test_gates(action, expected, verdict):
    r = rubric(
        metrics=[
            {
                "id": "a",
                "name": "A",
                "question": "?",
                "gate": {"below": 6, "action": action, "cap": 0.3},
            }
        ],
        bands=[{"min": 0.5, "label": "High"}, {"min": 0, "label": "Low"}],
    )
    result = score(r, Subject(id="x"), votes(a=5))
    assert result.score.n == expected
    assert result.verdict == verdict
    assert len(result.gates) == 1


def test_fixture_and_result_schema():
    r = load_rubric(FIXTURES / "story.yaml")
    subject = Subject.model_validate(json.loads((FIXTURES / "subjects.json").read_text())[0])
    data = json.loads((FIXTURES / "judgements.json").read_text())[subject.id]
    result = score(
        r, subject, {k: [Judgement.model_validate(j) for j in v] for k, v in data.items()}
    )
    assert result.groups["story"] == pytest.approx(26 / 35)
    assert result.score.n == pytest.approx((4 * 26 / 35 + 2 * 0.6 + 0.8) / 7)
    assert result.verdict == "Solid"
    Draft202012Validator(schema_for(Result)).validate(result.model_dump(mode="json", by_alias=True))


def test_determinism():
    run = RunRecord(id="fixed", started_at="2026-10-31T00:00:00Z")
    first = score(rubric(), Subject(id="x"), votes(a=8, b=4), run=run)
    assert first == score(rubric(), Subject(id="x"), votes(a=8, b=4), run=run)


def test_inherit_and_lines(tmp_path):
    (tmp_path / "base.yaml").write_text(json.dumps(rubric().model_dump(mode="json", by_alias=True)))
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nmetrics:\n  - id: a\n    weight: 4\n")
    assert load_rubric(child).metrics[0].weight == 4
    child.write_text("extends: base.yaml\nmetrics:\n  - id: a\n    weight: -4\n")
    with pytest.raises(RubricError, match=r"child.yaml:4:"):
        load_rubric(child)


def test_unknown_keys_and_extensions(tmp_path):
    data = rubric().model_dump(mode="json", by_alias=True)
    data["x-custom"] = {"anything": True}
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(data))
    assert load_rubric(path).model_extra["x-custom"] == {"anything": True}
    data["typo"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(RubricError):
        load_rubric(path)


def test_reconciliation_and_disagreement():
    r = rubric()
    entries = {"a": [Judgement(judge="one", value=0), Judgement(judge="two", value=10)]}
    result = score(r, Subject(id="x"), entries)
    assert result.score.n == 0.5
    assert result.flags


def test_agreement():
    assert cohens_kappa([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    assert cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == -1
    assert krippendorffs_alpha([[0, 0], [1, 1]]) == 1
    assert krippendorffs_alpha([[0, 1], [0, 1]]) == pytest.approx(-0.5)
    assert krippendorffs_alpha([[1, None], [2, None]]) is None


def test_sensitivity():
    r = rubric(bands=[{"min": 0.5, "label": "Pass"}, {"min": 0, "label": "Low"}])
    changes = sensitivity(r, [Subject(id="x")], {"x": votes(a=10, b=0)}, percent=10)
    assert any(item["verdict_flips"] == ["x"] for item in changes)


@pytest.mark.parametrize(
    "groups",
    [[{"id": "g", "name": "G", "parent": "g"}], [{"id": "g", "name": "G", "parent": "missing"}]],
)
def test_invalid_groups(groups):
    with pytest.raises(ValidationError):
        rubric(groups=groups)


def test_sum_exposes_raw_total():
    result = score(rubric(aggregate={"method": "sum"}), Subject(id="x"), votes(a=8, b=4))
    assert result.score.raw == 12
    assert result.score.n == pytest.approx(0.6)


def test_nested_block_propagates():
    r = rubric(
        groups=[{"id": "g", "name": "G", "aggregate": {"missing": "block"}}],
        metrics=[
            {"id": "a", "name": "A", "question": "?", "group": "g"},
            {"id": "b", "name": "B", "question": "?"},
        ],
    )
    assert score(r, Subject(id="x"), votes(b=8)).score.n is None


def test_na_empty_group_dropped():
    r = rubric(
        groups=[{"id": "g", "name": "G"}],
        metrics=[
            {
                "id": "a",
                "name": "A",
                "question": "?",
                "group": "g",
                "applies_if": {"field": "meta.optional", "exists": True},
            },
            {"id": "b", "name": "B", "question": "?"},
        ],
        aggregate={"missing": "block"},
    )
    assert score(r, Subject(id="x"), votes(b=8)).score.n == 0.8


def test_majority_tie_is_error():
    with pytest.raises(ValueError, match="tied"):
        score(
            rubric(),
            Subject(id="x"),
            {"a": [Judgement(judge="a", value=2), Judgement(judge="b", value=8)]},
            reconcile_method="majority",
        )


def test_ordinal_reconciliation():
    r = rubric(defaults={"scale": {"type": "ordinal", "levels": ["bad", "okay", "good"]}})
    result = score(
        r,
        Subject(id="x"),
        {"a": [Judgement(judge="a", value="bad"), Judgement(judge="b", value="good")]},
    )
    assert result.metrics["a"].n == 0.5


def test_lower_direction_gate_uses_raw_order():
    r = rubric(
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10, "direction": "lower_better"}},
        metrics=[{"id": "a", "name": "A", "question": "?", "gate": {"below": 3, "action": "flag"}}],
    )
    assert score(r, Subject(id="x"), votes(a=2)).gates
    assert not score(r, Subject(id="x"), votes(a=8)).gates


def test_unknown_scale_is_validation_error():
    with pytest.raises(ValidationError, match="unknown scale"):
        rubric(defaults={"scale": "missing"})


def test_inheritance_cycle(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml")
    b.write_text("extends: a.yaml")
    with pytest.raises(RubricError, match="cycle"):
        load_rubric(a)


def test_duplicate_yaml_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: one\nid: two")
    with pytest.raises(RubricError, match="Duplicate key"):
        load_rubric(path)


def test_url_loading_explicit_opt_in():
    with pytest.raises(RubricError, match="allow_urls"):
        load_rubric("https://example.invalid/rubric.yaml")


def test_zero_weight_group_does_not_block():
    r = rubric(
        groups=[{"id": "g", "name": "G", "weight": 0, "aggregate": {"missing": "block"}}],
        metrics=[
            {"id": "a", "name": "A", "question": "?", "group": "g"},
            {"id": "b", "name": "B", "question": "?"},
        ],
    )
    assert score(r, Subject(id="x"), votes(b=8)).score.n == 0.8
