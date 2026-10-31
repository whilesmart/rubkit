import csv
import io
import json

import pytest
from jsonschema import Draft202012Validator
from test_cli_judges import cli
from test_core import FIXTURES, rubric, votes

from rubkit import Subject, score
from rubkit.loading import schema_for
from rubkit.models import Judgement, Result
from rubkit.output import export_results, load_results, rerun


def fixture_result(tmp_path):
    output = tmp_path / "results.json"
    process = cli(
        "run",
        FIXTURES / "story.yaml",
        FIXTURES / "subjects.json",
        "--judgements",
        FIXTURES / "judgements.json",
        "--output",
        output,
    )
    assert process.returncode == 0, process.stderr
    return output


def test_self_contained_replay_cli(tmp_path):
    original = fixture_result(tmp_path)
    data = json.loads(original.read_text())[0]
    assert data["rubric"]["definition"]["id"] == "story-quality"
    assert data["rubric"]["definition"]["rubkit"] == "0.1"
    replay = cli("rerun", original)
    assert replay.returncode == 0, replay.stderr
    result = json.loads(replay.stdout)[0]
    assert result["score"] == data["score"]
    assert result["verdict"] == data["verdict"]
    assert result["run"]["id"] != data["run"]["id"]
    assert result["run"]["source_run_id"] == data["run"]["id"]
    assert result["usage"]["calls"] == 0
    Draft202012Validator(schema_for(Result)).validate(result)


def test_tampered_rubric_cannot_replay(tmp_path):
    original = fixture_result(tmp_path)
    data = json.loads(original.read_text())
    data[0]["rubric"]["definition"]["metrics"][0]["weight"] = 99
    original.write_text(json.dumps(data))
    replay = cli("rerun", original)
    assert replay.returncode == 2
    assert "hash" in replay.stderr


@pytest.mark.parametrize("format", ["json", "yaml", "csv"])
def test_export_cli(tmp_path, format):
    original = fixture_result(tmp_path)
    output = tmp_path / f"results.{format}"
    process = cli("export", original, "--output", output)
    assert process.returncode == 0, process.stderr
    if format == "csv":
        rows = list(csv.DictReader(io.StringIO(output.read_text())))
        assert len(rows) == 10
        assert rows[0]["subject"] == "story-042"
        assert rows[0]["judge_value"] == "4.0"
    else:
        values = load_results(output)
        assert len(values) == 1
        assert values[0].verdict == "Solid"
        report = cli("report", output)
        assert report.returncode == 0, report.stderr
        assert "Solid" in report.stdout


def test_csv_retains_zero_and_false_judgements():
    r = rubric(defaults={"scale": {"type": "binary"}})
    result = score(
        r,
        Subject(id="x"),
        {"a": [Judgement(judge="one", value=False)], "b": [Judgement(judge="one", value=True)]},
    )
    rows = list(csv.DictReader(io.StringIO(export_results([result], "csv"))))
    assert rows[0]["judge_value"] == "False"
    assert rows[0]["n"] == "0.0"
    assert rows[1]["judge_value"] == "True"


def test_arena_cli_preserves_ties(tmp_path):
    r = rubric()
    results = [
        score(r, Subject(id=name), votes(a=value, b=value))
        for name, value in [("first", 8), ("second", 8), ("third", 2)]
    ]
    path = tmp_path / "results.json"
    path.write_text(export_results(results))
    process = cli("arena", path)
    assert process.returncode == 0, process.stderr
    rows = json.loads(process.stdout)
    assert [row["rank"] for row in rows] == [1, 1, 3]
    assert [row["subject"] for row in rows] == ["first", "second", "third"]


def test_arena_rejects_different_rubrics(tmp_path):
    results = [
        score(rubric(version=version), Subject(id=version), votes(a=8))
        for version in ["1.0.0", "2.0.0"]
    ]
    path = tmp_path / "results.json"
    path.write_text(export_results(results))
    process = cli("arena", path)
    assert process.returncode == 2
    assert "same rubric" in process.stderr


def test_replay_preserves_mean_reconciliation():
    r = rubric()
    result = score(
        r,
        Subject(id="x"),
        {
            "a": [
                Judgement(judge="one", value=0),
                Judgement(judge="two", value=0),
                Judgement(judge="three", value=9),
            ]
        },
        reconcile_method="mean",
    )
    replay = rerun([result])[0]
    assert replay.score.n == pytest.approx(0.3)
    assert replay.run.options.reconcile_method == "mean"


def test_non_llm_dry_run(tmp_path):
    process = cli("run", FIXTURES / "story.yaml", FIXTURES / "subjects.json", "--dry-run")
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["estimated_llm_calls"] == 0


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "meta.flag", "equals": None},
        {"field": "meta.flag", "equals": True},
        {"field": "meta.flag", "in": [1, 2]},
        {"field": "meta.flag", "exists": False},
    ],
)
def test_condition_survives_self_contained_roundtrip(tmp_path, condition):
    r = rubric(metrics=[{"id": "a", "name": "A", "question": "?", "applies_if": condition}])
    result = score(r, Subject(id="x", meta={"flag": 1}), votes(a=8))
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())
    restored = load_results(path)[0]
    assert restored.rubric.definition == r
    assert rerun([restored])[0].score == result.score
