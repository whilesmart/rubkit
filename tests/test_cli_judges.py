import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_core import FIXTURES, rubric, votes

from rubrickit import Judgement, Subject
from rubrickit.judges import FunctionJudge, HumanJudge, estimate_calls, run
from rubrickit.loading import schema_for
from rubrickit.models import Result, Rubric
from rubrickit.trust import Preference, calibrate, pairwise, pairwise_ranking, ranking

ROOT = Path(__file__).parents[1]


def cli(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "rubrickit", *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_cli_end_to_end(tmp_path):
    validation = cli("validate", FIXTURES / "story.yaml")
    assert validation.returncode == 0, validation.stderr
    output = tmp_path / "result.json"
    execution = cli(
        "run",
        FIXTURES / "story.yaml",
        FIXTURES / "subjects.json",
        "--judgements",
        FIXTURES / "judgements.json",
        "--output",
        output,
    )
    assert execution.returncode == 0, execution.stderr
    results = json.loads(output.read_text())
    assert results[0]["verdict"] == "Solid"
    Draft202012Validator(schema_for(Result)).validate(results[0])
    report = cli("report", output)
    assert report.returncode == 0, report.stderr
    assert "Solid" in report.stdout
    assert "Weakest metrics" in report.stdout
    estimate = cli("run", FIXTURES / "story.yaml", FIXTURES / "subjects.json", "--estimate")
    assert json.loads(estimate.stdout) == {"estimated_llm_calls": 0}


def test_cli_invalid(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text('rubrickit: "0.1"\nid: test\nnope: true\n')
    result = cli("validate", path)
    assert result.returncode == 2
    assert str(path) + ":1:" in result.stderr
    assert "Traceback" not in result.stderr


class LocalAdapter:
    def __init__(self):
        self.calls = []

    def complete(self, *, model, prompt, temperature):
        self.calls.append((model, prompt, temperature))
        return json.dumps(
            {
                "evidence": ["Some text"],
                "rationale": "Matches anchor",
                "value": 8,
                "confidence": 0.9,
            }
        )


def llm_rubric():
    return rubric(
        judges={"llm": {"type": "llm", "provider": "local", "model": "test-model", "samples": 3}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["llm"]},
    )


def test_llm_adapter_samples_cache_and_estimate(tmp_path):
    r = llm_rubric()
    adapter = LocalAdapter()
    subjects = [Subject(id="x", content="Some text")]
    assert estimate_calls(r, subjects) == 6
    first = run(r, subjects, adapters={"local": adapter}, cache=tmp_path, shuffle=True)
    assert len(adapter.calls) == 6
    assert first[0].score.n == 0.8
    second = run(r, subjects, adapters={"local": adapter}, cache=tmp_path, shuffle=True)
    assert len(adapter.calls) == 6
    assert second[0].score == first[0].score
    entry = first[0].metrics["a"].judgements[0]
    assert entry.model == "test-model"
    assert "evidence before" in entry.prompt
    assert entry.evidence == ["Some text"]
    run(
        r,
        [Subject(id="x", content="Different")],
        adapters={"local": adapter},
        cache=tmp_path,
        shuffle=True,
    )
    assert len(adapter.calls) == 12


def test_human_function_and_calibration():
    r = rubric()
    answers = iter(["8", "Good"])
    human = HumanJudge("editor", lambda prompt: next(answers))
    value = human.judge(Subject(id="x"), r.metrics[0])
    assert value.value == 8
    function = FunctionJudge("check", lambda subject, metric: Judgement(judge="ignored", value=6))
    assert function.judge(Subject(id="x"), r.metrics[0]).judge == "check"
    calibration = calibrate(r, [(Subject(id="x"), r.metrics[0], value)], function)
    assert calibration.mean_absolute_error == pytest.approx(0.2)
    assert calibration.bias == pytest.approx(-0.2)


def test_registered_function_run():
    r = rubric(
        judges={"check": {"type": "function", "ref": "unused:callable"}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["check"]},
    )
    results = run(
        r,
        [Subject(id="x")],
        judges={"check": FunctionJudge("check", lambda s, m: Judgement(judge="check", value=10))},
    )
    assert results[0].score.n == 1


def test_schema_files_current():
    for name, model in [("rubric", Rubric), ("result", Result)]:
        path = ROOT / "src" / "rubrickit" / "schemas" / f"rubrickit-{name}-0.1.schema.json"
        assert json.loads(path.read_text()) == schema_for(model)
        Draft202012Validator.check_schema(schema_for(model))


def test_pairwise_and_rankings():
    r = rubric()
    left, right = Subject(id="left"), Subject(id="right")
    preferences = pairwise(r, left, right, lambda a, b, m: Preference(m.id, a.id, b.id, a.id))
    assert pairwise_ranking(["left", "right"], preferences) == [["left"], ["right"]]
    from rubrickit import score

    assert ranking([score(r, left, votes(a=10, b=10)), score(r, right, votes(a=2, b=2))]) == [
        ["left"],
        ["right"],
    ]


def test_runtime_assignment_and_no_network_imports():
    r = rubric(judges={"check": {"type": "function", "ref": "unused:callable"}})
    results = run(
        r,
        [Subject(id="x")],
        judges={"check": FunctionJudge("check", lambda s, m: Judgement(judge="check", value=10))},
        assignments={"a": ["check"], "b": ["check"]},
    )
    assert results[0].score.n == 1
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            'import rubrickit; import rubrickit.judges; import sys; assert "litellm" not in sys.modules; assert "anthropic" not in sys.modules',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_subjects_share_run_record():
    results = run(rubric(), [Subject(id="one"), Subject(id="two")])
    assert results[0].run == results[1].run
