import importlib.util
import json

import pytest
from test_cli_judges import cli
from test_core import rubric

from rubrickit import Subject
from rubrickit.judges import run
from rubrickit.router import ModelRouter

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None, reason="Requires llm extra"
)


def configuration():
    return {
        "model_list": [
            {
                "model_name": "quality",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "your-api-key-here",
                    "mock_response": json.dumps(
                        {
                            "evidence": ["Example"],
                            "rationale": "Meets anchor",
                            "value": 8,
                            "confidence": 0.9,
                        }
                    ),
                },
            }
        ],
        "router_settings": {"num_retries": 0, "timeout": 5},
    }


def test_real_router_builtin_mock(tmp_path):
    config = tmp_path / "router.json"
    config.write_text(json.dumps(configuration()))
    router = ModelRouter.from_file(config)
    completion = router.complete(model="quality", prompt="Evaluate", temperature=None)
    assert json.loads(completion.text)["value"] == 8
    assert completion.model is not None
    r = rubric(
        judges={"judge": {"type": "llm", "model": "quality"}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["judge"]},
    )
    result = run(r, [Subject(id="x")], router=router)[0]
    assert result.score.n == 0.8
    assert result.metrics["a"].judgements[0].route == "quality"
    assert result.metrics["a"].judgements[0].model == completion.model


def test_router_cli_boundary(tmp_path):
    config = tmp_path / "router.json"
    config.write_text(json.dumps(configuration()))
    r = rubric(
        judges={"judge": {"type": "llm", "model": "quality"}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["judge"]},
    )
    path = tmp_path / "rubric.json"
    path.write_text(r.model_dump_json(by_alias=True))
    subjects = tmp_path / "subjects.json"
    subjects.write_text('[{"id":"x","content":"Example"}]')
    result = cli("run", path, subjects, "--router-config", config)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["score"]["n"] == 0.8


def test_router_configuration_changes_cache_identity():
    first = configuration()
    second = configuration()
    second["model_list"][0]["litellm_params"]["mock_response"] = "{}"
    assert (
        ModelRouter(**{"model_list": first["model_list"]}).cache_identity
        != ModelRouter(**{"model_list": second["model_list"]}).cache_identity
    )
