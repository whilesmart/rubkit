import importlib.util
import json

import pytest
from test_cli_judges import cli
from test_core import rubric

from rubkit import Subject
from rubkit.judges import run
from rubkit.router import ModelRouter

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


def test_usage_metrics_and_cache_accounting(tmp_path):
    config = configuration()
    router = ModelRouter(config["model_list"])
    r = rubric(
        judges={"judge": {"type": "llm", "model": "quality", "samples": 2}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["judge"]},
    )
    first = run(r, [Subject(id="x")], router=router, cache=tmp_path)[0]
    assert first.usage.calls == 4
    assert first.usage.cache_hits == 0
    assert first.usage.prompt_tokens > 0
    assert first.usage.completion_tokens > 0
    assert first.usage.estimated_cost_usd >= 0
    assert first.usage.latency_seconds >= 0
    second = run(r, [Subject(id="x")], router=router, cache=tmp_path)[0]
    assert second.usage.calls == 0
    assert second.usage.cache_hits == 4
    assert second.usage.estimated_cost_usd == 0
    assert second.metrics["a"].judgements[0].usage == first.metrics["a"].judgements[0].usage


def test_real_router_estimate_and_dry_run_cli(tmp_path):
    from rubkit.estimates import estimate

    config = configuration()
    r = rubric(
        judges={"judge": {"type": "llm", "model": "quality", "samples": 2}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["judge"]},
    )
    router = ModelRouter(config["model_list"])
    prediction = estimate(r, [Subject(id="x", content="Example")], router=router)
    assert prediction.estimated_llm_calls == 4
    assert prediction.estimated_prompt_tokens_min > 0
    assert prediction.estimated_cost_max_usd > prediction.estimated_cost_min_usd >= 0
    path = tmp_path / "rubric.json"
    path.write_text(r.model_dump_json(by_alias=True))
    subjects = tmp_path / "subjects.json"
    subjects.write_text('[{"id":"x","content":"Example"}]')
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(config))
    process = cli("run", path, subjects, "--dry-run", "--router-config", router_path)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["estimated_llm_calls"] == 4


def test_rerun_rejudge_cli_uses_new_calls(tmp_path):
    r = rubric(
        judges={"judge": {"type": "llm", "model": "quality"}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["judge"]},
    )
    router = ModelRouter(configuration()["model_list"])
    first = run(r, [Subject(id="x")], router=router)[0]
    results = tmp_path / "results.json"
    results.write_text(json.dumps([first.model_dump(mode="json", by_alias=True)]))
    router_path = tmp_path / "router.json"
    router_path.write_text(json.dumps(configuration()))
    process = cli("rerun", results, "--rejudge", "--router-config", router_path)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)[0]
    assert result["usage"]["calls"] == 2
    assert result["run"]["source_run_id"] == first.run.id


def test_fallback_estimate_includes_all_deployments():
    config = configuration()
    second = {**config["model_list"][0], "model_name": "backup"}
    router = ModelRouter([*config["model_list"], second], fallbacks=[{"quality": ["backup"]}])
    assert router.models_for("quality") == ["openai/gpt-4o-mini"]
    with pytest.raises(ValueError, match="Unknown router alias"):
        router.models_for("missing")


def test_unknown_model_pricing_is_not_fabricated():
    from rubkit.estimates import estimate

    config = configuration()
    config["model_list"][0]["litellm_params"]["model"] = "openai/rubkit-unregistered-model"
    router = ModelRouter(config["model_list"])
    r = rubric(
        judges={"judge": {"type": "llm", "model": "quality"}},
        defaults={"scale": {"type": "numeric", "min": 0, "max": 10}, "judges": ["judge"]},
    )
    prediction = estimate(r, [Subject(id="x")], router=router)
    assert prediction.estimated_cost_min_usd is None
    assert prediction.estimated_cost_max_usd is None
    assert any("Pricing unavailable" in flag for flag in prediction.flags)
