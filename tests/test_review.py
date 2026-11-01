import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rubkit import Judgement, Subject
from rubkit.grounding import found_in, grounded
from rubkit.judges import FunctionJudge, run
from rubkit.loading import load_rubric
from rubkit.models import Result
from rubkit.review import compare, critique
from rubkit.router import ModelRouter
from rubkit.scoring import score

REAL = Path(__file__).parents[1] / "spec" / "fixtures" / "real"
# What a judge on gpt-6.1-terra answered for a short script, and for the same script with the app that
# saves the day named only at the moment it is used.
RESULTS = {
    r["subject"]["id"]: Result.model_validate(r)
    for r in json.loads((REAL / "script-results.json").read_text())
}
# A real answer whose evidence talks about the sentence instead of quoting it.
PARAPHRASED = json.loads((REAL / "paraphrased-judgement.json").read_text())


def rescored(result: Result) -> Result:
    rubric = load_rubric(REAL / "script-craft.yaml")
    return score(rubric, result.subject, {k: m.judgements for k, m in result.metrics.items()})


def test_a_pass_fail_metric_with_no_weight_fails_the_result_without_moving_the_score():
    cold, setup = rescored(RESULTS["bakery-cold"]), rescored(RESULTS["bakery"])

    assert cold.verdict == "failed" and setup.verdict == "Ready"
    assert [g["metric"] for g in cold.gates] == ["setup-before-resolution"] and setup.gates == []
    assert cold.metrics["setup-before-resolution"].value is False
    assert cold.score.reported == setup.score.reported == pytest.approx(100 * (3 + 2 * 0.6 + 2) / 7)


def test_critique_ranks_failed_gates_first_then_the_points_each_metric_loses():
    weaknesses = critique(rescored(RESULTS["bakery-cold"]), top=None)

    assert [w.metric for w in weaknesses] == ["setup-before-resolution", "escalation"]
    assert weaknesses[0].gate["action"] == "fail" and weaknesses[0].lost == 0
    assert weaknesses[0].evidence and weaknesses[0].rationale
    assert weaknesses[1].lost == pytest.approx(100 * 0.4 * 2 / 7)
    assert [w.metric for w in critique(rescored(RESULTS["bakery"]))] == ["escalation"]


def test_compare_shows_how_a_rewrite_moved_each_score_and_gate():
    moved = compare(rescored(RESULTS["bakery-cold"]), rescored(RESULTS["bakery"]))

    assert moved.same_definition
    assert moved.score.delta == pytest.approx(0)
    assert (moved.verdict.before, moved.verdict.after) == ("failed", "Ready")
    assert moved.gates_cleared == ["setup-before-resolution"] and moved.gates_added == []
    assert moved.metrics["setup-before-resolution"].delta == 1.0
    assert moved.metrics["escalation"].delta == pytest.approx(0)


def test_quoted_evidence_must_appear_in_the_subject():
    subject = Subject(id="s", content=PARAPHRASED["subject"])
    paraphrased = Judgement(judge="j", **PARAPHRASED["answer"])

    assert not grounded(paraphrased, subject)
    assert grounded(paraphrased.model_copy(update={"evidence": ["“we  ship on friday”"]}), subject)
    assert found_in(
        "Gentlemen! ... I fired the engineers.", "GENTLEMEN! And Samantha. I fired the engineers."
    )
    assert not found_in("...", "anything") and not grounded(
        paraphrased.model_copy(update={"evidence": []}), subject
    )
    assert all(
        j.grounded for r in RESULTS.values() for m in r.metrics.values() for j in m.judgements
    )


def _judging(*answers: dict) -> tuple[FunctionJudge, list]:
    asked = []

    def answer(subject, metric):
        asked.append(metric.id)
        return Judgement(judge="judge", **answers[min(len(asked), len(answers)) - 1])

    return FunctionJudge("judge", answer), asked


def _quoted_rubric(evidence: str):
    rubric = load_rubric(REAL / "script-craft.yaml")
    data = rubric.model_dump(mode="json", by_alias=True)
    data["metrics"] = [{**data["metrics"][0], "evidence": evidence}]
    return type(rubric).model_validate(data)


def test_a_judge_that_does_not_quote_is_asked_once_more_then_marked_ungrounded():
    subject = Subject(id="s", content=PARAPHRASED["subject"])
    quoted = {**PARAPHRASED["answer"], "evidence": ["We ship on Friday."]}

    judge, asked = _judging(PARAPHRASED["answer"], quoted)
    result = run(_quoted_rubric("quoted"), [subject], judges={"judge": judge})[0]
    assert asked == ["goal", "goal"] and result.metrics["goal"].judgements[0].grounded is True

    judge, asked = _judging(PARAPHRASED["answer"])
    result = run(_quoted_rubric("quoted"), [subject], judges={"judge": judge})[0]
    assert asked == ["goal", "goal"] and result.metrics["goal"].judgements[0].grounded is False
    assert result.flags == ["goal: evidence not found in the subject"]

    judge, asked = _judging(PARAPHRASED["answer"])
    result = run(_quoted_rubric("free"), [subject], judges={"judge": judge})[0]
    assert asked == ["goal"] and result.metrics["goal"].judgements[0].grounded is None


@pytest.mark.skipif(importlib.util.find_spec("litellm") is None, reason="Requires llm extra")
def test_a_judge_on_an_openai_compatible_endpoint_is_reached_by_its_alias(tmp_path, monkeypatch):
    captured = (REAL / "proxy-chat-completion.json").read_bytes()
    sent = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            sent.append(
                (
                    self.path,
                    self.headers["Authorization"],
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(captured)))
            self.end_headers()
            self.wfile.write(captured)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = (
        Path(__file__).parents[1] / "spec" / "fixtures" / "router.openai-compatible.example.yaml"
    ).read_text()
    path = tmp_path / "router.yaml"
    path.write_text(
        config.replace("http://127.0.0.1:8317/v1", f"http://127.0.0.1:{server.server_port}/v1")
    )
    monkeypatch.setenv("JUDGE_API_KEY", "your-api-key-here")
    try:
        completion = ModelRouter.from_file(path).complete(
            model="judge", prompt="Evaluate", temperature=None
        )
    finally:
        server.shutdown()

    assert json.loads(completion.text)["evidence"] == ["We ship on Friday."]
    assert completion.model == "gpt-6.1-terra" and completion.usage.prompt_tokens == 405
    path_sent, key, body = sent[0]
    assert path_sent == "/v1/chat/completions" and key == "Bearer your-api-key-here"
    assert body["model"] == "gpt-6.1-terra" and body["response_format"]["type"] == "json_schema"
