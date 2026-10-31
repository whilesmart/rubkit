from __future__ import annotations

from dataclasses import dataclass, field

from .judges import judgement_prompt
from .models import Rubric, Subject
from .router import ModelRouter, has_token_pricing
from .scoring import applies


@dataclass
class RunEstimate:
    estimated_llm_calls: int = 0
    estimated_prompt_tokens_min: int | None = 0
    estimated_prompt_tokens_max: int | None = 0
    completion_tokens_per_call_max: int = 2048
    estimated_cost_min_usd: float | None = 0.0
    estimated_cost_max_usd: float | None = 0.0
    flags: list[str] = field(default_factory=list)


def estimate(
    rubric: Rubric, subjects: list[Subject], *, router: ModelRouter | None = None
) -> RunEstimate:
    result = RunEstimate()
    requests = [
        (subject, metric, name)
        for subject in subjects
        for metric in rubric.metrics
        if applies(subject, metric.applies_if)
        for name in (metric.judges if metric.judges is not None else rubric.defaults.judges)
        if rubric.judges[name].type == "llm"
    ]
    if not requests:
        return result
    try:
        from litellm import cost_per_token, token_counter
    except ImportError as error:
        raise ImportError("Install rubkit[llm] for token and cost estimates") from error
    router = router or ModelRouter.from_rubric(rubric)
    for subject, metric, name in requests:
        config = rubric.judges[name]
        route = config.model if config.provider == "router" else f"{config.provider}/{config.model}"
        prompt = judgement_prompt(rubric, subject, metric, name)
        tokens = []
        input_costs = []
        output_costs = []
        for model in router.models_for(route):
            try:
                count = token_counter(model=model, messages=[{"role": "user", "content": prompt}])
                tokens.append(count)
            except Exception:
                result.flags.append(f"Tokenizer unavailable for {model}")
                continue
            try:
                if not has_token_pricing(model):
                    raise ValueError("Token pricing unavailable")
                low = cost_per_token(model=model, prompt_tokens=count, completion_tokens=0)
                high = cost_per_token(model=model, prompt_tokens=count, completion_tokens=2048)
                input_costs.append(sum(low))
                output_costs.append(sum(high))
            except Exception:
                result.flags.append(f"Pricing unavailable for {model}")
        count_models = len(router.models_for(route))
        samples = config.samples
        result.estimated_llm_calls += samples
        if len(tokens) == count_models and result.estimated_prompt_tokens_min is not None:
            result.estimated_prompt_tokens_min += min(tokens) * samples
            result.estimated_prompt_tokens_max += max(tokens) * samples
        else:
            result.estimated_prompt_tokens_min = result.estimated_prompt_tokens_max = None
        if len(input_costs) == count_models and result.estimated_cost_min_usd is not None:
            result.estimated_cost_min_usd += min(input_costs) * samples
            result.estimated_cost_max_usd += max(output_costs) * samples
        else:
            result.estimated_cost_min_usd = result.estimated_cost_max_usd = None
    result.flags = list(dict.fromkeys(result.flags))
    result.flags.append(
        "Estimates exclude retries, failed calls, schema overhead, and cache savings"
    )
    return result
