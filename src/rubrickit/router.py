from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Rubric


@dataclass(frozen=True)
class Completion:
    text: str
    model: str | None


class ModelRouter:
    def __init__(
        self,
        model_list: list[dict[str, Any]],
        *,
        fallbacks: list[dict[str, list[str]]] | None = None,
        num_retries: int = 0,
        timeout: float = 60,
    ):
        try:
            from litellm import Router
        except ImportError as error:
            raise ImportError("Install rubrickit[llm] to use the model router") from error
        config = {
            "model_list": model_list,
            "fallbacks": fallbacks,
            "num_retries": num_retries,
            "timeout": timeout,
        }
        self.cache_identity = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        self.client = Router(**config)

    @classmethod
    def from_file(cls, path: str | Path) -> ModelRouter:
        config = yaml.safe_load(Path(path).read_text())
        if not isinstance(config, dict) or "model_list" not in config:
            raise ValueError("Router config requires model_list")
        settings = config.get("router_settings", {})
        unknown = set(settings) - {"fallbacks", "num_retries", "timeout"}
        if unknown:
            raise ValueError(f"Unsupported router settings: {sorted(unknown)}")
        return cls(config["model_list"], **settings)

    @classmethod
    def from_rubric(cls, rubric: Rubric) -> ModelRouter:
        models = {}
        for config in rubric.judges.values():
            if config.type != "llm":
                continue
            route = (
                config.model if config.provider == "router" else f"{config.provider}/{config.model}"
            )
            if "/" not in route:
                raise ValueError(f"Router alias {route!r} requires a router configuration")
            models[route] = {"model_name": route, "litellm_params": {"model": route}}
        return cls(list(models.values()))

    def complete(self, *, model: str, prompt: str, temperature: float | None) -> Completion:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "evidence": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
                "value": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["evidence", "rationale", "value", "confidence"],
        }
        options = {} if temperature is None else {"temperature": temperature}
        response = self.client.completion(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "judgement", "strict": True, "schema": schema},
            },
            **options,
        )
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("Incomplete model-router response")
        text = response.choices[0].message.content
        if not isinstance(text, str):
            raise ValueError("Model router returned no text judgement")
        return Completion(text=text, model=response.model)
