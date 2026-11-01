from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from .models import Result, Rubric


class RubricError(ValueError):
    pass


def schema_for(model: type[Rubric] | type[Result]) -> dict:
    schema = model.model_json_schema(by_alias=True)

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["patternProperties"] = {"^x-": {}}
            for child in list(node.values()):
                close(child)
        elif isinstance(node, list):
            for child in node:
                close(child)

    close(schema)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in ("metrics", "groups")
            and isinstance(value, list)
            and isinstance(result.get(key), list)
        ):
            indexed = {item["id"]: item for item in result[key]}
            seen_ids = set()
            for item in value:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise ValueError(f"Inherited {key} entries require a string id")
                if item["id"] in seen_ids:
                    raise ValueError(f"Duplicate {key} id: {item['id']}")
                seen_ids.add(item["id"])
                indexed[item["id"]] = merge(indexed.get(item["id"], {}), item)
            result[key] = list(indexed.values())
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    result.pop("extends", None)
    return result


def _locations(node, path=(), found=None):
    found = {} if found is None else found
    found[path] = node.start_mark.line + 1
    if isinstance(node, yaml.MappingNode):
        keys = set()
        for key, value in node.value:
            name = key.value
            if name in keys:
                raise RubricError(f"Duplicate key {name} at line {key.start_mark.line + 1}")
            keys.add(name)
            _locations(value, (*path, name), found)
    elif isinstance(node, yaml.SequenceNode):
        for index, value in enumerate(node.value):
            _locations(value, (*path, index), found)
    return found


def load_rubric(source: str | Path, *, allow_urls: bool = False, _seen=None) -> Rubric:
    seen = set() if _seen is None else _seen
    remote = urlparse(str(source)).scheme in ("http", "https")
    identity = str(source) if remote else str(Path(source).resolve())
    if identity in seen:
        raise RubricError(f"{source}:1: Inheritance cycle")
    seen.add(identity)
    try:
        if remote:
            if not allow_urls:
                raise RubricError(f"{source}:1: URL inheritance requires allow_urls=True")
            with urlopen(str(source), timeout=15) as response:
                text = response.read(2_000_001).decode("utf-8")
            if len(text) > 2_000_000:
                raise RubricError("Rubric exceeds size limit")
        else:
            text = Path(source).read_text()
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        locations = _locations(node) if node else {(): 1}
        data = json.loads(text) if str(source).endswith(".json") else yaml.safe_load(text)
        if not isinstance(data, dict):
            raise RubricError(f"{source}:1: Rubric must be an object")
        if "extends" in data and not isinstance(data["extends"], (str, type(None))):
            raise RubricError(
                f"{source}:{locations.get(('extends',), 1)}: extends must be a string"
            )
        if data.get("extends"):
            target = (
                urljoin(str(source), data["extends"])
                if remote
                else str(Path(source).parent / data["extends"])
            )
            if urlparse(data["extends"]).scheme in ("http", "https"):
                target = data["extends"]
            base = load_rubric(target, allow_urls=allow_urls, _seen=seen)
            data = merge(base.model_dump(mode="json", by_alias=True, exclude_unset=True), data)
        errors = sorted(
            Draft202012Validator(schema_for(Rubric)).iter_errors(data),
            key=lambda e: str(list(e.path)),
        )
        if errors:
            messages = []
            for error in errors:
                path = tuple(error.absolute_path)
                while path not in locations and path:
                    path = path[:-1]
                messages.append(
                    f"{source}:{locations.get(path, 1)}: {'.'.join(map(str, error.absolute_path))}: {error.message}"
                )
            raise RubricError("\n".join(messages))
        return Rubric.model_validate(data)
    except ValidationError as error:
        messages = []
        for item in error.errors():
            path = item["loc"]
            while path not in locations and path:
                path = path[:-1]
            messages.append(f"{source}:{locations.get(path, 1)}: {item['msg']}")
        raise RubricError("\n".join(messages)) from error
    except RubricError:
        raise
    except (yaml.YAMLError, OSError, ValueError) as error:
        mark = getattr(error, "problem_mark", None)
        raise RubricError(f"{source}:{mark.line + 1 if mark else 1}: {error}") from error


def write_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in [("rubric", Rubric), ("result", Result)]:
        (directory / f"rubkit-{name}-0.2.schema.json").write_text(
            json.dumps(schema_for(model), indent=2) + "\n"
        )
