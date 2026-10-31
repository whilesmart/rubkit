# Rubrickit

Evaluate subjects against portable YAML rubrics. Judges supply evidence and scores; the scoring engine applies scales, weights, gates, and verdict bands without making network calls.

Python 3.11 or newer. Licensed under MIT.

The package is not yet published on PyPI.

## Install

```sh
python -m pip install -e '.[dev]'
python -m pip install -e '.[llm]'
```

The `llm` extra provides a LiteLLM model router. Core imports do not load LiteLLM or provider SDKs.

## Try the scorecard

```sh
rubrickit validate spec/fixtures/story.yaml
rubrickit run spec/fixtures/story.yaml spec/fixtures/subjects.json \
  --judgements spec/fixtures/judgements.json --output result.json
rubrickit report result.json
```

Subjects are JSON objects with `id`, `content`, and optional `meta`. The CLI accepts one subject or a list. Imported judgements are nested by subject id, then metric id, with a list of judgements at each leaf. Results are always a JSON list.

```python
from rubrickit import Judgement, Subject, load_rubric, score

rubric = load_rubric("spec/fixtures/story.yaml")
result = score(
    rubric,
    Subject(id="draft", content="A short story", meta={"branded": False}),
    {"desire": [Judgement(judge="editor", value=4, evidence=["I need that title by Friday"])]},
)
print(result.score.reported)
```

## Rubrics

Rubrics require `rubrickit: "0.1"`, `id`, `version`, `name`, and at least one metric. Every metric needs a scale, either inline or through `defaults.scale`. Named scales, defaults, nested groups, gates, bands, and simple metadata conditions follow the v0.1 format.

Unknown fields are rejected except fields beginning with `x-`. Rubric and result JSON Schemas ship in `src/rubrickit/schemas/`. Pydantic additionally checks references, cycles, scale bounds, and other constraints involving multiple fields.

Inheritance merges metrics and groups by id, merges mapping fields recursively, and replaces other lists. Relative paths resolve against the child rubric. URL loading is available through `load_rubric(path, allow_urls=True)`.

Measured maps use these explicit shapes:

```yaml
type: measured
unit: ms
map: {type: linear, best: 100, worst: 1000}
```

```yaml
type: measured
unit: points
map:
  type: points
  points: [[0, 0], [10, 0.5], [20, 1]]
```

Linear mappings clamp outside their endpoints. Point mappings interpolate between adjacent points and clamp outside the first and last point. `direction: lower_better` inverts the mapped value, including measured scales. Set best and worst accordingly to avoid a double inversion.

## Scoring rules

* Weighted means and geometric means use declared weights. Minimum ignores positive weight magnitude. Weight zero excludes a metric or group from aggregation.
* N/A entries always drop out. Missing scores reweight, count as normalized zero, or block. Groups inherit the overall missing policy unless their aggregation explicitly declares one. Blocking a counted group blocks the total.
* `sum` requires direct numeric metrics sharing the same complete scale definition. It adds raw values without weight multipliers and exposes the total as `score.raw` or `group_sums`. Its normalized value uses the sum's attainable minimum and maximum, so it can feed bands and parent groups. Nested child groups under a sum are rejected. Missing-zero entries contribute the raw endpoint corresponding to normalized zero.
* Gates compare against raw scale order. For numeric and measured values, `below` means a smaller raw number, even when lower is better. Ordinal gates compare declared level positions; binary gates compare false before true. Gates apply to reconciled scores. Caps apply before report rescaling and bands; a fail gate sets verdict to `failed`.
* Samples from each judge reconcile first using that judge's configuration. Judges then have equal votes, reconciled by median by default. `score(..., reconcile_method="mean")` or `"majority"` selects another method across judges. Majority ties raise an error.
* Mean and median operate on normalized scores. For numeric scales the raw value uses the same statistic. For categorical scales the displayed raw label is the closest observed level, with the lower normalized level breaking ties. The normalized score retains the exact statistic.
* A disagreement flag uses the spread of all normalized judgements and the smallest participating judge threshold.
* The rubric hash covers the resolved rubric including defaults. Scoring values are deterministic; run ids and timestamps are new for each invocation. Pass a fixed `RunRecord` to `score` when comparing entire result documents.

## Judges and routing

A judge implements `judge(subject, metric) -> Judgement`. `HumanJudge` prompts for a JSON score and rationale. `FunctionJudge` wraps a Python callable returning `Judgement`; rubric function references use `module:attribute`.

LLM judges send one metric per call with evidence requested before the score. They use the shared model router. Configure a provider and model directly in a rubric, or use a router alias with a separate deployment configuration. Credentials belong in provider environment variables, never in rubric files.

Use `rubrickit run ... --estimate` for the uncached LLM call count. `--cache directory` reuses sampled judgements. Cache keys include the resolved rubric, subject, judge, sample index, and seed. `--shuffle --seed 42` changes metric and judge order reproducibly. Repeated independent calls are controlled by `samples`.

For an alias-based judge:

```yaml
judges:
  reviewer:
    type: llm
    provider: router
    model: quality
    samples: 3
defaults:
  judges: [reviewer]
```

Add this to a rubric with scales and metrics. Run it with a deployment configuration:

```sh
rubrickit run rubric.yaml subjects.json --router-config spec/fixtures/router.example.yaml --cache .rubrickit-cache
```

The example routes `quality` to Anthropic and declares an OpenAI fallback. Set the corresponding provider API keys in your environment. A provider/model rubric entry also uses the router; no direct provider client is created. Models must support structured JSON output. Temperature is omitted unless explicitly configured.

The Python runner accepts a `router=ModelRouter.from_file(path)`, custom judge instances, and adapters for an existing inference gateway. `assignments={"metric-id": ["judge-name"]}` overrides rubric judge assignments for a run. Results record the resolved assignments and rubric hash. All subjects in one run share the run id and start timestamp.

## Trust tools

`rubrickit.trust` provides nominal Cohen's kappa, nominal or interval Krippendorff's alpha with missing ratings, normalized calibration error, absolute ranking with ties, pairwise preferences, and pairwise win-count ranking. Undefined agreement returns `None`.

`sensitivity` perturbs each metric and group weight independently by a percentage in both directions, reporting verdict and ranking changes. It requires complete, rankable baseline results. Pairwise ranking counts wins and half-points for ties; callers should provide a balanced comparison schedule.

## Development

```sh
python -m pytest
ruff check .
ruff format --check .
python -m build
```

Regenerate packaged schemas after model changes:

```sh
python -c 'from pathlib import Path; from rubrickit.loading import write_schemas; write_schemas(Path("src/rubrickit/schemas"))'
```

## Publishing

Create a PyPI account with two-factor authentication, then add a pending GitHub trusted publisher with these values:

| Field | Value |
| --- | --- |
| PyPI project | `rubrickit` |
| Owner | `whilesmart` |
| Repository | `rubrickit` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Create the `pypi` environment in the repository settings. Run the **Publish to PyPI** workflow manually from the revision to release. It tests, builds, checks the distribution metadata, and publishes using GitHub's identity without storing an API token. The first successful upload creates the PyPI project. A pending publisher does not reserve the name.
