# Rubkit

Evaluate subjects against portable YAML rubrics. Judges supply evidence and scores; the scoring engine applies scales, weights, gates, and verdict bands without making network calls.

Python 3.11 or newer. Licensed under MIT.

The package is not yet published on PyPI. The distribution, Python import, and CLI names are `rubkit`.

## Install

```sh
python -m pip install -e '.[dev]'
python -m pip install -e '.[llm]'
```

The `llm` extra provides a LiteLLM model router. Core imports do not load LiteLLM or provider SDKs.

## Try the scorecard

```sh
rubkit validate spec/fixtures/story.yaml
rubkit run spec/fixtures/story.yaml spec/fixtures/subjects.json \
  --judgements spec/fixtures/judgements.json --output result.json
rubkit report result.json
```

Subjects are JSON objects with `id`, `content`, and optional `meta`. The CLI accepts one subject or a list. Imported judgements are nested by subject id, then metric id, with a list of judgements at each leaf. Results are always a JSON list.

```python
from rubkit import Judgement, Subject, load_rubric, score

rubric = load_rubric("spec/fixtures/story.yaml")
result = score(
    rubric,
    Subject(id="draft", content="A short story", meta={"branded": False}),
    {"desire": [Judgement(judge="editor", value=4, evidence=["I need that title by Friday"])]},
)
print(result.score.reported)
```

## Rubrics

Rubrics require `rubkit: "0.1"` or `"0.2"`, `id`, `version`, `name`, and at least one metric. Every metric needs a scale, either inline or through `defaults.scale`. Named scales, defaults, nested groups, gates, bands, and simple metadata conditions follow the same format in both; 0.2 adds quoted evidence.

Unknown fields are rejected except fields beginning with `x-`. Rubric and result JSON Schemas ship in `src/rubkit/schemas/`. Pydantic additionally checks references, cycles, scale bounds, and other constraints involving multiple fields.

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
* The rubric hash covers the resolved rubric including defaults. Results embed that resolved rubric. Scoring values are deterministic; run ids and timestamps are new for each invocation. Pass a fixed `RunRecord` to `score` when comparing entire result documents.

### Pass/fail checks

A check that must hold whatever the score is a binary metric with weight 0 and a fail gate. It never moves the score, and a false answer sets the verdict to `failed`:

```yaml
- id: setup-before-resolution
  name: Answer set up before it resolves
  question: Was the product mentioned before the moment it solves the problem?
  scale: {type: binary}
  weight: 0
  gate: {below: true, action: fail, message: The resolution was never set up}
```

### Quoted evidence

`evidence: quoted` on a metric asks the judge to copy each evidence item word for word from the subject's text, joining separate parts with `...`. Each quote is matched against the text, ignoring case, spacing and surrounding quotation marks. A judge whose evidence is not found is asked once more; if it still fails, its judgement is kept with `grounded: false` and the metric and result carry the flag `evidence not found in the subject`. Metrics without it record `grounded: null`.


## Judges and routing

A judge implements `judge(subject, metric) -> Judgement`. `HumanJudge` prompts for a JSON score and rationale. `FunctionJudge` wraps a Python callable returning `Judgement`; rubric function references use `module:attribute`.

LLM judges send one metric per call with evidence requested before the score. They use the shared model router. Configure a provider and model directly in a rubric, or use a router alias with a separate deployment configuration. Credentials belong in provider environment variables, never in rubric files.

Use `rubkit run ... --estimate` for the uncached LLM call count. `--cache directory` reuses sampled judgements. Cache keys include the resolved rubric, subject, judge, sample index, and seed. `--shuffle --seed 42` changes metric and judge order reproducibly. Repeated independent calls are controlled by `samples`.

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
rubkit run rubric.yaml subjects.json --router-config spec/fixtures/router.example.yaml --cache .rubkit-cache
```

The example routes `quality` to Anthropic and declares an OpenAI fallback. Set the corresponding provider API keys in your environment. A provider/model rubric entry also uses the router; no direct provider client is created. Models must support structured JSON output. Temperature is omitted unless explicitly configured.

Any OpenAI-compatible endpoint works as a deployment through `api_base`, with its key read from the environment; `spec/fixtures/router.openai-compatible.example.yaml` routes the alias `judge` this way. Pick a judge model other than the one that wrote what it grades.

The Python runner accepts a `router=ModelRouter.from_file(path)`, custom judge instances, and adapters for an existing inference gateway. `assignments={"metric-id": ["judge-name"]}` overrides rubric judge assignments for a run. Results record the resolved assignments and rubric hash. All subjects in one run share the run id and start timestamp. `workers=n` judges a subject's metrics in up to n threads at once, so a judge must be safe to call from several threads; results keep the rubric's order.

## Export, replay, and compare

JSON and YAML results embed the resolved rubric and original judgements. An embedded rubric must match its recorded hash before it can be replayed.

```sh
rubkit run spec/fixtures/story.yaml spec/fixtures/subjects.json \
  --judgements spec/fixtures/judgements.json --output results.yaml
rubkit export results.yaml --output results.csv
rubkit rerun results.yaml --output replay.json
rubkit rerun results.yaml --rejudge --router-config router.yaml --output fresh.json
rubkit arena results.yaml --output rankings.json
```

`rerun` reuses saved judgements by default and makes no judge calls. `--rejudge` invokes the original judges again against the saved subject and rubric. It requires any original Python function dependencies or provider credentials. Router deployments and credentials remain external to the result.

CSV exports contain one row per judgement, with rows for unscored metrics as well. They preserve zero and false scores, evidence, judge rationale, and model names. CSV is a viewing format; use JSON or YAML for replay.

In Python, `critique(result, top=3)` lists what to fix first: failed gates, then capped and flagged ones, then the metrics losing the most points on the report scale, each with its rationale and evidence. Points lost are measured as under a weighted mean. `compare(before, after)` gives the change in score, verdict, each metric's normalized score and the gates added or cleared between two results of the same rubric, and says whether both used the same resolved definition.

`arena` compares saved results from the same resolved rubric. It rejects blocked results, duplicate subject ids, and differing rubric hashes. Equal scores share a rank, and the next rank skips tied positions.

## Usage and cost estimates

```sh
rubkit run rubric.yaml subjects.json --dry-run --router-config router.yaml
```

`--estimate` counts uncached LLM calls without loading LLM dependencies. `--dry-run` adds token and dollar estimates through LiteLLM, using the exact judging prompts. It considers all deployments and configured fallbacks for each route. The output-token allowance is 2,048 per call. Unknown pricing is `null` with a flag. Estimates exclude retry costs, failed calls, schema overhead, and cache savings; they are not spending limits.

Completed judgements record token usage, estimated cost, and latency when the router supplies them. `result.usage` totals new calls and counts cache hits separately. Cached entries retain their original call records but contribute no new cost. Custom adapters without usage data report unknown totals. Pricing comes from LiteLLM's model catalog and may differ from your bill.

## Trust tools

`rubkit.trust` provides nominal Cohen's kappa, nominal or interval Krippendorff's alpha with missing ratings, normalized calibration error, absolute ranking with ties, pairwise preferences, and pairwise win-count ranking. Undefined agreement returns `None`.

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
python -c 'from pathlib import Path; from rubkit.loading import write_schemas; write_schemas(Path("src/rubkit/schemas"))'
```

## License

MIT. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for the retained upstream notice.

## Publishing

Create a PyPI account with two-factor authentication, then add a pending GitHub trusted publisher with these values:

| Field | Value |
| --- | --- |
| PyPI project | `rubkit` |
| Owner | `whilesmart` |
| Repository | `rubkit` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Create the `pypi` environment in the repository settings. Run the **Publish to PyPI** workflow manually from the revision to release. It tests, builds, checks the distribution metadata, and publishes using GitHub's identity without storing an API token. The first successful upload creates the PyPI project. A pending publisher does not reserve the name.
