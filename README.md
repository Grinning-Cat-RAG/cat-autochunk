# AutoChunk

A [Grinning Cat](https://github.com/Grinning-Cat-RAG/grinning-cat-core) plugin that, for every Cheshire Cat (agent) on
which it is active, periodically:

1. **optimizes the choice of the chunker**: among all the chunkers available to the Cheshire Cat (the core ones and the
   ones added by its plugins), and their parameters, it searches the configuration that best serves the retrieval on the
   datalake, with a metaheuristic algorithm of [pyVolutionary](https://github.com/matteocacciola/pyvolutionary) chosen
   in the settings;
2. **re-ingests the whole datalake** (the files stored by the file manager of the Cheshire Cat) with the chosen
   chunker, which becomes the active chunker of the Cheshire Cat.

The runs are scheduled by **White Rabbit** with a cron expression set in the settings of the plugin.

## How it works

### Fitness of a chunker

1. A random sample of the datalake files is parsed (with the parsers of the Cheshire Cat).
2. The LLM of the Cheshire Cat generates a synthetic evaluation set: for random passages of the sampled files, a
   question and the **verbatim excerpt** of the passage answering it (questions whose excerpt is not in the passage are
   discarded).
3. Each candidate configuration splits the sample exactly as the RabbitHole would (hooks
   `before_rabbithole_splits_documents` and `finalize_oversized_chunks`, merge of the short chunks included), and the
   chunks are embedded with the embedder of the Cat (embeddings are cached across candidates).
4. Each question is matched against the chunks (cosine similarity, top-k): a chunk is relevant when it comes from the
   file of the question and holds at least 60% of the excerpt. The fitness is
   `MRR@k − size_penalty × (average chunk tokens / 1000)`.

The active chunker is evaluated first, as a baseline. It is replaced only when the best configuration found is
different **and** better by at least `min_improvement`; otherwise nothing is re-ingested.

### Search space

- one discrete variable selects the chunker (settings class);
- every field of every chunker settings class becomes a variable: numbers (bounds from the pydantic constraints and the
  default value, or from `search_space_overrides`), booleans, `Literal`s and enums. Strings, secrets and fields without
  a usable value are kept fixed (to the value in use, for the active chunker);
- configurations are repaired (e.g. the overlap cannot exceed half of the chunk size); invalid ones get the worst
  fitness. Each distinct configuration is evaluated once per run.

Discrete choices are encoded as continuous intervals `[0, n)` (floored), so every choice is equally reachable by the
continuous metaheuristics.

### Safe re-ingestion

For every file of the datalake:

- the file is parsed and split with the new chunker, and the new points are stored **before** the old ones are deleted;
- the old points of the file are then deleted **by id**: if anything fails, the file keeps its old points, and it is
  retried by the next run;
- the metadata given at upload time (the ones with the same value on every chunk of the file) are carried over.

Only the points of the declarative memory whose `source` is a file of the datalake are touched: **every other point of
the vector database** (sources that are not in the file manager, URLs, memories without a source, the episodic
memories of the chats, the procedures) **is never modified nor deleted**.

The files of the chats (sub-folders of the agent folder) and the files derived from an ingestion (e.g. the images
extracted by the multimodal ingestion) are not part of the datalake.

### Scheduling

- one White Rabbit cron job per Cheshire Cat (`autochunk:<agent_id>`), scheduled when the plugin is activated on the
  agent, rescheduled when its settings change, removed when the plugin is deactivated or the agent is destroyed, and
  re-aligned at startup;
- a per-agent Redis lock prevents concurrent runs across the workers sharing the White Rabbit job store.

## Settings (per Cheshire Cat)

| Setting | Default | Description |
|---|---|---|
| `enabled` | `true` | Enables the scheduled runs. |
| `cron_expression` | `0 3 * * 0` | 5-field cron expression, UTC (crontab semantics: `0`/`7` = Sunday). |
| `algorithm` | `GreyWolfOptimization` | pyVolutionary optimizer class (see `GET /autochunk/algorithms`). |
| `algorithm_parameters` | `{}` | Extra parameters of the algorithm configuration, e.g. `{"c1": 0.1, "c2": 0.1, "w": [0.35, 1]}` for `ParticleSwarmOptimization`. |
| `population_size` | `8` | Population of the optimizer. Some algorithms need more agents (e.g. 3 for `GreyWolfOptimization`, 4 for `BeeColonyOptimization`): pyVolutionary reports the minimum. Parameters in `algorithm_parameters` bounded by the population (e.g. `n_elites` of `BiogeographyBasedOptimization`) must fit it. |
| `max_cycles` | `6` | Generations of the optimizer. |
| `max_evaluations` | `60` | Budget of distinct configurations evaluated per run. |
| `max_runtime_minutes` | `120` | Time budget of the optimization. |
| `seed` | `null` | Random seed, for reproducible runs. |
| `allowed_chunkers` | `[]` | Chunker settings classes to consider (empty: all). |
| `search_space_overrides` | `{}` | Per parameter, keyed by `"<Class>.<field>"`: `{"min", "max"}`, `{"choices": [...]}` or `{"fixed": value}`. |
| `sample_max_files` | `10` | Files sampled for the evaluation. |
| `sample_max_chars_per_file` | `200000` | Truncation of the sampled files. |
| `questions_per_file` | `3` | Synthetic questions per sampled file. |
| `min_questions` | `5` | Below this number of valid questions, the run changes nothing. |
| `top_k` | `5` | k of MRR@k. |
| `size_penalty` | `0.05` | Penalty per 1000 tokens of average chunk size. |
| `min_improvement` | `0.02` | Minimum fitness gain to switch chunker. |
| `lock_ttl_minutes` | `720` | Expiration of the per-agent run lock. |

Example of `search_space_overrides`:

```json
{
  "RecursiveTextChunkerSettings.chunk_size": {"min": 128, "max": 1024},
  "RecursiveTextChunkerSettings.encoding_name": {"choices": ["cl100k_base", "o200k_base"]}
}
```

## Endpoints

| Method | Path | Permission | Description |
|---|---|---|---|
| `GET` | `/autochunk/status` | `CHUNKER` read | Scheduled job, running state, last report, files pending re-ingestion. |
| `POST` | `/autochunk/run` | `CHUNKER` write | Runs now (one-shot White Rabbit job), even if `enabled` is false. |
| `GET` | `/autochunk/algorithms` | `CHUNKER` read | The available pyVolutionary algorithms. |

The report of the last run (stored in Redis) holds the search space, the sampled files, the baseline and the best
configuration with their metrics (fitness, MRR, hit rate, average chunk size), and the outcome of the re-ingestion.

## Requirements

- the core plugin White Rabbit (scheduling);
- a configured LLM (questions generation) and embedder;
- `pyvolutionary` >= 2.7.1 (installed by the Cat from `pyproject.toml`).

## Development

The tests run with the virtual environment of the core (they need the `cat` package):

```bash
/path/to/grinning-cat-core/.venv/bin/python -m pytest tests
```

The Cat imports (and security-scans) every `.py` file of a plugin, tests included: tests and `conftest.py` must not
import `pytest` at module level nor use dynamic imports.
