# Condition Briefing

A demo tool for a health system strategy team. Given a medical condition, it runs a fixed research
plan over PubMed and ClinicalTrials.gov and produces a briefing on the current standard of care,
emerging treatments, and the companies and institutions involved.

Provenance is part of the data model. Every report statement cites evidence items, every evidence
item holds a verbatim quote from a source, every source records the query and time that retrieved
it, and every step records its parameters, models and prompt versions. The report shows a short
version of this. A chat assistant answers questions about the rest and can re-run the plan from
any step.

## Run locally

```bash
cp .env.example .env        # set OPENAI_API_KEY
docker compose up -d --build
open http://127.0.0.1:8090  # APP_PORT overrides the port
```

LLM call traces are at http://127.0.0.1:6016 (Phoenix, project `briefing`; `PHOENIX_PORT` overrides
the port). Every OpenAI request is a span with its prompt, response, tool calls, tokens and latency.
Tracing is on only when `PHOENIX_COLLECTOR_ENDPOINT` is set, which compose does for the app.

Runs are stored as JSON under `./data/runs/<run_id>/`. A full run takes about 2 minutes and about
100 LLM calls, most of them to the small model.

Without Docker:

```bash
uv sync
uv run --env-file .env streamlit run briefing/ui.py
uv run --env-file .env python -m briefing.cli run "type 2 diabetes"
uv run --env-file .env python -m briefing.cli fork <run_id> search_plan '{"search_plan": {"year_from": 2024}}'
```

## The research plan

| # | Step | Kind | Output |
|---|---|---|---|
| 1 | scope | LLM | Canonical condition, synonyms, MeSH terms, subtopics |
| 2 | search_plan | LLM | PubMed queries (Q1..) and ClinicalTrials.gov queries (T1..), plus user-added queries |
| 3 | retrieve | code | Each query's URL, time, hit count and returned IDs; source records (abstracts, trial registry data) |
| 4 | screen_extract | LLM, one call per source | Include or exclude with a reason; evidence items (E001..) with verbatim quotes |
| 5 | aggregate | code | Included trials by phase, status, intervention class and sponsor |
| 6 | synthesize | LLM | Report; each statement (S1..) cites evidence IDs and is marked `evidence` or `inference` |
| 7 | verify | code + LLM | Quote found in source text (exact, near with the source passage, missing); citations exist; sponsors match trials; an LLM support check rates each statement against its quotes and trial records |

Each step reads earlier outputs from the store and writes one record:
`<run_id>/steps/<NN>_<step>.json` with `params`, `outputs`, `summary` and `meta` (duration, models,
prompt id and hash, token counts).

## Re-running from a step

A finished run is never changed. A re-run creates a child run: steps before the fork point are
copied (marked `inherited`) and the rest run again with the new parameters. The fork point is the
earlier of the requested step and the first step whose parameters changed. The chat assistant does
this through its `rerun` tool and then reports a diff (sources added and removed, sections changed,
verification results). The sidebar shows the lineage of the current run.

Parameters per step are listed in `default_params()` in `briefing/pipeline.py`. Examples:

- "Only 2024 onwards" changes `search_plan.year_from` and starts at search_plan.
- "Drop PMID:123" changes `screen_extract.exclude_ids` and starts at screen_extract.
- "Focus implications on rural hospitals" changes `synthesize.instructions` and starts at synthesize.

## Provenance in the report and in chat

The report carries numbered source citations, statement IDs, an analysis marker on inference
statements, a verification banner, the query table with hit counts and retrieval times, and a
table of steps with models and prompt versions. The provenance bundle download is a zip of the run
record, all step records and the report.

The assistant uses these tools: `get_plan`, `get_step`, `get_source`, `trace` (statement,
evidence or player to quotes, sources, queries and screening reasons), `search`, `rerun`,
`diff_runs`, `list_runs`. It is told to answer from tool results only.

## Layout

```
briefing/
  config.py    settings from environment variables
  sources.py   PubMed E-utilities and ClinicalTrials.gov v2 clients
  llm.py       OpenAI structured-output wrapper with usage accounting
  store.py     run store: file:// (local) or gs:// (Cloud Storage)
  pipeline.py  steps, prompts, run creation, fork, execute, diff
  report.py    Markdown report and provenance bundle
  agent.py     chat assistant (OpenAI Responses API with function tools)
  ui.py        Streamlit app
  cli.py       command line runner
```

## GCP

The container is stateless and listens on `$PORT`. Configuration and secrets come from environment
variables. Runs go to Cloud Storage when `STORE_URL=gs://bucket/prefix`; the service account
credentials come from the Cloud Run metadata server. `docker compose --profile gcs up` runs a
second copy of the app against a local GCS emulator on port 8091 to test that path.

Planned deployment:

- Artifact Registry repository; image built with `gcloud builds submit --tag`.
- Cloud Storage bucket for runs.
- Secret Manager secret for `OPENAI_API_KEY`, mounted as an environment variable.
- Cloud Run service with its own service account (`roles/storage.objectAdmin` on the bucket,
  `roles/secretmanager.secretAccessor` on the secret), session affinity on (Streamlit uses a
  websocket per session), request timeout 3600 s, private ingress with IAM or IAP.

A run executes inside the user's Streamlit session. If the browser disconnects, the run stops and
is marked `running` in the store. For production the pipeline would move to a Cloud Run job
(`python -m briefing.cli run ...` already works as its entry point) with status read from the store.

## Tests

Both scripts run against real data and services.

```bash
# Browser scenarios in the order a user meets them (app must be running).
# Opening, citations, new research, assistant question, assistant re-run, run switching,
# plan/evidence tabs, downloads. --skip-llm leaves out the three scenarios that call OpenAI.
uv run python tests/ui_smoke.py --condition migraine

# Independent checks of a finished run, sharing no code with the app: re-fetches cited sources
# live and searches the quotes, checks sponsors, date range and phase counts, and asks a
# different model whether each statement follows from its quotes.
uv run --env-file .env python tests/check_run.py <run_id> [--no-judge]
```

## Limits of the demo

- Sources are abstracts and registry records only. No full text, no paywalled guidelines
  (NCCN, ADA Standards text and similar).
- The support check is itself an LLM judgment. Statements it rates "partly supported" are marked
  in the report with the reason; "not supported" ones are flagged.
- A question sent while the assistant is still answering interrupts the earlier one.
- A re-run of synthesize rewrites the whole report, so the diff reports changed sections, not
  changed sentences.
- No authentication or multi-user separation.
