# POD scoring pipeline — AWS

Single container Lambda that scores Proof-of-Delivery photos for quality and flags
bad PODs for the operations team.

**Trigger: event-driven, one trip per invocation.** When a rider raises a POD
verification request, Sarathy (`SpmdTripService.sendPodVerificationRequest`)
asynchronously invokes this Lambda with `{"trip_id": N, "pod_links": [...]}`; the
handler resolves that trip's POD images, scores them and upserts into `pod_scores`.
The old daily EventBridge sweep is still in the template but **DISABLED** — it is
now only a backfill tool.

**No SAM.** Infra is plain **CloudFormation** ([`infra/stack.yaml`](infra/stack.yaml))
applied with [`provision-stack.sh`](provision-stack.sh); image delivery is
[`deploy.sh`](deploy.sh). Full runbook: [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Two modes, one function

| Event | Mode | Behaviour |
|---|---|---|
| `{"trip_id": 12345, "pod_links": [...]}` | **single trip** (live path) | Resolves that trip's PODs via `TRIP_QUERY` (parameterised on `trip_id`), falling back to the `pod_links` carried in the event. Scores + upserts in one pass — no windowing, no continuation. Links this trip has already scored are skipped. |
| `{"start_date": "…", "end_date": "…"}` | **batch, date range** | `RANGE_QUERY` with both dates bound as parameters. |
| `{"query": "SELECT …"}` | **batch, ad-hoc SQL** | Validated as a single read-only `SELECT`/`WITH`, then run as-is. |
| `{}` | **batch, default** | The original whole-dataset run over `SOURCE_QUERY`, unchanged. |
| `{"warmup": true}` | **warm ping** | Loads the checkpoint and returns. Scores nothing. |

Envelopes are unwrapped, so an EventBridge `detail` or a single-record SQS body
carrying the same JSON routes to the single-trip path too.

### Single-trip contract

```json
{"trip_id": 12345, "pod_links": ["https://.../a.jpg"], "awb": "ABC123"}
```

`pod_links` and `awb` are optional. `TRIP_QUERY` wins when it returns rows (it
carries the real AWB); the event payload is the fallback for the common case where
Sarathy fires the instant the request is raised, before the source row exists. With
no AWB from either side, rows are keyed `TRIP-<trip_id>`.

### Re-requests: only new work

A rider who re-raises a request has usually replaced *some* of the photos. The
handler asks `pod_scores` which of this trip's links are already `status='scored'`
(within `RESCORE_LOOKBACK_DAYS`) and downloads only the rest — so an unchanged
photo is never scored twice, and a replaced one always is. The check is keyed on
the **link**, not the trip: the trip is not a stable unit of "done" precisely
because its links change. All links already scored ⇒ `status: "already_scored"`,
zero downloads, zero writes.

### Staying warm under load

A cold container pays for the image pull, the torch import and the checkpoint
load. Riders trigger scoring at unpredictable times, so that cost is paid on a
schedule instead of on the request:

- **Warm-container reuse** — the model, the Postgres connection and the HTTP
  connection pool are module globals, built once and reused by every later
  invocation that lands on the same container. The DB handle is health-checked
  (`SELECT 1`) and silently redialled if the server hung up.
- **Warm pool** — `WarmupSchedule` pings every 5 minutes with
  `{"warmup": true, "fanout": N}`. Lambda routes concurrent invocations to
  *different* containers, so the ping self-invokes `N-1` times and holds each
  container briefly; without the hold they would all be served by one container.
- **Concurrency** — several trips scored at once simply run in parallel
  containers, each warm or warming. `ReservedConcurrency` caps that fan-out (and
  therefore the load on the source image host and the DB);
  `TRIP_MAX_DOWNLOAD_WORKERS` (8) keeps a single trip from spending 64 threads on
  its handful of images.

`WarmPoolSize` must stay below `ReservedConcurrency` — warm pings hold slots too.
For a hard latency floor rather than a best-effort one, put **provisioned
concurrency** on an alias instead; it removes cold starts entirely but bills for
the reserved 10 GB around the clock, which is why the warmer is the default.

### Batch architecture (backfill path, unchanged)

One EventBridge trigger (empty event `{}`) runs one invocation that processes the
entire day in bounded-memory windows and is safe against the 15-minute wall:

- **Concurrent downloads** — `ThreadPoolExecutor`; the download, not the model, was
  the original bottleneck.
- **Bounded memory** — one `WINDOW_SIZE` of images in memory at a time; flat regardless
  of dataset size.
- **Resume-from-checkpoint** — scored rows in Postgres are the checkpoint; a retry
  processes only the remainder.
- **Idempotent upsert** — unique `(awb, pod_link, run_date)`; retries fill gaps, never dup.
- **Every input gets an outcome** — download failures are recorded (`status='download_failed'`),
  never silently dropped and never penalised.
- **Clock-aware continuation** — near the wall it flushes and queues exactly one
  checkpointed continuation (bounded by `MAX_CONTINUATIONS`); async retries + SQS DLQ
  back it up.

## Configuration

Runtime env (set by CloudFormation, not baked into code): **single-trip source**
(`TRIP_QUERY` — must keep the named `%(trip_id)s` placeholder; it is bound as a
query parameter, never string-interpolated), **batch source**
(`SOURCE_QUERY` — the SQL that returns the day's POD rows), **pipeline tuning**
(`MAX_DOWNLOAD_WORKERS`, `WINDOW_SIZE`, `INFERENCE_BATCH_SIZE`), **scoring**
(`FLAG_THRESHOLD`, `IMAGENET_NORMALIZE`), **batch triggers** (`RANGE_QUERY` —
keep the `%(start_date)s` / `%(end_date)s` placeholders — and `ALLOW_ADHOC_QUERY`),
**re-request de-dup** (`RESCORE_LOOKBACK_DAYS`), **warm pool** (`WARM_FANOUT`,
`WARM_HOLD_SECONDS`, `TRIP_MAX_DOWNLOAD_WORKERS`), **resilience**
(`CONTINUATION_SAFETY_MS`, `MAX_CONTINUATIONS`), and **Postgres** (`PG_*`). See [`config.env.example`](config.env.example);
use Secrets Manager / CI secrets for real passwords.

> **Do not disable `IMAGENET_NORMALIZE`.** The model was trained with ImageNet
> normalization; scoring without it collapses recall.

## Layout

- `lambda_scorer/` — Docker image: `Dockerfile`, `handler.py`, `model/best.pt`, `src/`, `tests/`.
- `infra/` — `stack.yaml` (CloudFormation) and `schema.sql` (with the idempotency key).
- `eval/` — `evaluate_model.py` + `run_gold_eval.sh`: measure the model against a human gold set.
- `deploy.sh` — create the ECR repo if absent, stage `infra/schema.sql` into the build context, build the CPU image, push (`DRY_RUN`/`SKIP_LAMBDA_UPDATE` supported).
- `provision-stack.sh` — `aws cloudformation deploy` (VPC, Lambda, schedules, DLQ, IAM, log group, alarms, env).
- `teardown.sh` — destroy an environment, including dropping `pod_scores`.
- `DEPLOYMENT.md` — step-by-step deploy runbook.

## Model weights (`lambda_scorer/model/best.pt`)

Trained `MultiHeadEfficientNet` (EfficientNet-B0 backbone, 4 attribute heads) checkpoint,
tracked in Git and copied into the image at `/opt/model/best.pt`. The Docker build
verifies it loads into the architecture with 0 key mismatches.

## Tests

```bash
cd aws/lambda_scorer && python3 -m venv .venv && source .venv/bin/activate \
  && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements.txt -r requirements-dev.txt \
  && pytest tests/ -v
```

## Build smoke-test (no AWS)

```bash
cd aws && DRY_RUN=true ./deploy.sh
```

## Triggering

Live path — Sarathy invokes it (`InvocationType=Event`) on each POD verification
request. Manually, the same thing:

```bash
aws lambda invoke --function-name pod-pipeline-stg \
  --invocation-type Event \
  --payload '{"trip_id": 12345}' /dev/null
```

Batch runs are normally started from Sarathy's admin API — one selection per
request, either a date range or a read-only query:

```bash
curl -X POST "$SARATHY/trip/pod-verification/trigger-scoring" \
  -H 'Content-Type: application/json' \
  -d '{"startDate": "2026-08-01", "endDate": "2026-08-20"}'

curl -X POST "$SARATHY/trip/pod-verification/trigger-scoring" \
  -H 'Content-Type: application/json' \
  -d '{"query": "SELECT awb, trip_id, pod FROM pod_manual_verification WHERE node_id = 42"}'
```

The API validates the selection and invokes the Lambda asynchronously, so `200`
means *accepted*, not *scored*. The same guard runs again inside the Lambda —
`ALLOW_ADHOC_QUERY=false` refuses raw SQL outright, whatever the caller sends.

Backfill sweep (the disabled daily schedule):

```bash
aws scheduler update-schedule --name pod-scoring-daily-stg --state ENABLED   # or:
aws lambda invoke --function-name pod-pipeline-stg --payload '{}' out.json
```

The schedule ships as `State: DISABLED` (cron `25 18` UTC / 23:55 IST) in `stack.yaml`.

## Bootstrapping

Merging to `stag-main` (staging) or `prod-main` (production, one approval) takes
an empty account to a working service: CI creates the ECR repository, pushes the
image, creates the stack, then invokes `{"migrate": true}` so the function
applies `infra/schema.sql` from inside the VPC — CircleCI cannot reach a private
RDS, but the Lambda already lives there. `aws/teardown.sh` reverses it.

## Who may invoke

Sarathy's IAM principal needs `lambda:InvokeFunction` on the function. Either grant
it in Sarathy's own IAM policy, or pass `INVOKER_PRINCIPAL_ARNS` to
`provision-stack.sh` to add a resource-based permission here.
