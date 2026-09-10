# POD scoring pipeline — AWS

Single container Lambda that scores Proof-of-Delivery photos for quality and flags
bad PODs for the operations team.

**Trigger: event-driven, one trip per invocation.** When a rider raises a POD
verification request, Sarathy (`SpmdTripService.sendPodVerificationRequest`)
asynchronously invokes this Lambda with `{"trip_id": N, "pod_links": [...]}`; the
handler resolves that trip's POD images, scores them and posts the results back.
The old daily EventBridge sweep is still in the template but **DISABLED** — it is
now only a backfill tool.

**This function holds no database credentials and issues no SQL.** `pod_scores`
is sarathy's table, created by sarathy's Flyway migration `V198__pod_scores.sql`.
The scorer asks sarathy's `/internal/pod-scoring` API what to score and posts
back what it found. Two services owning one service's schema was the problem
that closed: sarathy is the only writer.

**No SAM.** Infra is plain **CloudFormation** ([`infra/stack.yaml`](infra/stack.yaml))
applied with [`provision-stack.sh`](provision-stack.sh); image delivery is
[`deploy.sh`](deploy.sh). Full runbook: [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Two modes, one function

| Event | Mode | Behaviour |
|---|---|---|
| `{"trip_id": 12345, "pod_links": [...]}` | **single trip** (live path) | Scores the `pod_links` carried in the event, asking sarathy for them when it carries none. Scores + posts in one pass — no windowing, no continuation. Links this trip has already scored are skipped. |
| `{"start_date": "…", "end_date": "…"}` | **batch, date range** | Pages `GET /internal/pod-scoring/trips` over the range. |
| `{}` | **batch, default** | Today's PODs — what the disabled schedule sends. |
| `{"warmup": true}` | **warm ping** | Loads the checkpoint and returns. Scores nothing. |

Envelopes are unwrapped, so an EventBridge `detail` or a single-record SQS body
carrying the same JSON routes to the single-trip path too.

### Single-trip contract

```json
{"trip_id": 12345, "pod_links": ["https://.../a.jpg"], "awb": "ABC123"}
```

`pod_links` and `awb` are optional, but the event **wins when it carries links** —
they are the trip as it was at the instant the rider raised the request, which is
the freshest view there is. Sarathy is asked when the event carries no links (a
console test event, say) and **also when it carries no AWB**, because a
`TRIP-<id>` placeholder is not an AWB and nothing downstream can substitute for
the real one. With no AWB from either side, rows fall back to `TRIP-<trip_id>`.

If sarathy is unreachable and the event carried links, the run proceeds on those
links alone; if it carried none, the invocation fails with `502` rather than
recording a half-empty result.

### Re-requests: only new work

A rider who re-raises a request has usually replaced *some* of the photos.
Sarathy returns, alongside the trip's links, the subset that already carries a
score (within `pod.scoring.rescore-lookback-days`, its own setting), and the
handler downloads only the rest — so an unchanged photo is never scored twice,
and a replaced one always is. Deciding that on the side that owns the data is
the point: it is one query there and no round trip here. The check is keyed on
the **link**, not the trip: the trip is not a stable unit of "done" precisely
because its links change. All links already scored ⇒ `status: "already_scored"`,
zero downloads, zero writes.

### Staying warm under load

A cold container pays for the image pull, the torch import and the checkpoint
load. Riders trigger scoring at unpredictable times, so that cost is paid on a
schedule instead of on the request:

- **Warm-container reuse** — the model, the image-download connection pool and
  the sarathy client's own pool are module globals, built once and reused by
  every later invocation that lands on the same container.
- **Warm pool** — `WarmupSchedule` pings every 5 minutes with
  `{"warmup": true, "fanout": N}`. Lambda routes concurrent invocations to
  *different* containers, so the ping self-invokes `N-1` times and holds each
  container briefly; without the hold they would all be served by one container.
- **Concurrency** — several trips scored at once simply run in parallel
  containers, each warm or warming. `ReservedConcurrency` caps that fan-out (and
  therefore the load on the source image host and on sarathy);
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
- **Resume-from-checkpoint** — the `alreadyScoredLinks` sarathy returns are the
  checkpoint; re-paging the range costs a query and skips the work already done.
- **Idempotent upsert** — sarathy upserts on `(awb, pod_link, run_date)`; retries
  fill gaps, never duplicate.
- **Every input gets an outcome** — download failures are recorded (`status='download_failed'`),
  never silently dropped and never penalised.
- **Clock-aware continuation** — near the wall it flushes and queues exactly one
  checkpointed continuation (bounded by `MAX_CONTINUATIONS`); async retries + SQS DLQ
  back it up.

## Configuration

Runtime env (set by CloudFormation, not baked into code): **sarathy**
(`SARATHY_BASE_URL` — required; `SARATHY_TIMEOUT`, `SARATHY_RETRIES`,
`SARATHY_PAGE_SIZE`), **pipeline tuning** (`MAX_DOWNLOAD_WORKERS`, `WINDOW_SIZE`,
`INFERENCE_BATCH_SIZE`), **scoring** (`FLAG_THRESHOLD`, `IMAGENET_NORMALIZE`),
**warm pool** (`WARM_FANOUT`, `WARM_HOLD_SECONDS`, `TRIP_MAX_DOWNLOAD_WORKERS`),
and **resilience** (`CONTINUATION_SAFETY_MS`, `MAX_CONTINUATIONS`). See
[`config.env.example`](config.env.example).

`SARATHY_BASE_URL` must resolve and route from the Lambda's subnets and **must
not be publicly reachable**: `/internal/pod-scoring` is protected by network
isolation alone and carries no app-level token, in the same way as the rest of
sarathy's service-to-service surface. There are no database variables to set —
the de-dup window lives on sarathy as `pod.scoring.rescore-lookback-days`.

> **Do not disable `IMAGENET_NORMALIZE`.** The model was trained with ImageNet
> normalization; scoring without it collapses recall.

## Layout

- `lambda_scorer/` — Docker image: `Dockerfile`, `handler.py`, `model/best.pt`, `src/`, `tests/`.
- `infra/` — `stack.yaml` (CloudFormation) and `ci-policy.json` (the CI user's IAM policy).
- `eval/` — `evaluate_model.py` + `run_gold_eval.sh`: measure the model against a human gold set.
- `deploy.sh` — create the ECR repo if absent, build the CPU image, push (`DRY_RUN`/`SKIP_LAMBDA_UPDATE` supported).
- `provision-stack.sh` — `aws cloudformation deploy` (VPC, Lambda, schedules, DLQ, IAM, log group, alarms, env).
- `teardown.sh` — destroy an environment's AWS resources. It leaves `pod_scores` alone; that table is sarathy's.
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

The suite stands a fake `SarathyClient` in front of the handler, so it needs no
network and no database. CI runs it inside the built image, so pytest sees the
same pinned torch/timm/cv2 stack that ships.

## Build smoke-test (no AWS)

```bash
cd aws && DRY_RUN=true ./deploy.sh
```

## Triggering

Every mode is one `aws lambda invoke`. Two flags matter:
`--cli-binary-format raw-in-base64-out` is required on AWS CLI v2, and
`--invocation-type Event` makes it fire-and-forget. Without `Event` the CLI
waits for the response — fine for a trip or a warm ping, but a batch sweep can
run for 15 minutes and the CLI gives up after 60 seconds, so **always use
`Event` for batch** (or add `--cli-read-timeout 0`).

```bash
FN=pod-pipeline-stg; R=us-east-2
inv() { aws lambda invoke --function-name "$FN" --region "$R" \
          --cli-binary-format raw-in-base64-out --payload "$1" "${2:-/dev/stdout}"; }
```

**One trip** — the live path, identical to what Sarathy sends:

```bash
inv '{"trip_id": 12345}'
# links optional; supplied when the source row does not exist yet
inv '{"trip_id": 12345, "awb": "ABC123", "pod_links": ["https://.../a.jpg"]}'
```

Returns `status: complete`, or `already_scored` when sarathy reports every link
as already scored. Re-running is safe — it only downloads
links that are new or changed.

**A date range** (async — this can be long):

```bash
aws lambda invoke --function-name "$FN" --region "$R" --invocation-type Event \
  --cli-binary-format raw-in-base64-out \
  --payload '{"start_date": "2026-09-01", "end_date": "2026-09-09"}' /dev/null
```

**Today's PODs** — what the disabled schedule sends:

```bash
aws lambda invoke --function-name "$FN" --region "$R" --invocation-type Event \
  --cli-binary-format raw-in-base64-out --payload '{}' /dev/null
```

**Housekeeping:**

```bash
inv '{"warmup": true, "fanout": 1}'   # load the model, score nothing
```

There is no migrate event. The schema is sarathy's, and sarathy's Flyway
migration creates it.

### From the console

[`events/`](events/) holds a saved payload per mode — score one trip, backfill a
range or a day, warm up. Save them once as *shareable* test events
on the function and anyone with console access can run one from the Test tab, no
terminal or local credentials needed. `events/README.md` covers the setup and
the two gotchas (the console invokes synchronously, and the payloads are
identical across environments).

### Without AWS credentials

Sarathy's admin API covers the batch modes, for people who should not have
deploy keys. A `200` means *accepted*, not *scored*.

```bash
curl -X POST "$SARATHY/trip/pod-verification/trigger-scoring" \
  -H 'Content-Type: application/json' \
  -d '{"startDate": "2026-09-01", "endDate": "2026-09-09"}'
```

Raising a POD verification request the normal way (`/trip/send-pod-verification`)
also triggers scoring for that trip — that is the live path.

### Seeing what happened

An async invoke returns nothing, so read the outcome from the table or the logs:

```sql
SELECT status, count(*), max(scored_at)
FROM pod_scores WHERE run_date = CURRENT_DATE GROUP BY status;
```

```bash
aws logs tail "/aws/lambda/$FN" --region "$R" --since 15m --follow
```

### On a schedule

The backfill sweep exists but ships **DISABLED** — per-trip events are the live
trigger. Its time is set in IST (`BackfillSchedule`, default `cron(15 0 * * ? *)`
in `Asia/Kolkata` — 00:15 daily), and `BackfillState=ENABLED` turns it on. Read
the caveats above the resource in `stack.yaml` first: the batch path's resume
checkpoint is per `run_date`, so a nightly sweep re-scores links that per-trip
runs already covered on earlier days.

## Where the data lives

In **sarathy**, and only sarathy reaches it. This project has no database
credentials, no driver and no SQL; everything crosses the boundary as HTTP:

| Endpoint | Used for |
|---|---|
| `GET /internal/pod-scoring/trips/{tripId}` | one trip's POD links + which already carry a score |
| `GET /internal/pod-scoring/trips?startDate&endDate&cursor&limit` | a page of trips for a backfill, keyset-paginated on trip id |
| `POST /internal/pod-scoring/scores` | write a batch of scores; idempotent on `(awb, podLink, runDate)` |

Sarathy answers those from its own `trip` table and owns `pod_scores` through
Flyway migration `V198__pod_scores.sql`. The endpoints are namespaced
`/internal` and reachable only inside the VPC — they carry no app-level token,
so exposing them publicly would expose an unauthenticated write.

### Why the reads look the way they do on sarathy's side

**The `trip` table, not `kaptaan`.** `kaptaan` is a derived, analytics-shaped
table rebuilt by a pipeline, so it lags — and on staging that pipeline does not
run at all, leaving it empty. Sarathy reads the live `trip` table for both the
single-trip and the range query, dated on `COALESCE(closed_at, updated_at)` —
when the trip reached its final state, which is when its POD was captured.
`created_at` would date a trip to when it was *planned*, pulling in trips whose
photos do not exist yet.

**Never filter on `kaptaan.metadata`.** The column is there, but the prod
pipeline stopped populating it, so `metadata->>'podVerificationStatus'` matches
nothing — a query using it returns zero rows and reads as "no PODs to score"
rather than as a broken query, the worst kind of failure. Verification status
lives on sarathy's `trip.metadata`, written by `sendPodVerificationRequest`.

## Bootstrapping

Merging to `stag-main` (staging) or `prod-main` (production, one approval) takes
an empty account to a working service: CI creates the ECR repository, pushes the
image, creates the stack and smoke-tests the live function. There is no schema
step — `pod_scores` arrives with sarathy's own deploy, which is also why CircleCI
never needed database access here. `aws/teardown.sh` reverses the AWS side.

## Who may invoke

Sarathy's IAM principal needs `lambda:InvokeFunction` on the function. Either grant
it in Sarathy's own IAM policy, or pass `INVOKER_PRINCIPAL_ARNS` to
`provision-stack.sh` to add a resource-based permission here.
