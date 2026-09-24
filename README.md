# POD_Verifier

The POD quality scorer: an EfficientNet model in a container Lambda that looks at every
proof-of-delivery photo a rider uploads and says whether it actually shows the parcel, at
the door, legibly.

- **[`docs/architecture.html`](docs/architecture.html)** — the whole system, end to end. Start here.
- **[`aws/README.md`](aws/README.md)** — how the function works and how to trigger it.
- **[`aws/DEPLOYMENT.md`](aws/DEPLOYMENT.md)** — CloudFormation + Docker/ECR (no SAM, no CDK).

---

## The one thing to know

**This service owns no data.** It has no database driver — the image build fails if one
reappears in `requirements.txt`. It asks **sarathy** what to score over HTTP and posts
back what it found, so sarathy stays the only writer to its own schema.

```
                      ┌──────────────────────────────────────┐
  rider app ──────────▶ sarathy                              │
   (URLs, not bytes)  │   trip.pod, pod_scores, the schedule  │
                      └───────────▲───────────────┬──────────┘
                                  │ HTTP          │ async invoke
                                  │               ▼
                      ┌───────────┴──────────────────────────┐
                      │ POD_Verifier (this repo)             │
                      │   download → EfficientNet → results  │
                      └──────────────────────────────────────┘
                                  │
                                  ▼  images only
                                 S3
```

Four endpoints, all VPC-internal, all namespaced `/internal/pod-scoring`:

| | |
|---|---|
| `GET /trips/{tripId}` | one trip's links + which already carry a score |
| `GET /trips` | a keyset page of trips in a window |
| `GET /retry-links` | links whose download failed and are due another attempt |
| `POST /scores` | upsert a batch of results |

`SARATHY_BASE_URL` must not resolve publicly — `provision-stack.sh` refuses to deploy
one that does. There is no app-level token; network isolation is the whole of the
protection.

---

## Three ways in

All three converge on `score_and_record`: download a window of images concurrently, run
inference in batches, post every outcome back. A failed download is recorded as a row,
never dropped.

**Per trip** — the latency-sensitive path, and the reason a small pool of containers is
kept warm. Sarathy invokes asynchronously from exactly two places, in this order:

1. **A POD verification request** (`/trip/send-pod-verification`). A rider without the
   delivery OTP cannot close the trip themselves, so they ask a TL to check the photo. The
   trip is **still open** and a human is about to approve or reject it — the score needs to
   be on screen when they do. This is the one that matters for latency.
2. **Trip completion** (`/app/complete`), which the app calls once the TL approves. It is
   also the *only* per-trip score for the ordinary flow, where the rider has the OTP and
   never raises a request at all.

```json
{"trip_id": 50645055, "pod_links": ["https://…"], "source": "sarathy.sendPodVerificationRequest"}
```

Sarathy skips the invoke when every link already carries a score, so in the verification
flow completion is normally a no-op. It fires when the POD changed in between — a TL
rejected, the rider re-uploaded — which is exactly when the earlier score no longer
applies.

**The sweep is the safety net, but only for completed trips**: it selects on
`status = 'COMPLETED'`. A trip waiting on a TL is not in its scope, so a dropped
verification-request invoke means that TL decides without a score. Nothing is lost
permanently — completion or the sweep catches it afterwards — but it will not show up as
missing rows.

**The sweep, every 30 minutes** — an empty event scores everything updated in the last
`SWEEP_LOOKBACK_HOURS` (26), then runs the retry pass.

```json
{}
```

26 hours rather than "today" on purpose. A run at 23:30 asking for today covers to 23:30,
and every trip completing before midnight would fall into no run at all. A rolling window
has no such seam, costs nothing extra — sarathy reports what already carries a score —
and re-covers a run that failed or was throttled.

**Manual** — named dates score exactly those days and skip the retry pass. The retry
event reaches links the schedule can no longer see: past the 24-hour cutoff, or out of
attempts.

```json
{"start_date": "2026-09-01", "end_date": "2026-09-09"}
{"retry_failed": true}
{"retry_failed": {"since": "2026-09-01T00:00:00Z", "until": "2026-09-10T00:00:00Z"}}
```

---

## The upload race

This shapes more of the code than anything else, so it is worth stating plainly.

`trip.pod` is filled from presigned URLs the **rider's app generated**, recorded verbatim
the moment it calls `/app/save-info`. Nothing checks that the object reached S3. The
bytes follow on whatever connection the rider has. So when the scorer asks S3 for the
image, it is quite often not there yet — and essentially every `download_failed` row in
`pod_scores` is that gap rather than a dead link.

**In the invocation.** A 403, a 404 or a suspiciously short body is retried at +5s then
+15s, bounded by the invocation's deadline (`DOWNLOAD_RETRY_DELAYS`). A timeout or a 5xx
is *not* retried in line — that is the host being unwell, and the HTTP adapter already
handles it at the socket level.

A flat 5–10 second sleep on every invocation was considered and rejected: it would pay
the wait for the ~90% of images already present, and delay every rider's answer to help
the few that are not.

**Across runs.** Sarathy owns the schedule — attempts at +30m, 1h, 2h, 4h, 8h, 16h,
stopping 24 hours after the first failure, which is when the app's own background upload
task abandons the image. The scorer just walks what `/retry-links` hands back and reports
what recovered:

```
POD retry sweep: attempted=412 recovered=96 still_failed=316 force=false
```

---

## Scoring

Four sigmoid heads — `context_valid`, `package_visible`, `label_readable`,
`image_clarity` — combined by fixed weights into `pod_score`.

- **`IMAGENET_NORMALIZE=true` is required.** The model was trained with ImageNet
  normalization; disabling it is the correctness bug this pipeline was rebuilt to fix.
- **`FLAG_THRESHOLD`** defaults to `0.7`. The gold-set evaluation favours 0.55–0.60 for
  precision-first penalising. If you change it, change the matching `0.7` in sarathy's
  `pod_scores_flagged` view too.
- Download failures are excluded from PASS/FLAG. A fetch that failed is our problem, not
  the rider's, and an AWB is never penalised for it.

---

## Repository layout

```
aws/
  lambda_scorer/
    handler.py            routing, the three paths, download retry, continuation
    sarathy_client.py     the only route to any data
    src/model.py          MultiHeadEfficientNet
    model/best.pt         the canonical checkpoint (see .gitignore exception)
    tests/                146 tests, 100% coverage on handler + client
  infra/stack.yaml        CloudFormation — 17 resources, 24 parameters
  deploy.sh               ECR repo + build + push
  provision-stack.sh      cloudformation deploy, with the public-URL guard
  teardown.sh             destroy an environment
docs/architecture.html    the whole system, end to end
```

Everything tunable is a Lambda environment variable set by `stack.yaml` /
`provision-stack.sh`. `config.env.example` carries names and placeholders only — never
commit real values.

---

## Tests

```bash
cd aws/lambda_scorer && python -m pytest tests/ -q
```

146 tests, 100% coverage on `handler.py` and `sarathy_client.py`. They stand a fake
`SarathyClient` in front of the handler and assert on what it was asked for and what it
was handed — that is the whole of the service boundary, so it is the whole of what the
tests need to pin down. One torch-gated test exercises the real inference wiring.

---

## Deploying

Merge to `stag-main` for staging (automatic) or `prod-main` for production (one
approval). CircleCI creates everything from nothing — ECR repository, CloudFormation
stack, Lambda, schedules, alarms.

**Order matters when both sides change:** sarathy first, migration and API in the same
artifact, then this repo. The scorer calls `/retry-links`, which does not exist until
sarathy ships. Keep the EventBridge schedule disabled until both are up.

See **[`aws/DEPLOYMENT.md`](aws/DEPLOYMENT.md)**.

---

## The trained checkpoint

`aws/lambda_scorer/model/best.pt` is canonical and lives in Git (see the `.gitignore`
exception). It is baked into the image at build time and verified to load with zero key
mismatches before the image is pushed.

```bash
cp /path/to/your/trained/best.pt aws/lambda_scorer/model/best.pt
git add aws/lambda_scorer/model/best.pt && git commit -m "Update trained checkpoint"
```

Use Git LFS or a private artifact store if the file grows large.
